# MapleStory Patch Notes RAG

A RAG system that answers questions about how MapleStory has changed over time, based only on official patch notes. See `CLAUDE.md` for the full project spec.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Phase 1: Scrape

```
python src/scrape.py
```

Fetches the last 15 major-version Patch Notes posts from MapleStory's public news CMS API and saves the raw JSON to `data/raw/`, plus a manifest at `data/raw/index.json`. Re-running skips posts already downloaded; pass `--force` to re-fetch everything.

## Phase 2: Clean

```
python src/clean.py
```

Splits each raw article's HTML into `(section, subsection, text)` blocks and writes both a structured `data/processed/{id}.json` and a readable `data/processed/{id}.txt` per patch.

## Phase 3: Chunk

```
python src/chunk.py
```

Splits each section's text into ~800-character chunks tagged with date/patch/section/entity/source metadata, written to `data/processed/chunks.json`.

## Phase 4: Embed

```
python src/embed.py
```

Embeds every chunk (`sentence-transformers`, `all-MiniLM-L6-v2`) into a local Chroma collection.

**Note:** the vector store is kept at `%LOCALAPPDATA%\MaplePatchNotesRag\chroma`, *not* inside this project folder. This folder is synced by OneDrive, and OneDrive's real-time sync corrupts Chroma's binary index files while they're being written. Keep the vector store outside any cloud-synced folder.

## Phase 5: Retrieve

```
python src/retrieve.py "what changed for Kaiser since March 2026"
```

Entity- and date-aware retrieval: recognizes class/boss names and phrases like "since June", "before March 2026", or "v.269", pre-filters chunks accordingly, then ranks semantically and groups results by patch.

## Phase 6: Generate

```
python src/generate.py "what changed for Kaiser since March 2026"
```

Runs Phase 5 retrieval, then asks a Groq-hosted model (`openai/gpt-oss-120b`) to answer using only the retrieved excerpts, citing the patch/date for every claim. Needs a free API key from [console.groq.com](https://console.groq.com) saved in `.env` as `GROQ_API_KEY=...`.

## Phase 7: Evaluate

```
python src/eval.py
```

Runs a hand-built set of 10 questions (facts verified against the actual scraped patches) through 4 retrieval configs -- baseline, +cross-encoder reranking, +LLM query rewriting, +both -- and reports mean retrieval recall for each, then a full-generation answer-quality check on the winning config.

**Finding:** on this corpus, plain semantic + metadata-filtered retrieval (the Phase 5 baseline) already scores highest. Cross-encoder reranking (`src/rerank.py`) is neutral. LLM query rewriting measurably *hurts* recall -- it tends to paraphrase away from the patch notes' exact terminology, which the baseline already matches well. Both remain available as opt-in parameters on `retrieve()` (`use_reranker`, `semantic_query`) rather than being turned on by default, since the evidence doesn't support it. Diagnosing two initial answer misses also led to a real fix: `retrieve()`'s per-patch chunk limit now scales up when a query narrows to few patches, instead of a flat cap that was dropping relevant detail on narrow single-version queries.

## Phase 8: Interface

```
python src/cli.py
```

Loads the embedding model, vector store, and Groq client once, then answers questions in a loop instead of paying that startup cost on every invocation like `generate.py` does. Type `exit` or Ctrl+C to quit.

## Tests

```
pytest tests/
```

Unit tests for the two most silently-regressable pieces of logic: `parse_query`'s date/version/entity/content-type parsing (`src/retrieve.py`), and `find_entities`/`paragraphs_to_chunks`'s entity-tagging edge cases and paragraph-packing rule (`src/chunk.py`). No network, embedding model, or vector store required.
