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

Runs a hand-built set of 12 questions -- 10 answerable (facts verified against the actual scraped patches) plus 2 negative controls (a real class confirmed, by scanning every chunk's metadata, to have zero changes in the specific version/range asked about) -- through 4 retrieval configs -- baseline, +cross-encoder reranking, +LLM query rewriting, +both -- and reports mean retrieval recall and negative-control clean rate for each, a per-question breakdown for the winning config, then a full-generation pass on that config checking answer-keyword hits, negative-control declines, and citation validity (every `v.NNN` an answer cites must be a version actually in its retrieved context -- checked mechanically, not trusted).

Recall alone only measures whether answerable questions get answered; it says nothing about whether an unanswerable one gets correctly declined instead of a fabricated answer. That's what the negative controls and citation check are for. Answering "no" here isn't an LLM judgment call to get right or wrong: `get_answer()` (`src/generate.py`) short-circuits to a fixed decline message whenever `retrieve()` returns zero patches, with no LLM call in that path at all -- so the negative-control check is really testing retrieval precision (did metadata filtering correctly find nothing to hallucinate from?), not the model's honesty.

**Finding:** on this corpus, plain semantic + metadata-filtered retrieval (the Phase 5 baseline) already scores highest (1.00 mean recall across all 4 configs on the current test set -- see below). LLM query rewriting is not clearly better than the baseline here -- it tends to paraphrase away from the patch notes' exact terminology, which the baseline already matches well -- so it stays opt-in rather than on by default. Cross-encoder reranking (`src/rerank.py`) had *measurably hurt* recall (1.00 -> 0.86, and 0.76 combined with rewriting) until a real bug was found and fixed: the multi-patch clustering step's `RELEVANCE_MARGIN` constant was tuned against embedding-distance's scale (~0.55-0.85) but was being applied unchanged to the cross-encoder's raw logit scores (~-11 to -4) whenever reranking was on, so it rejected nearly every patch but the single top scorer. With that fixed, reranking is neutral on this test set -- it still reorders chunks within a kept patch, it just no longer breaks which patches get kept. Both remain opt-in parameters on `retrieve()` (`use_reranker`, `semantic_query`) since neither shows a clear win here, not because either is broken.

Diagnosing answer misses along the way led to two more fixes: `retrieve()`'s per-patch chunk limit now scales up when a query narrows to few patches (a flat cap was dropping relevant detail on narrow single-version queries), and `generate.py`'s `build_context` now caps total excerpt size (a "what new content" question's deterministic path had no per-patch chunk cap and could assemble a context large enough to exceed Groq's free-tier 8000-tokens/minute request limit, a hard 413).

## Phase 8: Interface

```
python src/cli.py
```

Loads the embedding model, vector store, and Groq client once, then answers questions in a loop instead of paying that startup cost on every invocation like `generate.py` does. Type `exit` or Ctrl+C to quit.

## Tests

```
pytest tests/
```

Unit tests for the most silently-regressable pieces of logic: `parse_query`'s date/version/entity/content-type parsing (`src/retrieve.py`), `find_entities`/`paragraphs_to_chunks`'s entity-tagging edge cases and paragraph-packing rule (`src/chunk.py`), and `eval.py`'s own scoring functions (`retrieval_recall`, `retrieval_is_clean`, `answer_keyword_hit_rate`, `citation_versions_valid`) -- the eval harness's correctness matters as much as the pipeline's. No network, embedding model, or vector store required.
