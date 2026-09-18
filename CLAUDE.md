# MapleStory Patch History RAG — Project Spec

## What This Project Is
A RAG system that answers questions about how MapleStory has changed over time, based only on official patch notes. The user asks things like "what changed for Bowmaster since June" or "when was this item introduced, and has it changed since," and the system retrieves and synthesizes the relevant patch note excerpts across multiple patches, with dates and sources cited.

This is **not** a general MapleStory Q&A bot. It does not answer "how do I beat this boss" or "what's the best build" from wiki/guide knowledge. It only knows what official patch notes say, and its core value is tracking change over time across many patches — something tedious to do by scrolling through patch notes manually.

## Goals
- Build a working, well-engineered RAG project
- Keep everything free: no paid APIs, no paid models
- Keep the dataset small enough to fully read and manually verify every answer
- Demonstrate real retrieval design (not just "call an embedding API"), specifically:
  - Metadata-based filtering (date, patch name, section, entity mentioned)
  - Multi-chunk retrieval across time for a single entity (not just single best-match retrieval)
- Include a basic evaluation step comparing retrieval/answer quality against a hand-written test set

## Explicitly Out of Scope (for now)
- maplestorywiki.net or any wiki source
- maplestory.fandom.com (legacy wiki)
- Hidden Street, orangemushroom.net
- Reddit (PRAW or otherwise)
- Multi-source conflict handling / synthesis across sources
- Router agents / LangGraph (deferred, not needed for this scope)

These may return as stretch goals later, but are not part of the initial build.

## Data Source
- **Source:** Official MapleStory patch notes, maplestory.nexon.net (News / Patch Notes section)
- **Scope:** Last ~12 months of patch notes, roughly 10-15 posts (major patches; minor hotfix posts optional/stretch)
- **Format:** Static HTML per post, consistent template across posts (title, date, sections like "Class Changes," "New Content," "Bug Fixes")
- **Access method:** Scraping with `requests` + `BeautifulSoup` (no official API exists for this data)

## Pipeline / Plan

1. **Scrape**
   - Pull the last ~15 patch note post URLs, dates, and raw HTML
   - Save raw HTML locally first, so later steps don't require re-hitting the site

2. **Clean & structure**
   - Strip HTML/images/ads, extract plain text
   - Split each post into its existing sections (Class Changes, New Content, Bug Fixes, etc.) — patch notes are already pre-organized this way

3. **Chunk**
   - Break each section into smaller chunks (a few paragraphs each)
   - Attach metadata to every chunk:
     ```json
     {
       "date": "YYYY-MM-DD",
       "patch_name": "string",
       "section": "string",
       "entity_mentioned": "string (class/item name, if extractable)",
       "source_url": "string"
     }
     ```

4. **Embed + store**
   - Multilingual/English sentence embeddings (free, local — sentence-transformers)
   - Vector store: Chroma

5. **Retrieval**
   - For entity + time-based queries (e.g., "what changed for Bowmaster since June"):
     - Pre-filter chunks by keyword match on entity name
     - Sort/filter by date range
     - Use semantic similarity as secondary ranking, not the only signal
   - For general queries, standard semantic retrieval applies

6. **Generation**
   - LLM: Groq API (free tier) as default; Ollama local as fallback/alternative
   - Answer should cite specific patch dates/names for any claim about change over time

7. **Evaluation**
   - Hand-built test set of questions with known correct answers (pulled from the actual scraped patches)
   - Evaluate retrieval accuracy (right chunks pulled?) and answer quality, informally or with RAGAS
   - Use this to justify/tune chunking and filtering decisions

8. **Interface**
   - Minimal — command-line loop or basic Streamlit app is sufficient for v1. Do not over-invest in UI.

## Tech Stack (finalized)
- Orchestration: LangChain
- Vector store: Chroma
- Embeddings: sentence-transformers (free, local)
- LLM: Groq API (Llama 3), Ollama as local alternative
- Retrieval techniques: metadata filtering, query rewriting, cross-encoder reranking
- Eval: hand-built test set, optionally RAGAS
- Version control: Git/GitHub, phase-by-phase commits

## Key Differentiators to Build Toward
- Metadata-driven filtering by date/patch/entity, not just semantic similarity
- Multi-chunk synthesis across time for a single entity (the "what changed since X" query type)
- A real evaluation step showing retrieval quality before/after tuning decisions

## Success Criteria for v1
- Successfully scraped and cleaned ~10-15 patch notes
- Can correctly answer "what changed for [known class] in patch [known date]" using only retrieved chunks
- Can correctly answer "what has changed for [known class] since [date]" by pulling and synthesizing across multiple patches
- Every answer cites the specific patch(es) it drew from
- A short eval writeup showing at least one chunking/retrieval decision that was tested and improved

## Working Method
- Use Claude Code in Plan Mode, phase by phase (scrape → clean → chunk → embed → retrieve → generate → eval → UI)
- Confirm each phase works before moving to the next
- Keep dataset small throughout v1; expanding the date range later is a simple re-run of the scraper, not a redesign
