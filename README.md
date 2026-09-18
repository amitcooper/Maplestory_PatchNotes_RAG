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
