"""Phase 5: entity- and date-aware retrieval over the Chroma collection.

Implements the spec's core retrieval design: pre-filter by entity/date metadata,
rank the filtered set semantically, then group results by patch so a "what
changed since X" query returns a spread across patches instead of being
dominated by whichever single chunk happens to rank highest.
"""

import json
import os
import re
import sys
from datetime import date as Date
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunk import KNOWN_ENTITIES

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
# Must match embed.py: kept outside the OneDrive-synced project folder,
# since OneDrive corrupts Chroma's binary index files (see embed.py).
CHROMA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MaplePatchNotesRag" / "chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "maple_patch_notes"

CHUNKS_PER_PATCH = 3
MAX_PATCHES = 8

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

SINCE_MONTH_RE = re.compile(r"\b(since|after)\s+([a-zA-Z]+)\.?\s*(\d{4})?\b", re.IGNORECASE)
BEFORE_MONTH_RE = re.compile(r"\bbefore\s+([a-zA-Z]+)\.?\s*(\d{4})?\b", re.IGNORECASE)
IN_MONTH_RE = re.compile(r"\bin\s+([a-zA-Z]+)\.?\s*(\d{4})?\b", re.IGNORECASE)
SINCE_VERSION_RE = re.compile(r"\b(since|after)\s+v\.?\s?(\d{3})\b", re.IGNORECASE)
VERSION_RE = re.compile(r"\bv\.?\s?(\d{3})\b", re.IGNORECASE)


def resolve_month_year(month_name: str, year: str | None, today: Date) -> Date:
    month = MONTHS[month_name.lower()]
    if year:
        return Date(int(year), month, 1)
    # No year given: assume the most recent occurrence of that month at or
    # before today, since that's what a person casually means by "since June".
    return Date(today.year if month <= today.month else today.year - 1, month, 1)


def next_month(d: Date) -> Date:
    return Date(d.year + 1, 1, 1) if d.month == 12 else Date(d.year, d.month + 1, 1)


def parse_query(query: str, version_dates: dict[int, str], today: Date | None = None) -> dict:
    # Filtering is done as a Python post-filter (see retrieve()) rather than
    # via Chroma's `where` clause: this chromadb version's $gte/$lt operators
    # only accept int/float operands, not date strings, and the corpus is
    # small enough that scanning it in Python is simpler than working around that.
    today = today or Date.today()
    date_min = date_max = version_eq = None
    description_parts = []

    since_version = SINCE_VERSION_RE.search(query)
    since_month = SINCE_MONTH_RE.search(query)
    before_month = BEFORE_MONTH_RE.search(query)
    in_month = IN_MONTH_RE.search(query)
    bare_version = VERSION_RE.search(query)

    if since_version:
        version = int(since_version.group(2))
        if version in version_dates:
            date_min = version_dates[version]
            description_parts.append(f"since v.{version} ({date_min})")
    elif since_month and since_month.group(2).lower() in MONTHS:
        d = resolve_month_year(since_month.group(2), since_month.group(3), today)
        date_min = d.isoformat()
        description_parts.append(f"since {date_min}")
    elif in_month and in_month.group(1).lower() in MONTHS:
        d = resolve_month_year(in_month.group(1), in_month.group(2), today)
        date_min = d.isoformat()
        date_max = next_month(d).isoformat()
        description_parts.append(f"in {date_min[:7]}")
    elif before_month and before_month.group(1).lower() in MONTHS:
        d = resolve_month_year(before_month.group(1), before_month.group(2), today)
        date_max = d.isoformat()
        description_parts.append(f"before {date_max}")
    elif bare_version and not since_version:
        version = int(bare_version.group(1))
        if version in version_dates:
            version_eq = version
            description_parts.append(f"only v.{version}")

    entity = None
    query_lower = query.lower()
    for name in KNOWN_ENTITIES:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", query_lower):
            entity = name
            description_parts.append(f"entity={name}")
            break

    return {
        "date_min": date_min,
        "date_max": date_max,
        "version_eq": version_eq,
        "entity": entity,
        "description": "; ".join(description_parts) or "none",
    }


def load_version_dates() -> dict[int, str]:
    manifest = json.loads((RAW_DIR / "index.json").read_text(encoding="utf-8"))
    return {entry["version"]: entry["liveDate"][:10] for entry in manifest}


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve(query: str, collection: chromadb.Collection, version_dates: dict[int, str]) -> dict:
    parsed = parse_query(query, version_dates)
    has_filter = any([parsed["date_min"], parsed["date_max"], parsed["version_eq"], parsed["entity"]])

    # Scan the whole collection whenever a filter is active, so the metadata
    # post-filter below can't lose eligible chunks to a truncated top-k --
    # the corpus is small enough (a few thousand chunks) that this is cheap.
    n_results = collection.count() if has_filter else max(CHUNKS_PER_PATCH * MAX_PATCHES, 20)

    result = collection.query(query_texts=[query], n_results=n_results)

    candidates = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        if parsed["date_min"] and meta["date"] < parsed["date_min"]:
            continue
        if parsed["date_max"] and meta["date"] >= parsed["date_max"]:
            continue
        if parsed["version_eq"] and meta["version"] != parsed["version_eq"]:
            continue
        if parsed["entity"] and parsed["entity"] not in meta["entity_mentioned"]:
            continue
        candidates.append({"text": doc, "meta": meta, "distance": dist})

    by_patch: dict[int, list[dict]] = {}
    for c in candidates:
        by_patch.setdefault(c["meta"]["version"], []).append(c)

    patches = []
    for version, chunks in by_patch.items():
        chunks = chunks[:CHUNKS_PER_PATCH]  # already semantically ranked by chromadb
        patches.append(
            {
                "version": version,
                "patch_name": chunks[0]["meta"]["patch_name"],
                "date": chunks[0]["meta"]["date"],
                "source_url": chunks[0]["meta"]["source_url"],
                "chunks": chunks,
            }
        )

    is_time_query = (parsed["date_min"] or parsed["date_max"]) and parsed["entity"] is not None
    patches.sort(key=lambda p: p["date"] if is_time_query else -sum(c["distance"] for c in p["chunks"]))
    patches = patches[:MAX_PATCHES]

    return {"query": query, "filter_description": parsed["description"], "patches": patches}


def print_results(result: dict) -> None:
    print(f"Query: {result['query']!r}")
    print(f"Detected filters: {result['filter_description']}")
    if not result["patches"]:
        print("No matching chunks found.")
        return
    print(f"Results across {len(result['patches'])} patch(es):\n")
    for patch in result["patches"]:
        print(f"-- {patch['patch_name']} ({patch['date']}) -- {patch['source_url']}")
        for c in patch["chunks"]:
            heading = c["meta"]["section"]
            if c["meta"]["subsection"]:
                heading += f" > {c['meta']['subsection']}"
            print(f"   [{heading}] (distance {c['distance']:.3f})")
            preview = c["text"][:200].replace("\n", " ")
            print(f"     {preview}...")
        print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question here"')
        return
    query = " ".join(sys.argv[1:])

    collection = get_collection()
    version_dates = load_version_dates()
    result = retrieve(query, collection, version_dates)
    print_results(result)


if __name__ == "__main__":
    main()
