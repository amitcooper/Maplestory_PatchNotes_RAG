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
# Split across however many patches survive filtering (see per_patch_limit).
# Needs to be generous: with a couple of patches, dividing a too-small
# budget in half was enough to push a real fact (an item's own drop chunk,
# ranked outside the per-patch top few by raw distance) back out of the
# context -- confirmed by watching a real answer regress when a second,
# weaker patch got admitted alongside the dominant one. 32 keeps the
# many-patches case unchanged (~4/patch at MAX_PATCHES=8) while giving a
# dominant single patch room for its full relevant chunk set even when 1-2
# weaker patches ride along with it.
MAX_TOTAL_CHUNKS = 32
RELEVANCE_MARGIN = 0.3  # see the entity-filter patch-pruning note in retrieve()

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

# A "what new X were added" question needs every matching patch, not just
# whichever chunks happen to rank highest by embedding similarity -- the
# signal (a handful of chunks out of ~3000) is too sparse for semantic search
# to reliably surface (confirmed: it was missing real new-job releases like
# Sia Astelle and Ren). Patch notes consistently title these sections
# "New Job: X" / "New Boss: X" / "New Area: X", so a direct heading scan is
# both more precise and more complete than semantic ranking for this case.
CONTENT_TYPE_KEYWORDS = {
    "job": "Job", "class": "Job", "classes": "Job", "jobs": "Job",
    "boss": "Boss", "bosses": "Boss",
    "area": "Area", "areas": "Area", "map": "Area", "maps": "Area", "region": "Area",
}
NEW_CONTENT_RE = re.compile(
    r"\bnew\s+(" + "|".join(CONTENT_TYPE_KEYWORDS) + r")\b", re.IGNORECASE
)


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

    # Collect every known entity mentioned, not just the first -- a query like
    # "what did Malefic Star and Jupiter drop" names two bosses at once, and
    # stopping at the first match silently dropped the second one entirely.
    entities = []
    query_lower = query.lower()
    for name in KNOWN_ENTITIES:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", query_lower):
            entities.append(name)
    if entities:
        description_parts.append(f"entities={', '.join(entities)}")

    content_type = None
    content_match = NEW_CONTENT_RE.search(query)
    if content_match:
        content_type = CONTENT_TYPE_KEYWORDS[content_match.group(1).lower()]
        description_parts.append(f"new-content-type={content_type}")

    return {
        "date_min": date_min,
        "date_max": date_max,
        "version_eq": version_eq,
        "entities": entities,
        "content_type": content_type,
        "description": "; ".join(description_parts) or "none",
    }


def load_version_dates() -> dict[int, str]:
    manifest = json.loads((RAW_DIR / "index.json").read_text(encoding="utf-8"))
    return {entry["version"]: entry["liveDate"][:10] for entry in manifest}


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)


def find_new_content(collection: chromadb.Collection, content_type: str, parsed: dict) -> dict:
    # Deterministic path for "what new jobs/bosses/areas were added": scans
    # every chunk's subsection heading for the "New <Type>: <Name>" pattern
    # patch notes consistently use, rather than relying on semantic ranking
    # to surface a handful of relevant chunks out of ~3000.
    heading_re = re.compile(r"^(?:\[[^\]]*\]\s*)?new\s+" + re.escape(content_type) + r"\s*:", re.IGNORECASE)
    all_items = collection.get(include=["documents", "metadatas"])

    # Group by (version, subsection) rather than just version: a single patch
    # can introduce more than one of the same content type (v.269 has both
    # "New Boss: Malefic Star" and "New Boss: Kai") -- capping per patch alone
    # let whichever one's chunks came first in the scan crowd out the other
    # entirely once the cap was hit.
    by_subsection: dict[tuple[int, str], list[dict]] = {}
    for doc, meta in zip(all_items["documents"], all_items["metadatas"]):
        if not meta["subsection"] or not heading_re.match(meta["subsection"]):
            continue
        if parsed["date_min"] and meta["date"] < parsed["date_min"]:
            continue
        if parsed["date_max"] and meta["date"] >= parsed["date_max"]:
            continue
        if parsed["version_eq"] and meta["version"] != parsed["version_eq"]:
            continue
        key = (meta["version"], meta["subsection"])
        by_subsection.setdefault(key, []).append({"text": doc, "meta": meta})

    # Some patches split a "New X: Name" announcement across many small
    # sub-headings (Quick Overview / Skills / Equipment), others put the
    # whole thing under one heading -- cap each entry's own chunk count so
    # one heavily-detailed entry can't flood the answer either.
    CHUNKS_PER_ENTRY = 4

    by_patch: dict[int, list[dict]] = {}
    for (version, _subsection), chunks in by_subsection.items():
        by_patch.setdefault(version, []).extend(chunks[:CHUNKS_PER_ENTRY])

    patches = []
    for version, chunks in by_patch.items():
        patches.append(
            {
                "version": version,
                "patch_name": chunks[0]["meta"]["patch_name"],
                "date": chunks[0]["meta"]["date"],
                "source_url": chunks[0]["meta"]["source_url"],
                "chunks": chunks,
            }
        )
    patches.sort(key=lambda p: p["date"], reverse=True)
    return patches


def retrieve(
    query: str,
    collection: chromadb.Collection,
    version_dates: dict[int, str],
    semantic_query: str | None = None,
    use_reranker: bool = False,
) -> dict:
    # semantic_query lets a caller (e.g. eval.py testing query rewriting) swap
    # in a different string for the embedding search while entity/date parsing
    # still runs against the user's original wording.
    parsed = parse_query(query, version_dates)

    if parsed["content_type"]:
        patches = find_new_content(collection, parsed["content_type"], parsed)
        return {"query": query, "filter_description": parsed["description"], "patches": patches}

    has_filter = any([parsed["date_min"], parsed["date_max"], parsed["version_eq"], parsed["entities"]])

    # Scan the whole collection whenever a filter is active, so the metadata
    # post-filter below can't lose eligible chunks to a truncated top-k --
    # the corpus is small enough (a few thousand chunks) that this is cheap.
    n_results = collection.count() if has_filter else max(CHUNKS_PER_PATCH * MAX_PATCHES, 20)

    result = collection.query(query_texts=[semantic_query or query], n_results=n_results)

    candidates = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        if parsed["date_min"] and meta["date"] < parsed["date_min"]:
            continue
        if parsed["date_max"] and meta["date"] >= parsed["date_max"]:
            continue
        if parsed["version_eq"] and meta["version"] != parsed["version_eq"]:
            continue
        if parsed["entities"] and not any(e in meta["entity_mentioned"] for e in parsed["entities"]):
            continue
        candidates.append({"text": doc, "meta": meta, "distance": dist})

    if use_reranker:
        from rerank import rerank

        candidates = rerank(semantic_query or query, candidates)

    by_patch: dict[int, list[dict]] = {}
    for c in candidates:
        by_patch.setdefault(c["meta"]["version"], []).append(c)

    def relevance(c: dict) -> float:
        # Higher is better either way: rerank_score already is, distance is
        # inverted since it's a similarity *distance* (lower = more similar).
        return c["rerank_score"] if "rerank_score" in c else -c["distance"]

    if parsed["entities"] and len(by_patch) > 1:
        # A short boss/class name (e.g. "Baldrix") also shows up as one row in
        # unrelated multi-boss reward tables in other patches ("Defeat Baldrix
        # for N points", alongside a dozen other bosses) -- entity_mentioned
        # correctly tags those too, but they're not actually *about* Baldrix,
        # and letting them count as "matching patches" was diluting the
        # per-patch chunk budget so thin that the one patch that actually
        # introduces Baldrix (and lists what it drops) lost its own relevant
        # chunks to the cap. Keep a patch only if it's a strong match for AT
        # LEAST ONE requested entity -- judged per-entity (each against its
        # own best match), not globally, so asking about two bosses at once
        # doesn't let one's stronger embedding match crowd out the other's
        # own home patch just because it scores a bit worse in isolation.
        #
        # MARGIN=0.3 is a real trade-off, not a clean answer: a class with
        # genuine recurring content (e.g. Kaiser, balance-tweaked across many
        # patches) forms a tight distance cluster ~0.75-0.83 with one
        # standout best match ~0.58 -- and a genuinely single-patch entity's
        # residual incidental mentions can land in that exact same 0.75-0.83
        # range. There's no distance value that admits one cluster and
        # excludes the other; a tighter margin (0.15) was tested and starved
        # multi-patch entities down to a single result. This value favors not
        # losing real multi-patch history over perfectly excluding a
        # possible extra incidental patch or two.
        keep_versions: set[int] = set()
        for entity_name in parsed["entities"]:
            entity_by_patch: dict[int, list[dict]] = {}
            for c in candidates:
                if entity_name in c["meta"]["entity_mentioned"]:
                    entity_by_patch.setdefault(c["meta"]["version"], []).append(c)
            if not entity_by_patch:
                continue
            best_for_entity = max(relevance(c) for chunks in entity_by_patch.values() for c in chunks)
            for version, chunks in entity_by_patch.items():
                if max(relevance(c) for c in chunks) >= best_for_entity - RELEVANCE_MARGIN:
                    keep_versions.add(version)
        by_patch = {version: chunks for version, chunks in by_patch.items() if version in keep_versions}

    # When few patches match, let each keep more chunks instead of a flat cap --
    # e.g. a single-version query like "v.271" was losing specific facts (a
    # boss's phase-3 mechanic, say) that ranked #4 within that one patch's
    # candidates purely because CHUNKS_PER_PATCH was a flat 3 regardless of
    # how many other patches were competing for context-window budget.
    per_patch_limit = max(CHUNKS_PER_PATCH, MAX_TOTAL_CHUNKS // max(1, len(by_patch)))

    patches = []
    for version, chunks in by_patch.items():
        chunks = chunks[:per_patch_limit]  # already ranked (reranked if use_reranker)
        patches.append(
            {
                "version": version,
                "patch_name": chunks[0]["meta"]["patch_name"],
                "date": chunks[0]["meta"]["date"],
                "source_url": chunks[0]["meta"]["source_url"],
                "chunks": chunks,
            }
        )

    # Most-recent-first: a changelog reads newest-to-oldest, and that's what
    # someone asking "what's changed since X" actually wants to see first.
    is_time_query = bool((parsed["date_min"] or parsed["date_max"]) and parsed["entities"])
    if is_time_query:
        patches.sort(key=lambda p: p["date"], reverse=True)
    else:
        patches.sort(key=lambda p: -sum(relevance(c) for c in p["chunks"]))
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
            if "rerank_score" in c:
                score = f"rerank {c['rerank_score']:.3f}"
            elif "distance" in c:
                score = f"distance {c['distance']:.3f}"
            else:
                score = "heading match"
            print(f"   [{heading}] ({score})")
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
