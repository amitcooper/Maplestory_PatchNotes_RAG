"""Phase 3: split each cleaned section into retrieval-sized chunks with metadata.

Each chunk gets: date, patch_name, version, section, subsection, entity_mentioned,
source_url. entity_mentioned is matched against a fixed roster of class and boss
names read directly off the v.271 patch notes (its Job Skill Changes and Boss/Mob
sections list the full current roster) -- it's necessarily incomplete for anything
introduced before/after that snapshot, but gives real entity filtering for v1
rather than none. Extending the roster as gaps are noticed is expected upkeep.
"""

import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

MAX_CHUNK_CHARS = 800

CLASSES = [
    "Hero", "Paladin", "Dark Knight",
    "Arch Mage (Fire/Poison)", "Arch Mage (Ice/Lightning)", "Bishop",
    "Bow Master", "Marksman", "Pathfinder",
    "Night Lord", "Shadower", "Dual Blade",
    "Buccaneer", "Corsair", "Cannoneer",
    "Wind Archer", "Dawn Warrior", "Night Walker", "Blaze Wizard", "Thunder Breaker", "Mihile",
    "Mercedes", "Aran", "Phantom", "Luminous", "Evan", "Shade",
    "Wild Hunter", "Battle Mage", "Mechanic", "Blaster", "Xenon",
    "Demon Avenger", "Demon Slayer",
    "Kaiser", "Angelic Buster", "Cadena", "Kain",
    "Zero", "Kinesis",
    "Illium", "Ark", "Adele", "Khali",
    "Lynn", "Mo Xuan",
    "Kanna", "Hayato",
    "Hoyoung", "Lara", "Ren",
    "Sia Astelle", "Erel Light",
]

BOSSES = [
    "Crimson Queen", "Vellum", "Lotus", "Damien", "Will", "Chosen Seren",
    "Black Mage", "Kalos the Guardian", "Kaling", "First Adversary",
    "Malefic Star", "Limbo", "Baldrix", "Jupiter", "Kai",
]

# Longest names first so e.g. "Kalos the Guardian" matches before a shorter
# entity name that happens to be its substring would ever get the chance to.
KNOWN_ENTITIES = sorted(set(CLASSES + BOSSES), key=len, reverse=True)
ENTITY_PATTERNS = [(name, re.compile(r"\b" + re.escape(name) + r"\b")) for name in KNOWN_ENTITIES]


def clean_patch_name(raw_name: str) -> str:
    name = re.sub(r"^\[.*?\]\s*", "", raw_name)
    name = re.sub(r"\s*patch notes\s*$", "", name, flags=re.IGNORECASE)
    return name.strip(" -")


# Some entity names double as ordinary phrasing in boilerplate eligibility
# clauses -- "Zero characters that completed Story Quest Act 2" appears in
# nearly every event's requirements regardless of whether that event has
# anything to do with the Zero class. Verified against the actual data: of
# 367 chunks mentioning "Zero", 325 (89%) were exactly this pattern; the
# other 42 are genuine content ("Zero's Ego Weapon", "HEXA Wind Cutter" for
# Zero, etc.), which phrase it differently and so aren't caught by this.
ENTITY_EXCLUDE_SUFFIX = {
    # No leading ^ -- .match(text, pos) already anchors to start exactly at
    # pos; a ^ here would (counterintuitively) only ever match at absolute
    # position 0 of the string, so this exclusion would silently never fire.
    "Zero": re.compile(r"\s+[Cc]haracters?\b"),
}


# A chunk that names more than this many distinct entities isn't really
# "about" any single one of them -- it's a reference table (e.g. a boss
# reward/difficulty table spanning a dozen bosses) or, just as often, several
# different classes' one-line bug fixes that landed in the same chunk purely
# because they're adjacent paragraphs under a generic heading like "General
# Skill Bug Fixes" and fit the same size budget, not because they're related.
# Confirmed on the real data: one v.257 chunk tags 29 different classes
# (Kaiser among them) because it's a cross-class skill-name reference table,
# and this was measurably diluting entity-filtered retrieval for classes like
# Kaiser that also have genuine, specific per-patch skill changes elsewhere.
MAX_ENTITIES_PER_CHUNK = 4


def find_entities(text: str) -> list[str]:
    # Case-sensitive on purpose: proper nouns are capitalized in patch notes
    # prose, which is what keeps a common word like "Will" (the boss) from
    # matching every sentence that uses "will" as an auxiliary verb.
    found = []
    for name, pattern in ENTITY_PATTERNS:
        exclude_suffix = ENTITY_EXCLUDE_SUFFIX.get(name)
        for m in pattern.finditer(text):
            if exclude_suffix and exclude_suffix.match(text, m.end()):
                continue
            found.append(name)
            break

    if len(found) > MAX_ENTITIES_PER_CHUNK:
        return []
    return found


def paragraphs_to_chunks(paragraphs: list[str], max_chars: int) -> list[str]:
    # Packing by character budget alone isn't enough once table rows are
    # split one-class/boss-per-paragraph (see clean.py): several single-class
    # rows easily fit under max_chars together, but if each names a different
    # class, the packed chunk can still cross MAX_ENTITIES_PER_CHUNK -- which
    # then zeroes out entity tagging for the whole chunk, including whichever
    # real fact triggered this in the first place. Tracking the running
    # distinct-entity count during packing (not just after the fact) avoids
    # ever assembling a chunk that would trip that cap.
    chunks = []
    current: list[str] = []
    current_len = 0
    current_entities: set[str] = set()
    for para in paragraphs:
        para_entities = set(find_entities(para))
        exceeds_chars = current_len + len(para) + 1 > max_chars
        exceeds_entities = len(current_entities | para_entities) > MAX_ENTITIES_PER_CHUNK
        if current and (exceeds_chars or exceeds_entities):
            chunks.append("\n".join(current))
            current = []
            current_len = 0
            current_entities = set()
        current.append(para)
        current_len += len(para) + 1
        current_entities |= para_entities
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_chunks() -> list[dict]:
    manifest = json.loads((RAW_DIR / "index.json").read_text(encoding="utf-8"))
    all_chunks = []

    for entry in manifest:
        cleaned = json.loads((PROCESSED_DIR / f"{entry['id']}.json").read_text(encoding="utf-8"))
        date = entry["liveDate"][:10]
        patch_name = clean_patch_name(entry["name"])

        chunk_num = 0
        for section_entry in cleaned["sections"]:
            paragraphs = section_entry["text"].split("\n")
            # The subsection heading itself (e.g. "New Boss: Baldrix", "New
            # Job: Ren") often carries the strongest topical signal in the
            # whole chunk, but clean.py stores it only in metadata, not in
            # the body text -- so it was never part of what got embedded.
            # A query like "what bosses were added" needs it IN the text to
            # match at all; prepending it here is what actually fixes that.
            heading_prefix = f"{section_entry['subsection']}\n" if section_entry["subsection"] else ""
            for chunk_text in paragraphs_to_chunks(paragraphs, MAX_CHUNK_CHARS):
                chunk_text = heading_prefix + chunk_text
                entities = find_entities(chunk_text)
                if section_entry["subsection"]:
                    for name in find_entities(section_entry["subsection"]):
                        if name not in entities:
                            entities.append(name)

                chunk_num += 1
                all_chunks.append(
                    {
                        "chunk_id": f"{entry['id']}-{chunk_num}",
                        "text": chunk_text,
                        "date": date,
                        "patch_name": patch_name,
                        "version": entry["version"],
                        "section": section_entry["section"],
                        "subsection": section_entry["subsection"],
                        "entity_mentioned": ", ".join(entities),
                        "source_url": entry["source_url"],
                    }
                )

    return all_chunks


def print_summary(chunks: list[dict]) -> None:
    total = len(chunks)
    avg_len = sum(len(c["text"]) for c in chunks) / total
    with_entity = sum(1 for c in chunks if c["entity_mentioned"])

    print(f"{total} chunks written to {CHUNKS_PATH}")
    print(f"average chunk length: {avg_len:.0f} chars")
    print(f"chunks with at least one identified entity: {with_entity} ({100 * with_entity / total:.0f}%)")
    print()

    counts = Counter(c["version"] for c in chunks)
    for version in sorted(counts):
        print(f"  v.{version}: {counts[version]} chunks")


def main() -> None:
    chunks = build_chunks()
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(chunks)


if __name__ == "__main__":
    main()
