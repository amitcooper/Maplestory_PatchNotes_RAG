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


def find_entities(text: str) -> list[str]:
    # Case-sensitive on purpose: proper nouns are capitalized in patch notes
    # prose, which is what keeps a common word like "Will" (the boss) from
    # matching every sentence that uses "will" as an auxiliary verb.
    return [name for name, pattern in ENTITY_PATTERNS if pattern.search(text)]


def paragraphs_to_chunks(paragraphs: list[str], max_chars: int) -> list[str]:
    chunks = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current and current_len + len(para) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 1
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
