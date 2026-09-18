"""Phase 2: turn each raw patch notes article into readable, section-tagged text.

Each article's `body` is HTML with a flat two-level heading structure: <h1> marks
a top-level category (e.g. "New Updates", "Other Events"), <h2> marks a topic or
entity within it (e.g. a class group, a boss, a feature). That structure varies
in granularity between patches (some use many <h1>s, some lump everything under
one), so section/subsection are read from whatever headings are actually present
rather than assumed.
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

BLOCK_TAGS = ["h1", "h2", "p", "li", "table", "blockquote"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_sections(body_html: str) -> list[dict]:
    soup = BeautifulSoup(body_html, "html.parser")
    for tag in soup.find_all(["img", "script", "style"]):
        tag.decompose()

    sections = []
    current_section = None
    current_subsection = None
    current_texts: list[str] = []

    def flush():
        text = "\n".join(current_texts).strip()
        if current_section is not None and text:
            sections.append(
                {"section": current_section, "subsection": current_subsection, "text": text}
            )
        current_texts.clear()

    for tag in soup.find_all(BLOCK_TAGS):
        # get_text() recurses into descendants, so a block tag nested inside
        # another matched block tag (e.g. <li> inside a <table>, or a nested
        # <ul><li>) would otherwise be captured twice: once via its own match,
        # once already folded into its ancestor's text.
        if any(parent.name in BLOCK_TAGS for parent in tag.parents):
            continue
        text = normalize_whitespace(tag.get_text(separator=" ", strip=True))
        if not text:
            continue
        if tag.name == "h1":
            flush()
            current_section = text
            current_subsection = None
        elif tag.name == "h2":
            flush()
            current_subsection = text
        else:
            current_texts.append(text)
    flush()

    return sections


def clean_article(raw_path: Path) -> dict:
    article = json.loads(raw_path.read_text(encoding="utf-8"))
    sections = split_into_sections(article["body"])
    return {
        "id": article["id"],
        "name": article["name"],
        "liveDate": article["liveDate"],
        "sections": sections,
    }


def render_readable_text(cleaned: dict) -> str:
    lines = [cleaned["name"], cleaned["liveDate"], ""]
    for entry in cleaned["sections"]:
        heading = entry["section"]
        if entry["subsection"]:
            heading += f" > {entry['subsection']}"
        lines.append(f"## {heading}")
        lines.append(entry["text"])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW_DIR / "index.json").read_text(encoding="utf-8"))

    print(f"{'version':<7} {'sections':<9} {'chars':<8} name")
    for entry in manifest:
        raw_path = RAW_DIR / f"{entry['id']}.json"
        cleaned = clean_article(raw_path)

        json_path = PROCESSED_DIR / f"{entry['id']}.json"
        json_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

        txt_path = PROCESSED_DIR / f"{entry['id']}.txt"
        readable = render_readable_text(cleaned)
        txt_path.write_text(readable, encoding="utf-8")

        total_chars = sum(len(s["text"]) for s in cleaned["sections"])
        version = f"v.{entry['version']}" if entry["version"] else "v.?"
        print(f"{version:<7} {len(cleaned['sections']):<9} {total_chars:<8} {entry['name']}")


if __name__ == "__main__":
    main()
