"""Phase 2: turn each raw patch notes article into readable, section-tagged text.

Each article's `body` is HTML with a flat two-level heading structure: <h1> marks
a top-level category (e.g. "New Updates", "Other Events"), <h2> marks a topic or
entity within it (e.g. a class group, a boss, a feature). That structure varies
in granularity between patches (some use many <h1>s, some lump everything under
one), so section/subsection are read from whatever headings are actually present
rather than assumed.
"""

import copy
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

BLOCK_TAGS = ["h1", "h2", "p", "li", "table", "blockquote"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def own_text_of_li(li_tag) -> str:
    # A <li> that itself wraps a nested <ul>/<ol> is a label (e.g. "Phase 3")
    # for a whole nested bullet list, not a leaf item -- get_text() on it
    # would otherwise flatten every nested bullet into one paragraph (the
    # same failure mode <table> had: see clean_article's table handling).
    # Stripping the nested list from a copy leaves just this li's own label
    # text; the nested <li>s are matched and emitted separately since nested
    # list items are no longer skipped as "already covered by an ancestor".
    if not li_tag.find(["ul", "ol"]):
        return normalize_whitespace(li_tag.get_text(separator=" ", strip=True))
    clone = copy.deepcopy(li_tag)
    for nested_list in clone.find_all(["ul", "ol"]):
        nested_list.decompose()
    return normalize_whitespace(clone.get_text(separator=" ", strip=True))


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
        # another matched block tag (e.g. <li> inside a <table>) would
        # otherwise be captured twice: once via its own match, once already
        # folded into its ancestor's text. <li> is exempted from this skip
        # (unless it's inside a <table>, already handled separately below):
        # a nested <ul><li> is used as a "Phase 3"-style label wrapping many
        # real bullet items, and skipping the nested items would flatten all
        # of them into the label's own text via get_text() recursion -- the
        # same failure mode as the unsplit <table> case (see below), just
        # via nested lists instead of table rows. own_text_of_li strips any
        # nested list out of the label's own text instead, so the nested
        # items can be matched and emitted as their own paragraphs here too.
        if tag.name == "li":
            if tag.find_parent("table"):
                continue
        elif any(parent.name in BLOCK_TAGS for parent in tag.parents):
            continue
        if tag.name == "table":
            # A single get_text() over the whole table flattens every row --
            # e.g. every class's Link Skill, or every boss's bug fix -- into
            # one giant paragraph. chunk.py's paragraph packing never splits
            # within a paragraph, so that became one oversized chunk naming
            # more entities than MAX_ENTITIES_PER_CHUNK allows, which zeroes
            # out its entity tag entirely -- silently dropping every real,
            # entity-specific fact the table contained from entity-filtered
            # retrieval (confirmed: this is exactly why a genuine v.269
            # Kaiser skill change inside a Link Skill Changes table never
            # matched a "Kaiser" query). One row per table is its own class's
            # or boss's row already -- emitting each as its own paragraph
            # lets normal chunking and entity-tagging work on it correctly.
            rows = tag.find_all("tr")
            if not rows:
                # No <tr> structure to split on (e.g. a layout-only table) --
                # fall back to the old whole-table behavior rather than
                # silently dropping its content.
                whole = normalize_whitespace(tag.get_text(separator=" ", strip=True))
                if whole:
                    current_texts.append(whole)
                continue
            for row in rows:
                row_text = normalize_whitespace(row.get_text(separator=" ", strip=True))
                if row_text:
                    current_texts.append(row_text)
            continue
        text = own_text_of_li(tag) if tag.name == "li" else normalize_whitespace(tag.get_text(separator=" ", strip=True))
        if not text:
            continue
        if tag.name == "h1":
            flush()
            current_section = text
            current_subsection = None
        elif tag.name == "h2":
            flush()
            # "Quick Overview" is a generic label reused for every new job's
            # (and boss's, event's, ...) at-a-glance stat/weapon/link-skill
            # table across the whole corpus -- it never names the entity
            # itself, unlike its preceding sibling heading ("New Job: Ren").
            # Resetting current_subsection to it would strip that entity's
            # name away right where it matters most: confirmed this is why
            # Ren's actual class/job/stat/weapon/link-skill table was never
            # tagged with "Ren" at all and so never matched a "Ren" query --
            # find_entities() had no chance, since neither the heading nor
            # the table's own body text ("Class Anima Job Warrior...") ever
            # says "Ren". Keeping the prior subsection preserves that context
            # for chunk.py's heading-prefix injection to carry into the text.
            if text != "Quick Overview":
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
