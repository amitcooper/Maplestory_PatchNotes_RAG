"""Phase 1: fetch MapleStory patch notes and save them locally.

Data source: the public JSON CMS API that backs the maplestory.nexon.net /
www.nexon.com news pages (not documented, but public, unauthenticated, and
unrestricted by robots.txt on either domain).
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

LIST_URL = "https://g.nexonstatic.com/maplestory/cms/v1/news"
ARTICLE_URL = "https://g.nexonstatic.com/maplestory/cms/v1/news/{id}"
ARTICLE_PAGE_URL = "https://www.nexon.com/maplestory/news/update/{id}/{slug}"
HEADERS = {"User-Agent": "MaplePatchNotesRag/0.1 (personal research project)"}
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
VERSION_RE = re.compile(r"v\.(\d+)")


def get_with_retries(url: str) -> requests.Response:
    # Large article bodies occasionally hit a mid-download connection reset
    # from the CDN; a short retry-with-backoff clears it without manual reruns.
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(REQUEST_DELAY_SECONDS * attempt * 2)


def slugify(name: str) -> str:
    # Nexon's slugs split camelCase words first (MapleStory -> Maple-Story),
    # then replace non-alphanumeric runs with a single hyphen. Reverse-engineered
    # from an observed article URL; must match exactly or citation links 404.
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", "-", name)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def fetch_patch_notes_list(count: int) -> list[dict]:
    resp = get_with_retries(LIST_URL)
    items = resp.json()
    patch_notes = [
        item
        for item in items
        if item.get("category") == "update" and "patch notes" in item.get("name", "").lower()
    ]
    patch_notes.sort(key=lambda item: item["liveDate"])
    return patch_notes[-count:]


def fetch_article(item_id: int) -> dict:
    return get_with_retries(ARTICLE_URL.format(id=item_id)).json()


def scrape(count: int, force: bool) -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for item in fetch_patch_notes_list(count):
        item_id = item["id"]
        raw_path = RAW_DIR / f"{item_id}.json"

        if raw_path.exists() and not force:
            article = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            article = fetch_article(item_id)
            raw_path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(REQUEST_DELAY_SECONDS)

        name = article["name"]
        version_match = VERSION_RE.search(name)
        version = int(version_match.group(1)) if version_match else None
        slug = slugify(name)

        manifest.append(
            {
                "id": item_id,
                "version": version,
                "name": name,
                "liveDate": article["liveDate"],
                "source_url": ARTICLE_PAGE_URL.format(id=item_id, slug=slug),
                "raw_path": str(raw_path.relative_to(RAW_DIR.parent.parent)),
            }
        )

    manifest.sort(key=lambda entry: entry["liveDate"])
    index_path = RAW_DIR / "index.json"
    index_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def print_summary(manifest: list[dict]) -> None:
    print(f"\n{len(manifest)} patch notes posts saved to {RAW_DIR}\n")
    for entry in manifest:
        date = entry["liveDate"][:10]
        version = f"v.{entry['version']}" if entry["version"] else "v.?"
        print(f"  {date}  {version:<7} {entry['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape MapleStory patch notes.")
    parser.add_argument("--count", type=int, default=15, help="number of most recent patch notes posts to fetch")
    parser.add_argument("--force", action="store_true", help="re-fetch even if a raw file already exists")
    args = parser.parse_args()

    manifest = scrape(args.count, args.force)
    print_summary(manifest)


if __name__ == "__main__":
    main()
