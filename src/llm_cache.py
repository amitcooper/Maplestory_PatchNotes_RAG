"""Tiny disk-backed cache for LLM calls.

Groq doesn't do response caching for you server-side (unlike, say,
Anthropic's prompt caching) -- there's nothing to lean on there, so
avoiding a repeated, byte-for-byte identical request is entirely on this
application. This matters most for eval.py, which resends the same
handful of questions against the same retrieved context on every run --
rerunning it during development bought zero new information for real
token spend, and was the direct cause of a Groq free-tier daily-limit
exhaustion during one debugging session.

The cache key always folds in the model name and full system prompt, not
just the question/context -- so changing either (a prompt tweak, a model
swap) invalidates every affected entry automatically instead of silently
serving an answer that was written under an old prompt.

Trade-off worth naming: generate_answer runs at temperature=0.2, so two
fresh calls with identical input could already return slightly different
(but similarly valid) phrasings. Caching pins the first answer as *the*
answer for that exact input going forward. For this project -- a
personal patch-notes QA tool where reproducibility and free-tier cost
matter more than sampling variety -- that's the right trade.
"""

import hashlib
import json
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "llm_cache.json"


def _cache_key(parts: tuple[str, ...]) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _load() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt or half-written cache file shouldn't break the app --
        # worst case is a few avoidable cache misses, not a crash.
        return {}


def get(parts: tuple[str, ...]) -> str | None:
    return _load().get(_cache_key(parts))


def put(parts: tuple[str, ...], value: str) -> None:
    cache = _load()
    cache[_cache_key(parts)] = value
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
