"""Tests for llm_cache.py's tiny disk-backed cache. Redirects CACHE_PATH to
a pytest tmp_path so these never touch the real data/cache/llm_cache.json.
"""

import llm_cache


def test_get_on_empty_cache_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "CACHE_PATH", tmp_path / "cache.json")
    assert llm_cache.get(("model", "prompt", "question")) is None


def test_put_then_get_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "CACHE_PATH", tmp_path / "cache.json")
    key = ("openai/gpt-oss-120b", "system prompt", "What changed for Kaiser?", "context text")
    llm_cache.put(key, "Kaiser got a buff in v.269.")
    assert llm_cache.get(key) == "Kaiser got a buff in v.269."


def test_different_key_parts_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "CACHE_PATH", tmp_path / "cache.json")
    llm_cache.put(("model", "prompt", "question A"), "answer A")
    llm_cache.put(("model", "prompt", "question B"), "answer B")
    assert llm_cache.get(("model", "prompt", "question A")) == "answer A"
    assert llm_cache.get(("model", "prompt", "question B")) == "answer B"


def test_changing_any_key_part_invalidates_the_entry():
    # A prompt or model change must produce a different key -- serving a
    # stale answer written under an old prompt is the one failure mode
    # this cache must never have.
    key_v1 = ("model", "old system prompt", "question")
    key_v2 = ("model", "new system prompt", "question")
    assert llm_cache._cache_key(key_v1) != llm_cache._cache_key(key_v2)


def test_corrupt_cache_file_is_treated_as_empty_not_raised(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(llm_cache, "CACHE_PATH", cache_path)
    assert llm_cache.get(("model", "prompt", "question")) is None


def test_persists_across_separate_load_calls(tmp_path, monkeypatch):
    # put() writes to disk immediately (no in-memory-only state) -- a
    # second process (e.g. a fresh `python src/eval.py` run) must see it.
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(llm_cache, "CACHE_PATH", cache_path)
    llm_cache.put(("model", "prompt", "question"), "cached answer")
    assert cache_path.exists()
    # Reload from disk explicitly, simulating a new process's first read.
    assert llm_cache._load() == {llm_cache._cache_key(("model", "prompt", "question")): "cached answer"}
