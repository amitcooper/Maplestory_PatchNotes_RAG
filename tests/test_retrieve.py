"""Tests for retrieve.py's parse_query: the regex-based date/version/entity/
content-type parsing that drives metadata filtering. A fixed `today` is passed
on every call so "since June"-style relative phrasing resolves deterministically
instead of depending on the day the tests happen to run.
"""

from datetime import date

from retrieve import parse_query

TODAY = date(2026, 9, 22)
VERSION_DATES = {267: "2026-03-17", 269: "2026-06-16", 271: "2026-09-08"}


def test_since_month_sets_date_min_and_no_date_max():
    parsed = parse_query("what changed for Kaiser since March 2026", VERSION_DATES, TODAY)
    assert parsed["date_min"] == "2026-03-01"
    assert parsed["date_max"] is None
    assert parsed["entities"] == ["Kaiser"]


def test_since_month_without_year_assumes_most_recent_past_occurrence():
    # "today" is September 2026, so a bare "since June" means June 2026, not 2025.
    parsed = parse_query("what changed since June", VERSION_DATES, TODAY)
    assert parsed["date_min"] == "2026-06-01"


def test_since_version_uses_that_versions_live_date():
    parsed = parse_query("what changed since v.269", VERSION_DATES, TODAY)
    assert parsed["date_min"] == VERSION_DATES[269]
    assert parsed["version_eq"] is None


def test_before_month_sets_date_max_and_no_date_min():
    parsed = parse_query("what changed before June", VERSION_DATES, TODAY)
    assert parsed["date_max"] == "2026-06-01"
    assert parsed["date_min"] is None


def test_in_month_sets_a_bounded_range():
    parsed = parse_query("what changed in June", VERSION_DATES, TODAY)
    assert parsed["date_min"] == "2026-06-01"
    assert parsed["date_max"] == "2026-07-01"


def test_bare_version_sets_version_eq_not_date_range():
    parsed = parse_query("what changed for Kaiser in v.271", VERSION_DATES, TODAY)
    assert parsed["version_eq"] == 271
    assert parsed["date_min"] is None
    assert parsed["date_max"] is None


def test_unknown_version_is_ignored():
    parsed = parse_query("what changed in v.999", VERSION_DATES, TODAY)
    assert parsed["version_eq"] is None


def test_multiple_entities_are_all_collected():
    parsed = parse_query("what did Malefic Star and Jupiter drop", VERSION_DATES, TODAY)
    assert set(parsed["entities"]) == {"Malefic Star", "Jupiter"}


def test_content_type_keywords_map_to_canonical_type():
    assert parse_query("what new bosses were added", VERSION_DATES, TODAY)["content_type"] == "Boss"
    assert parse_query("what new jobs came out", VERSION_DATES, TODAY)["content_type"] == "Job"
    assert parse_query("any new maps this patch", VERSION_DATES, TODAY)["content_type"] == "Area"


def test_content_type_detected_without_new_immediately_before_keyword():
    # Regression: "new" and the content-type word don't have to be adjacent.
    # This exact phrasing used to fall through to unfiltered semantic search
    # and miss real additions (Baldrix, Malefic Star, Kai, Jupiter).
    parsed = parse_query("which bosses added to the game recently?", VERSION_DATES, TODAY)
    assert parsed["content_type"] == "Boss"


def test_content_type_not_triggered_by_type_word_alone():
    # "class" with no newness word should not be treated as a new-content query.
    parsed = parse_query("were there any class changes", VERSION_DATES, TODAY)
    assert parsed["content_type"] is None


def test_query_with_no_filters_returns_all_none():
    parsed = parse_query("how is the game doing lately", VERSION_DATES, TODAY)
    assert parsed["date_min"] is None
    assert parsed["date_max"] is None
    assert parsed["version_eq"] is None
    assert parsed["entities"] == []
    assert parsed["content_type"] is None
    assert parsed["description"] == "none"
