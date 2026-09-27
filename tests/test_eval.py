"""Tests for eval.py's pure scoring functions -- retrieval_recall,
retrieval_is_clean (the negative-control check), answer_keyword_hit_rate,
and citation_versions_valid. None of these touch the network, the embedding
model, or the vector store, so they're plain unit tests like the rest of
this suite.
"""

from eval import (
    answer_keyword_hit_rate,
    citation_versions_valid,
    retrieval_is_clean,
    retrieval_recall,
)


def _result(*versions: int) -> dict:
    return {"patches": [{"version": v, "chunks": []} for v in versions]}


def test_retrieval_recall_full_match():
    assert retrieval_recall(_result(269, 271), [269, 271]) == 1.0


def test_retrieval_recall_partial_match():
    assert retrieval_recall(_result(269), [267, 269, 271]) == 1.0 / 3


def test_retrieval_recall_no_expected_versions_returns_none():
    # No ground truth to check against -- not the same as a recall of 0.
    assert retrieval_recall(_result(269), []) is None


def test_retrieval_is_clean_true_when_nothing_retrieved():
    assert retrieval_is_clean(_result()) is True


def test_retrieval_is_clean_false_when_anything_retrieved():
    # Even one spurious match on a negative-control question is a real miss --
    # it's what would let get_answer() reach the LLM with irrelevant context
    # instead of deterministically declining.
    assert retrieval_is_clean(_result(271)) is False


def test_answer_keyword_hit_rate_all_present():
    assert answer_keyword_hit_rate("Iron Will now grants more Max HP.", ["Iron Will", "Max HP"]) == 1.0


def test_answer_keyword_hit_rate_partial():
    assert answer_keyword_hit_rate("Only mentions Iron Will here.", ["Iron Will", "Death's Grasp"]) == 0.5


def test_answer_keyword_hit_rate_no_expected_keywords_returns_none():
    assert answer_keyword_hit_rate("Anything at all.", []) is None


def test_answer_keyword_hit_rate_whitespace_insensitive():
    # "85 %" vs "85%" is a formatting choice, not a wrong answer.
    assert answer_keyword_hit_rate("Drop rate increased to 85 %.", ["85%"]) == 1.0


def test_answer_keyword_hit_rate_apostrophe_insensitive():
    # The model consistently writes a typographic right single quote
    # (U+2019); the hand-written test set uses a plain apostrophe.
    assert answer_keyword_hit_rate("Fixed Death’s Grasp triggering while bound.", ["Death's Grasp"]) == 1.0


def test_citation_versions_valid_true_when_all_cited_versions_retrieved():
    assert citation_versions_valid("Changed in v.271 and v.269.", _result(269, 271)) is True


def test_citation_versions_valid_ignores_lv_level_phrasing():
    # "Lv. 210" (a character level, from real patch-note phrasing) must not
    # be misread as a citation to v.210 -- confirmed as a real false
    # positive from a live eval run before CITATION_RE gained its leading
    # \b. Only v.271 (a real citation) is retrieved here, so this must
    # still come back True, not False on the phantom "v.210".
    answer = "v.271 - Decreased EXP required to level up from Lv. 210-259."
    assert citation_versions_valid(answer, _result(271)) is True


def test_citation_versions_valid_false_on_fabricated_citation():
    # v.999 was never in the retrieved context -- this is exactly the
    # fabricated-citation case the check exists to catch.
    assert citation_versions_valid("Changed in v.271 and v.999.", _result(271)) is False


def test_citation_versions_valid_none_when_answer_cites_nothing():
    assert citation_versions_valid("No patch version mentioned here.", _result(271)) is None
