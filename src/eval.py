"""Phase 7: evaluation harness.

Hand-built test set of questions with known-correct facts, pulled by reading
the actual scraped patches (not invented). Compares four retrieval configs --
baseline, +reranker, +query rewriting, +both -- on retrieval recall (did the
patch(es) that actually discuss this land in the results?), then runs full
generation on whichever config wins to sanity-check answer quality too.

Also includes negative-control questions (a real entity confirmed, against
data/processed/chunks.json, to be absent from the specific version/range
asked about) to catch hallucination -- recall alone only measures whether
answerable questions get answered, never whether an unanswerable one gets
correctly declined instead of a fabricated answer -- and a mechanical check
that every patch version an answer cites was actually in its retrieved
context, since "every answer cites its patch" is a claim worth verifying
rather than trusting.
"""

import os
import re
import sys
from itertools import product
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from generate import NO_MATCH_MESSAGE, get_answer, rewrite_query
from retrieve import get_collection, load_version_dates, retrieve

TEST_SET = [
    {
        "question": "What changed for Kaiser since March 2026?",
        "expected_versions": [267, 269, 271],
        "expected_keywords": ["Iron Will"],
    },
    {
        "question": "What changed for Crimson Queen in v.271?",
        "expected_versions": [271],
        "expected_keywords": ["three"],
    },
    {
        "question": "What boss changes happened for Kaling in v.271?",
        "expected_versions": [271],
        "expected_keywords": ["Death's Grasp"],
    },
    {
        "question": "What changed for the Ring of Restraint in v.271?",
        "expected_versions": [271],
        "expected_keywords": ["85%"],
    },
    {
        "question": "How did EXP requirements for leveling up change in v.271?",
        "expected_versions": [271],
        "expected_keywords": ["210"],
    },
    {
        "question": "What changed for Hoyoung in v.267?",
        "expected_versions": [267],
        "expected_keywords": ["Sage"],
    },
    {
        "question": "What changed for Sia Astelle since v.268?",
        "expected_versions": [268, 269, 270, 271],
        "expected_keywords": [],
    },
    {
        "question": "has kaiser gotten any buffs recently",
        "expected_versions": [271],
        "expected_keywords": [],
    },
    {
        "question": "how did leveling get easier this year",
        "expected_versions": [271],
        "expected_keywords": ["EXP"],
    },
    {
        "question": "What new events were added in v.271?",
        "expected_versions": [271],
        "expected_keywords": ["Frieren"],
    },
    # Negative controls: real entities confirmed (by scanning every chunk's
    # entity_mentioned tag) to have zero chunks in the specific version/range
    # asked about, even though they have real changes in other patches --
    # e.g. Battle Mage appears in v.257-270 but not v.271. Retrieval should
    # come back empty rather than admit an unrelated chunk just because
    # nothing else scores better, and get_answer() should then decline
    # (NO_MATCH_MESSAGE) rather than ever reaching the LLM to answer from
    # irrelevant context.
    {
        "question": "What changed for Battle Mage in v.271?",
        "is_negative": True,
    },
    {
        "question": "What changed for Wild Hunter since v.270?",
        "is_negative": True,
    },
]

# Matches how the system prompt asks the model to cite patches, e.g.
# "v.269 - Ride the Lightning, 2026-06-16" -- loose on punctuation/spacing
# since that's not the thing being checked here. The leading \b is load-
# bearing: without it this also matched the "v." inside "Lv. 210" (patch
# notes phrase levels as "Lv. NNN") as a fake citation to v.210 --
# confirmed as a real false positive from a live eval run.
CITATION_RE = re.compile(r"\bv\.?\s?(\d{3})\b", re.IGNORECASE)


def retrieval_recall(result: dict, expected_versions: list[int]) -> float | None:
    if not expected_versions:
        return None
    got = {p["version"] for p in result["patches"]}
    return len(got & set(expected_versions)) / len(expected_versions)


def retrieval_is_clean(result: dict) -> bool:
    # For a negative control: did retrieval correctly surface nothing to
    # hallucinate from? get_answer() short-circuits to NO_MATCH_MESSAGE
    # -- with no LLM call at all -- exactly when this is true, so this one
    # check *is* the whole hallucination guard for these questions; there's
    # no separate LLM judgment call downstream of it to test as well.
    return not result["patches"]


def _normalize(text: str) -> str:
    # Whitespace-insensitive: the model sometimes writes "85 %" (space before
    # the sign) where the source text has "85%", which is still a correct
    # answer -- just don't let that formatting choice count as a miss.
    # Also fold typographic quotes to straight ones: the model consistently
    # writes "Death's Grasp" with a right single quotation mark (U+2019)
    # while the hand-written test set uses a plain apostrophe -- confirmed
    # this alone was reporting a correct answer as a keyword miss.
    text = text.replace("’", "'").replace("‘", "'")
    return "".join(text.lower().split())


def answer_keyword_hit_rate(answer: str, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    normalized_answer = _normalize(answer)
    hits = sum(1 for kw in expected_keywords if _normalize(kw) in normalized_answer)
    return hits / len(expected_keywords)


def citation_versions_valid(answer: str, result: dict) -> bool | None:
    # Mechanically checks the spec's "every answer cites its patch" claim:
    # every v.NNN the answer cites must be a version retrieve() actually
    # returned -- not one the model invented or pulled in from outside the
    # excerpts it was given. None means the answer cited no version at all
    # (nothing to check either way -- worth a human glance if it happens on
    # a question that should have a real, citable answer).
    cited = {int(m) for m in CITATION_RE.findall(answer)}
    if not cited:
        return None
    allowed = {p["version"] for p in result["patches"]}
    return cited <= allowed


def run_retrieval_config(collection, version_dates, client, use_reranker: bool, use_rewrite: bool) -> dict:
    """Runs every TEST_SET question's retrieval once for this config.
    Returns per-question rows plus the two summary numbers derived from
    them, so callers can print either level of detail."""
    rows = []  # (question, recall_or_None, is_clean_or_None)
    for case in TEST_SET:
        semantic_query = rewrite_query(client, case["question"]) if use_rewrite else None
        result = retrieve(
            case["question"], collection, version_dates,
            semantic_query=semantic_query, use_reranker=use_reranker,
        )
        if case.get("is_negative"):
            rows.append((case["question"], None, retrieval_is_clean(result)))
        else:
            rows.append((case["question"], retrieval_recall(result, case["expected_versions"]), None))

    recalls = [r for _, r, _ in rows if r is not None]
    negatives = [clean for _, _, clean in rows if clean is not None]
    return {
        "rows": rows,
        "mean_recall": sum(recalls) / len(recalls) if recalls else None,
        "negative_clean_rate": sum(negatives) / len(negatives) if negatives else None,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found. Add it to .env as GROQ_API_KEY=your_key_here.")
        return

    client = Groq(api_key=api_key)
    collection = get_collection()
    version_dates = load_version_dates()

    n_negative = sum(1 for c in TEST_SET if c.get("is_negative"))
    print(
        f"Evaluating retrieval across {len(TEST_SET) - n_negative} answerable + "
        f"{n_negative} negative-control questions, 4 configs...\n"
    )
    configs = list(product([False, True], repeat=2))  # (use_reranker, use_rewrite)
    results = {}
    for use_reranker, use_rewrite in configs:
        label = f"reranker={use_reranker:<5} rewrite={use_rewrite:<5}"
        try:
            outcome = run_retrieval_config(collection, version_dates, client, use_reranker, use_rewrite)
        except RateLimitError as e:
            # use_rewrite=True configs call rewrite_query (an LLM call) once
            # per question -- if Groq's daily token budget runs out here,
            # skip just this config rather than crashing the whole eval run.
            print(f"  {label} -> SKIPPED (Groq daily token limit reached: {e.message.splitlines()[0] if e.message else e})")
            continue
        results[(use_reranker, use_rewrite)] = outcome
        neg_str = (
            f"{outcome['negative_clean_rate']:.2f}" if outcome["negative_clean_rate"] is not None else "n/a"
        )
        print(
            f"  {label} -> mean retrieval recall: {outcome['mean_recall']:.2f}"
            f"  |  negative-control clean rate: {neg_str}"
        )

    if not results:
        print("\nNo retrieval configs could be evaluated -- Groq's daily token limit was already exhausted before this run started. Try again once it resets.")
        return

    # Recall decides the winner first (that's the metric with headroom to
    # lose), negative-control clean rate breaks ties -- a config tied on
    # recall that starts admitting hallucination-risk matches on questions
    # that should retrieve nothing is not actually an equal result.
    best_config = max(results, key=lambda k: (results[k]["mean_recall"], results[k]["negative_clean_rate"]))
    best_reranker, best_rewrite = best_config
    best = results[best_config]
    print(
        f"\nBest config: reranker={best_reranker}, rewrite={best_rewrite} "
        f"(recall {best['mean_recall']:.2f}, negative-control clean rate {best['negative_clean_rate']:.2f})"
    )

    print("\nPer-question retrieval detail for the winning config:")
    for question, recall, is_clean in best["rows"]:
        if is_clean is not None:
            mark = "OK  " if is_clean else "FAIL"
            print(f"  [{mark}] (negative control) {question!r}")
        elif recall is not None:
            mark = "OK  " if recall == 1.0 else f"{recall:.2f}"
            print(f"  [{mark}] {question!r}")
        else:
            print(f"  [ -- ] {question!r} (no retrieval ground truth for this one)")

    print(f"\nRunning full generation with the best config for an answer-quality sanity check...\n")
    keyword_scores = []
    citation_checks = []
    for case in TEST_SET:
        try:
            semantic_query = rewrite_query(client, case["question"]) if best_rewrite else None
            result = retrieve(
                case["question"], collection, version_dates,
                semantic_query=semantic_query, use_reranker=best_reranker,
            )
            answer = get_answer(client, case["question"], result)
        except RateLimitError as e:
            # Groq's free tier caps total tokens per *day*, not just per
            # request/minute -- expected to trip during a debugging session
            # that reruns eval.py (and other Groq calls) many times in a
            # short window. Not a bug: stop the generation pass cleanly and
            # report the partial results gathered so far instead of a raw
            # traceback, since this is a normal, recoverable free-tier state.
            print(f"  [SKIPPED] Groq daily token limit reached -- stopping generation pass early.")
            print(f"            ({e.message.splitlines()[0] if e.message else e})\n")
            break

        if case.get("is_negative"):
            declined = answer.strip() == NO_MATCH_MESSAGE
            mark = "OK" if declined else "MISS"
            detail = "declined correctly" if declined else f"DID NOT DECLINE: {answer[:80]!r}"
            print(f"  [{mark}] (negative control) {case['question']!r} -> {detail}")
            continue

        score = answer_keyword_hit_rate(answer, case["expected_keywords"])
        if score is not None:
            keyword_scores.append(score)
            mark = "OK" if score == 1.0 else "MISS"
            print(f"  [{mark}] {case['question']!r} -> keyword hit rate {score:.2f}")

        valid = citation_versions_valid(answer, result)
        if valid is not None:
            citation_checks.append(valid)
            if not valid:
                print(f"       ^ citation check FAILED: answer cites a version not in its retrieved context")

    if keyword_scores:
        print(f"\nMean answer keyword hit rate: {sum(keyword_scores) / len(keyword_scores):.2f}")
    if citation_checks:
        n_valid = sum(citation_checks)
        print(f"Citation validity: {n_valid}/{len(citation_checks)} answers cited only versions in their retrieved context")


if __name__ == "__main__":
    main()
