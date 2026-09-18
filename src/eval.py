"""Phase 7: evaluation harness.

Hand-built test set of questions with known-correct facts, pulled by reading
the actual scraped patches (not invented). Compares four retrieval configs --
baseline, +reranker, +query rewriting, +both -- on retrieval recall (did the
patch(es) that actually discuss this land in the results?), then runs full
generation on whichever config wins to sanity-check answer quality too.
"""

import os
import sys
from itertools import product
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from generate import build_context, generate_answer, rewrite_query
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
]


def retrieval_recall(result: dict, expected_versions: list[int]) -> float | None:
    if not expected_versions:
        return None
    got = {p["version"] for p in result["patches"]}
    return len(got & set(expected_versions)) / len(expected_versions)


def _normalize(text: str) -> str:
    # Whitespace-insensitive: the model sometimes writes "85 %" (space before
    # the sign) where the source text has "85%", which is still a correct
    # answer -- just don't let that formatting choice count as a miss.
    return "".join(text.lower().split())


def answer_keyword_hit_rate(answer: str, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    normalized_answer = _normalize(answer)
    hits = sum(1 for kw in expected_keywords if _normalize(kw) in normalized_answer)
    return hits / len(expected_keywords)


def run_retrieval_config(collection, version_dates, client, use_reranker: bool, use_rewrite: bool) -> float:
    scores = []
    for case in TEST_SET:
        semantic_query = rewrite_query(client, case["question"]) if use_rewrite else None
        result = retrieve(
            case["question"], collection, version_dates,
            semantic_query=semantic_query, use_reranker=use_reranker,
        )
        recall = retrieval_recall(result, case["expected_versions"])
        if recall is not None:
            scores.append(recall)
    return sum(scores) / len(scores)


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

    print(f"Evaluating retrieval recall across {len(TEST_SET)} questions, 4 configs...\n")
    configs = list(product([False, True], repeat=2))  # (use_reranker, use_rewrite)
    results = {}
    for use_reranker, use_rewrite in configs:
        label = f"reranker={use_reranker:<5} rewrite={use_rewrite:<5}"
        score = run_retrieval_config(collection, version_dates, client, use_reranker, use_rewrite)
        results[(use_reranker, use_rewrite)] = score
        print(f"  {label} -> mean retrieval recall: {score:.2f}")

    best_config = max(results, key=results.get)
    best_reranker, best_rewrite = best_config
    print(f"\nBest config: reranker={best_reranker}, rewrite={best_rewrite} (recall {results[best_config]:.2f})")

    print(f"\nRunning full generation with the best config for an answer-quality sanity check...\n")
    keyword_scores = []
    for case in TEST_SET:
        semantic_query = rewrite_query(client, case["question"]) if best_rewrite else None
        result = retrieve(
            case["question"], collection, version_dates,
            semantic_query=semantic_query, use_reranker=best_reranker,
        )
        if not result["patches"]:
            continue
        context = build_context(result)
        answer = generate_answer(client, case["question"], context)
        score = answer_keyword_hit_rate(answer, case["expected_keywords"])
        if score is not None:
            keyword_scores.append(score)
            mark = "OK" if score == 1.0 else "MISS"
            print(f"  [{mark}] {case['question']!r} -> keyword hit rate {score:.2f}")

    if keyword_scores:
        print(f"\nMean answer keyword hit rate: {sum(keyword_scores) / len(keyword_scores):.2f}")


if __name__ == "__main__":
    main()
