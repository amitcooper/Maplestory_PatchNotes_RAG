"""Phase 6: answer a question by retrieving chunks and asking an LLM to synthesize them.

Backend: Groq's free API (default). Kept behind a single generate_answer(prompt)
call so swapping in a local Ollama model later is a one-function change, not a
redesign -- not built now since nothing in this project needs it yet.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from chunk import KNOWN_ENTITIES
from retrieve import get_collection, load_version_dates, retrieve

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a knowledgeable MapleStory player explaining recent changes to a \
friend, using only the official patch notes excerpts provided below. Rules:
- Only use information in the provided excerpts. If they don't answer the question, say so plainly.
- Write like you're actually talking to someone: normal paragraphs and sentences. Don't default \
to a table or a bare bullet list of stat deltas unless the question is itself a request for a \
list (e.g. "list every boss added"). Explain what a change means, not just the raw before/after numbers.
- Every claim about something that changed must cite which patch it's from (e.g. "v.269 - Ride the Lightning, 2026-06-16").
- The excerpts are given newest patch first -- lead with the most recent change, then work backward if there's history worth mentioning.
- Be concise. Don't pad the answer with information the excerpts don't contain."""

REWRITE_SYSTEM_PROMPT = """Rewrite the user's question into a short, keyword-dense search \
query for a semantic search engine over MapleStory patch notes. Keep any class/boss/item \
names and dates exact. Expand vague phrasing into concrete terms (e.g. "gotten any buffs" \
-> "skill damage buff increase"; "easier to level" -> "EXP required decreased"). \
Return only the rewritten query, nothing else."""


def rewrite_query(client: Groq, question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


FOLLOWUP_SYSTEM_PROMPT = """Given the recent conversation below, rewrite the user's latest \
question into a fully self-contained question about MapleStory patch notes. Resolve any \
pronoun or vague reference (he, she, it, that, those, this, the boss, the class, etc.) to the \
specific named entity from the conversation. If the question already stands on its own, return \
it completely unchanged. Return only the question, nothing else."""

# Retrieval doesn't need to pay for an LLM call to notice a question is
# already self-contained (names a real class/boss, or a version/date) --
# only ambiguous-sounding follow-ups ("what about him", "does that stack")
# are worth resolving against prior turns.
PRONOUN_RE = re.compile(
    r"\b(he|him|his|she|her|it|its|they|them|their|that|this|these|those|"
    r"the boss|the class|the job|the skill|the item)\b",
    re.IGNORECASE,
)


def needs_followup_resolution(question: str) -> bool:
    lower = question.lower()
    if any(re.search(r"\b" + re.escape(name.lower()) + r"\b", lower) for name in KNOWN_ENTITIES):
        return False
    return bool(PRONOUN_RE.search(question))


def resolve_followup(client: Groq, history: list[dict], question: str) -> str:
    if not history or not needs_followup_resolution(question):
        return question
    convo = "\n".join(f"Q: {turn['question']}\nA: {turn['answer'][:400]}" for turn in history[-2:])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": f"Recent conversation:\n{convo}\n\nLatest question: {question}"},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def build_context(result: dict) -> str:
    blocks = []
    for patch in result["patches"]:
        header = f"### {patch['patch_name']} ({patch['date']})"
        body = "\n\n".join(
            f"[{c['meta']['section']}"
            + (f" > {c['meta']['subsection']}" if c["meta"]["subsection"] else "")
            + f"]\n{c['text']}"
            for c in patch["chunks"]
        )
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def generate_answer(client: Groq, question: str, context: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Patch notes excerpts:\n\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def print_sources(result: dict) -> None:
    print("\nSources:")
    for patch in result["patches"]:
        print(f"  - {patch['patch_name']} ({patch['date']}) -- {patch['source_url']}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    if len(sys.argv) < 2:
        print('Usage: python src/generate.py "your question here"')
        return
    question = " ".join(sys.argv[1:])

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found. Add it to .env as GROQ_API_KEY=your_key_here.")
        return

    collection = get_collection()
    version_dates = load_version_dates()
    result = retrieve(question, collection, version_dates)

    if not result["patches"]:
        print(f"Query: {question!r}")
        print("No matching patch notes found for this question.")
        return

    context = build_context(result)
    client = Groq(api_key=api_key)
    answer = generate_answer(client, question, context)

    print(f"Query: {question!r}\n")
    print(answer)
    print_sources(result)


if __name__ == "__main__":
    main()
