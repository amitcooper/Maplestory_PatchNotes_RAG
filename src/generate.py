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

SYSTEM_PROMPT = """You are answering a MapleStory player's question using only the official patch \
notes excerpts provided below. Rules:
- Be short: state the change itself in one line per relevant patch, with no mechanics, numbers, \
lore, or context beyond what directly answers the question, and no closing summary or offer to help further.
- Cover every patch in the excerpts that's actually relevant to the question -- brevity means \
cutting detail per patch, not dropping a whole patch that has a real answer just because a newer \
one also does. If a given patch's excerpts have nothing relevant, leave that patch out entirely.
- Only use information in the provided excerpts. If they don't answer the question, say so in one sentence.
- If the question asks for a list of items (e.g. "which bosses were added"), answer as one line per \
item: name, then patch/date, then at most a short clause of relevant detail -- not a paragraph per item.
- If the question asks what rewards/items a boss (or several bosses) grants, structure the answer \
per boss: the boss's name and patch/date on its own line, then every reward item on its own separate \
line below it -- never merge multiple rewards into one sentence. If a reward carries stat bonuses \
(STR/DEX/INT/LUK, Max HP/MP, Attack, etc.), list those stats too, not just the item's name.
- Every claim must cite which patch it's from (e.g. "v.269 - Ride the Lightning, 2026-06-16").
- The excerpts are given newest patch first -- lead with the most recent."""

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

def needs_followup_resolution(question: str) -> bool:
    # Retrieval doesn't need to pay for an LLM call when a question already
    # names a real class/boss/item -- it's self-contained regardless of
    # phrasing. But trying to shortcut the other direction with a hardcoded
    # list of "vague" words (he/it/that/the boss/...) doesn't work: a real
    # follow-up like "what items can I get from each?" uses a word ("each")
    # that was never on the list, so it silently skipped resolution entirely
    # and went to retrieval with no entity to anchor on -- confirmed this is
    # exactly why that question returned an unrelated grab-bag of items from
    # across the whole corpus. A question naming no known entity can't
    # retrieve correctly on its own anyway, so there's no real list of "vague
    # enough to resolve" words to maintain -- just always try resolving
    # anything that doesn't already name an entity, and trust the resolver's
    # own instruction to return an already-self-contained question unchanged.
    lower = question.lower()
    return not any(re.search(r"\b" + re.escape(name.lower()) + r"\b", lower) for name in KNOWN_ENTITIES)


def resolve_followup(client: Groq, history: list[dict], question: str) -> str:
    if not history or not needs_followup_resolution(question):
        return question
    # 400 chars silently dropped the tail of any list-style answer -- e.g. a
    # 5-item "which bosses were added" answer runs past 400 chars, so a
    # follow-up naming the last item ("Baldrix") never saw it and dropped it
    # from the resolved question entirely. 2000 comfortably covers a list
    # answer (SYSTEM_PROMPT already keeps these short) while still bounding
    # the cost of this cheap resolution call.
    convo = "\n".join(f"Q: {turn['question']}\nA: {turn['answer'][:2000]}" for turn in history[-2:])
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
