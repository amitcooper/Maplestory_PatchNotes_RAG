"""Phase 8: minimal interactive interface.

Loads the embedding model, vector store, and Groq client once, then answers
questions in a loop -- the spec asks for a CLI loop or basic Streamlit app,
nothing more, so this is intentionally just that.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from generate import build_context, generate_answer, print_sources
from retrieve import get_collection, load_version_dates, retrieve


def answer_question(client: Groq, collection, version_dates: dict[int, str], question: str) -> None:
    result = retrieve(question, collection, version_dates)
    if not result["patches"]:
        print("No matching patch notes found for this question.\n")
        return
    context = build_context(result)
    answer = generate_answer(client, question, context)
    print(f"\n{answer}\n")
    print_sources(result)
    print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found. Add it to .env as GROQ_API_KEY=your_key_here.")
        return

    print("Loading embedding model and vector store...")
    client = Groq(api_key=api_key)
    collection = get_collection()
    version_dates = load_version_dates()

    print('Ready. Ask about MapleStory patch notes, e.g. "what changed for Kaiser since March 2026".')
    print("Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Bye.")
            break

        answer_question(client, collection, version_dates, question)


if __name__ == "__main__":
    main()
