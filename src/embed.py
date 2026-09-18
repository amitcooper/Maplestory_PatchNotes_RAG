"""Phase 4: embed all chunks and load them into a local Chroma collection."""

import json
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

# Chroma's on-disk index lives outside the project folder on purpose: this
# project sits inside OneDrive, and OneDrive's real-time sync corrupts
# Chroma's binary HNSW index files while they're being written (confirmed:
# the same collection reads fine from a non-synced path, but reliably fails
# with "Error loading hnsw index" once it's under OneDrive).
CHROMA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MaplePatchNotesRag" / "chroma"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "maple_patch_notes"
BATCH_SIZE = 200


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def to_metadata(chunk: dict) -> dict:
    # Chroma rejects None in metadata values; subsection is None for chunks
    # that sit directly under an <h1> with no <h2> yet.
    return {
        "date": chunk["date"],
        "patch_name": chunk["patch_name"],
        "version": chunk["version"],
        "section": chunk["section"],
        "subsection": chunk["subsection"] or "",
        "entity_mentioned": chunk["entity_mentioned"],
        "source_url": chunk["source_url"],
    }


def build_collection(chunks: list[dict]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[to_metadata(c) for c in batch],
        )
        print(f"  embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    return collection


def sanity_check(collection: chromadb.Collection) -> None:
    query = "Kaiser skill changes"
    result = collection.query(query_texts=[query], n_results=3)

    print(f"\nSanity check -- semantic search for: {query!r}")
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        print(f"  [{meta['patch_name']} | {meta['date']} | {meta['section']}] (distance {dist:.3f})")
        print(f"    {doc[:120]}...")


def main() -> None:
    # Patch notes text contains characters (e.g. "->") that some Windows
    # console codepages can't encode; fall back instead of crashing on print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL}' into {CHROMA_DIR}...")
    collection = build_collection(chunks)
    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
    sanity_check(collection)


if __name__ == "__main__":
    main()
