"""Cross-encoder reranking: re-scores (query, chunk) pairs more precisely than
the bi-encoder cosine similarity used for the initial retrieval pass. Kept
optional (see retrieve.py's use_reranker flag) since Phase 7's eval is what
decides whether it's actually worth the extra latency for this dataset.
"""

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return candidates
    model = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
