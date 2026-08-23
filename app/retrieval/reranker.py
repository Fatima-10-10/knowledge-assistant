import time
from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        # A small, free, local cross-encoder model
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, candidates: list[str], top_k: int = 3):
    """
    Re-score first-stage candidates using a cross-encoder, which looks
    at the query and each chunk TOGETHER for a more accurate relevance
    score than bi-encoder similarity alone.
    """
    reranker = get_reranker()
    pairs = [[query, c] for c in candidates]

    start = time.time()
    scores = reranker.predict(pairs)
    elapsed = time.time() - start

    ranked = sorted(zip(scores, range(len(candidates))), reverse=True)
    return ranked[:top_k], elapsed