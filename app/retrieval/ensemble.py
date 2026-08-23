import numpy as np
from rank_bm25 import BM25Okapi
from app.embeddings.embedder import embed_texts, embed_text, cosine_similarity
from app.ingestion.loaders import SourceDocument


class Retriever:
    """
    The retriever abstraction: given a set of chunks, answer the
    question "which chunks are relevant to this query?" — swappable
    strategy underneath (similarity, BM25, or a blend of both).
    This is the interface everything downstream (Units 7-13) will
    call, without caring HOW retrieval works internally.
    """

    def __init__(self, docs: list[SourceDocument]):
        self.docs = docs
        self.texts = [d.content for d in docs]

        # Dense: precompute embeddings once for similarity search
        self.embeddings = embed_texts(self.texts)

        # Sparse: BM25 needs tokenized (word-split) text, not raw strings
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def similarity_search(self, query: str, k: int = 5, score_threshold: float = None):
        """
        Pure dense (meaning-based) search. score_threshold optionally
        drops results below a minimum similarity — so a bad match
        returns NOTHING instead of forcing a weak answer.
        """
        query_emb = embed_text(query)
        scores = [cosine_similarity(query_emb, emb) for emb in self.embeddings]
        ranked = sorted(zip(scores, range(len(self.docs))), reverse=True)

        if score_threshold is not None:
            ranked = [(s, i) for s, i in ranked if s >= score_threshold]

        return ranked[:k]

    def bm25_search(self, query: str, k: int = 5):
        """
        Sparse (keyword) search — good for exact terms that embeddings
        can blur together, e.g. specific numbers, names, jargon.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(scores, range(len(self.docs))), reverse=True)
        return ranked[:k]

    def hybrid_search(self, query: str, k: int = 5, dense_weight: float = 0.5):
        """
        Ensemble retriever: blend dense + sparse scores into one
        ranking. dense_weight=0.5 means equal trust in both signals;
        raise it to trust meaning-matching more, lower it to trust
        exact keyword matching more.
        """
        query_emb = embed_text(query)
        dense_scores = np.array([cosine_similarity(query_emb, emb) for emb in self.embeddings])

        tokenized_query = query.lower().split()
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Normalize both to 0-1 range so they're comparable before blending
        def normalize(arr):
            if arr.max() == arr.min():
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        combined = dense_weight * dense_norm + (1 - dense_weight) * sparse_norm
        ranked = sorted(zip(combined, range(len(self.docs))), reverse=True)
        return ranked[:k]

    def mmr_search(self, query: str, k: int = 5, fetch_k: int = 10, lambda_mult: float = 0.5):
        """
        Maximal Marginal Relevance: instead of just top-K by similarity
        (which can return near-duplicate chunks), this balances
        relevance against DIVERSITY — penalizing chunks too similar
        to ones already selected.
        lambda_mult=1.0 -> pure relevance (like normal search)
        lambda_mult=0.0 -> pure diversity (ignores relevance almost entirely)
        """
        query_emb = embed_text(query)
        dense_scores = [cosine_similarity(query_emb, emb) for emb in self.embeddings]
        candidates = sorted(zip(dense_scores, range(len(self.docs))), reverse=True)[:fetch_k]

        selected = []
        selected_indices = []

        while candidates and len(selected) < k:
            best_score = -float("inf")
            best_candidate = None

            for sim_to_query, idx in candidates:
                if not selected_indices:
                    diversity_penalty = 0
                else:
                    # How similar is this candidate to chunks we ALREADY picked?
                    sims_to_selected = [
                        cosine_similarity(self.embeddings[idx], self.embeddings[j])
                        for j in selected_indices
                    ]
                    diversity_penalty = max(sims_to_selected)

                mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = (sim_to_query, idx)

            selected.append(best_candidate)
            selected_indices.append(best_candidate[1])
            candidates.remove(best_candidate)

        return selected