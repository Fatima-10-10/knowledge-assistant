from app.core.llm import get_llm
from app.retrieval.ensemble import Retriever
from app.embeddings.embedder import embed_text


def generate_query_variants(query: str, n: int = 3, llm=None) -> list[str]:
    """Generate N different phrasings of the same question (reused idea from Unit 7's multi-query, now used for fusion)."""
    llm = llm or get_llm()
    prompt = f"""Generate {n} different search queries that would help answer
this question, using varied wording and angles. Respond with ONLY the
queries, one per line, no numbering.

QUESTION: {query}
"""
    response = llm.invoke(prompt).content.strip()
    variants = [line.strip() for line in response.split("\n") if line.strip()]
    return [query] + variants[:n]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists (one per query variant) into a single
    ranking. Each document gets a score based on ITS RANK POSITION in
    each list, not its raw similarity score — this makes it robust
    to different queries having very different score scales.

    Formula: score(doc) = sum over all rankings of 1 / (k + rank_of_doc)
    A document that ranks well across MULTIPLE query variants rises
    to the top, even if it was never #1 in any single one.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            if doc_idx not in scores:
                scores[doc_idx] = 0.0
            scores[doc_idx] += 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def rag_fusion_search(query: str, retriever: Retriever, k: int = 5, n_variants: int = 3):
    """
    Full RAG-Fusion pipeline:
    1. Generate multiple query variants
    2. Retrieve separately for each
    3. Combine rankings with Reciprocal Rank Fusion
    """
    variants = generate_query_variants(query, n=n_variants)

    all_rankings = []
    for variant in variants:
        results = retriever.hybrid_search(variant, k=10)
        ranking = [idx for score, idx in results]  # just the doc indices, in rank order
        all_rankings.append(ranking)

    fused = reciprocal_rank_fusion(all_rankings)
    return fused[:k], variants

def hyde_search(query: str, retriever: Retriever, k: int = 5, llm=None):
    """
    HyDE (Hypothetical Document Embeddings): instead of embedding the
    QUESTION directly, generate a hypothetical ANSWER first, then
    search using that answer's embedding. Works well when the question's
    phrasing is very different from how the answer would actually be
    worded in the source document.
    """
    llm = llm or get_llm()
    prompt = f"""Write a short, plausible-sounding answer to this question,
as if it were an excerpt from an official document. It doesn't need
to be factually correct -- it just needs to be written in the STYLE
the real answer would appear in.

QUESTION: {query}
"""
    hypothetical_answer = llm.invoke(prompt).content.strip()

    # Now search using the hypothetical answer's embedding, not the question's
    hyde_embedding = embed_text(hypothetical_answer)

    from app.embeddings.embedder import cosine_similarity
    scores = [cosine_similarity(hyde_embedding, emb) for emb in retriever.embeddings]
    ranked = sorted(zip(scores, range(len(retriever.docs))), reverse=True)

    return ranked[:k], hypothetical_answer