from app.retrieval.ensemble import Retriever
from app.retrieval.reranker import rerank
from app.core.llm import get_llm
from app.models.schemas import Citation, QueryResponse


class QueryPipeline:
    """
    The full retrieval-to-answer flow:
    hybrid search -> rerank -> generate answer WITH citations.
    This is what the API endpoint will call.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.retriever = Retriever(chunks)

    def answer(self, question: str, k: int = 3) -> QueryResponse:
        # Stage 1: fast first-pass retrieval, grab extra candidates
        first_stage = self.retriever.hybrid_search(question, k=10)

        # Stage 2: re-rank those candidates for precision
        candidate_texts = [self.chunks[idx].content for score, idx in first_stage]
        candidate_indices = [idx for score, idx in first_stage]
        reranked, _ = rerank(question, candidate_texts, top_k=k)

        top_chunks = [self.chunks[candidate_indices[local_idx]] for score, local_idx in reranked]

        # Build the prompt with clearly numbered sources, so the LLM
        # can reference "Source 1", "Source 2" etc. and we can map
        # those back to real citations afterward.
        context_blocks = []
        for i, chunk in enumerate(top_chunks):
            context_blocks.append(f"[Source {i+1}] {chunk.content}")
        context_text = "\n\n".join(context_blocks)

        llm = get_llm()
        prompt = f"""Answer the question using ONLY the sources below.
Reference sources by their number, e.g. "According to Source 1...".
If the sources don't contain the answer, say so clearly.

{context_text}

QUESTION: {question}
"""
        response = llm.invoke(prompt).content

        citations = [
            Citation(
                source_type=chunk.source_type,
                source_name=chunk.source_name,
                page=chunk.page,
                category=chunk.category,
                excerpt=chunk.content[:200]
            )
            for chunk in top_chunks
        ]

        return QueryResponse(answer=response, citations=citations)