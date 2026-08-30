from app.retrieval.reranker import rerank
from app.core.llm import get_llm
from app.models.schemas import Citation, QueryResponse
from app.vectorstore.store import ChromaStore


class QueryPipeline:
    """
    Retrieval-to-answer flow backed by a LIVE Chroma vector store,
    so documents can be added/deleted without restarting the server.
    Optionally scoped to a single document via doc_id.
    """

    def __init__(self):
        self.store = ChromaStore(collection_name="cortex_docs")

    def answer(self, question: str, k: int = 3, doc_id: str | None = None) -> QueryResponse:
        results = self.store.query_dense(question, n_results=10, doc_id=doc_id)

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            return QueryResponse(
                answer="I couldn't find any relevant information in the selected document(s).",
                citations=[]
            )

        reranked, _ = rerank(question, docs, top_k=min(k, len(docs)))

        context_blocks = []
        top_items = []
        for i, (score, local_idx) in enumerate(reranked):
            context_blocks.append(f"[Source {i+1}] {docs[local_idx]}")
            top_items.append(metas[local_idx])
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
                source_type=meta["source_type"],
                source_name=meta["source_name"],
                page=meta["page"] if meta["page"] != -1 else None,
                category=meta.get("category"),
                excerpt=docs[local_idx][:200]
            )
            for (score, local_idx), meta in zip(reranked, top_items)
        ]

        return QueryResponse(answer=response, citations=citations)