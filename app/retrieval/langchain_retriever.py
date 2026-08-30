from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from app.retrieval.ensemble import Retriever as OurRetriever


class LangChainCompatibleRetriever(BaseRetriever):
    """
    Wraps our custom Retriever (Unit 6) so it can be used anywhere
    LangChain expects a standard retriever -- chains, agents, etc.
    This is the "custom retriever class" the curriculum asks for.
    """
    our_retriever: OurRetriever
    chunks: list
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        results = self.our_retriever.hybrid_search(query, k=self.k)
        docs = []
        for score, idx in results:
            chunk = self.chunks[idx]
            docs.append(Document(
                page_content=chunk.content,
                metadata={
                    "source_name": chunk.source_name,
                    "page": chunk.page,
                    "category": chunk.category,
                    "score": score
                }
            ))
        return docs