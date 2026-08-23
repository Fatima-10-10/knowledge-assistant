import os
from pinecone import Pinecone, ServerlessSpec
from app.embeddings.embedder import embed_texts, embed_text
from app.ingestion.loaders import SourceDocument


class PineconeStore:
    """
    Cloud-hosted vector database. Unlike Chroma (local file on disk),
    this stores data on Pinecone's servers — useful for comparing a
    managed/cloud provider against a self-hosted one, per the
    curriculum's provider comparison.
    """

    def __init__(self, index_name: str = "knowledge-assistant"):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # MiniLM outputs 384-dimensional vectors (from Unit 4) —
        # Pinecone needs to know this upfront to create the index
        if index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(index_name)

    def add(self, docs: list[SourceDocument]):
        texts = [d.content for d in docs]
        embeddings = embed_texts(texts).tolist()

        vectors = []
        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            vectors.append({
                "id": f"{doc.source_name}_p{doc.page}_{i}",
                "values": emb,
                "metadata": {
                    "text": doc.content,
                    "source_type": doc.source_type,
                    "source_name": doc.source_name,
                    "page": doc.page if doc.page is not None else -1,
                    "category": doc.category if doc.category else "uncategorized",
                }
            })

        # Pinecone recommends batching inserts (upserts) in groups
        self.index.upsert(vectors=vectors)

    def query(self, question: str, n_results: int = 3, filter: dict | None = None):
        question_embedding = embed_text(question).tolist()
        results = self.index.query(
            vector=question_embedding,
            top_k=n_results,
            include_metadata=True,
            filter=filter,
        )
        return results

    def delete(self, doc_id: str):
        self.index.delete(ids=[doc_id])

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        return stats["total_vector_count"]