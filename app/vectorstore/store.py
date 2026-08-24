import chromadb
from app.embeddings.embedder import embed_texts, embed_text
from app.ingestion.loaders import SourceDocument


class ChromaStore:
    """
    Wraps ChromaDB — a free, local vector database that uses HNSW
    indexing under the hood. Supports full CRUD and metadata filtering.
    """

    def __init__(self, collection_name: str = "knowledge_base"):
        # persist_directory saves data to disk so it survives between runs
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        # get_or_create_collection = "namespace" design: different
        # collection names let you keep separate document sets isolated
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, docs: list[SourceDocument]):
        """Insert chunks into the vector store."""
        texts = [d.content for d in docs]
        embeddings = embed_texts(texts).tolist()
        ids = [f"{d.source_name}_p{d.page}_{i}" for i, d in enumerate(docs)]

        # Metadata lets us filter later (e.g. only search PDF chunks,
        # or only chunks tagged with a specific category)
        metadatas = [
            {
                "source_type": d.source_type,
                "source_name": d.source_name,
                "page": d.page if d.page is not None else -1,
                "category": d.category if d.category else "uncategorized",
                "doc_id": d.doc_id if d.doc_id else "unknown",
            }
            for d in docs
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, question: str, n_results: int = 3, where: dict | None = None):
        """
        Search for the most similar chunks. `where` enables metadata
        filtering, e.g. {"category": "Age Limitation"}.
        """
        question_embedding = embed_text(question).tolist()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results,
            where=where,
        )
        return results

    def update(self, doc_id: str, new_content: str, metadata: dict):
        """Update an existing chunk (e.g. if the source document changed)."""
        new_embedding = embed_text(new_content).tolist()
        self.collection.update(
            ids=[doc_id],
            embeddings=[new_embedding],
            documents=[new_content],
            metadatas=[metadata],
        )

    def delete(self, doc_id: str):
        """Remove a chunk entirely."""
        self.collection.delete(ids=[doc_id])

    def count(self) -> int:
        """How many chunks are currently stored."""
        return self.collection.count()

    def delete_by_doc_id(self, doc_id: str):
        """Remove all chunks belonging to one document."""
        self.collection.delete(where={"doc_id": doc_id})

    def query_dense(self, question: str, n_results: int = 10, doc_id: str | None = None):
        """Query, optionally scoped to a single document."""
        where = {"doc_id": doc_id} if doc_id else None
        return self.query(question, n_results=n_results, where=where)
