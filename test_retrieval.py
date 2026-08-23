from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.retrieval.ensemble import Retriever

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))
retriever = Retriever(chunks)

def show(label, results):
    print(f"\n=== {label} ===")
    for score, idx in results:
        print(f"Score: {score:.3f} | Category: {chunks[idx].category}")
        print(f"  {chunks[idx].content[:100]}")

query = "1.2 age requirement"  # deliberately includes an exact clause number

show("Similarity search (dense only)", retriever.similarity_search(query, k=3))
show("Similarity search with score_threshold=0.5", retriever.similarity_search(query, k=3, score_threshold=0.5))
show("BM25 search (sparse/keyword only)", retriever.bm25_search(query, k=3))
show("Hybrid search (dense + sparse blended)", retriever.hybrid_search(query, k=3))
show("MMR search (diverse results)", retriever.mmr_search(query, k=3))