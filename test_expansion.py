from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.retrieval.ensemble import Retriever
from app.retrieval.expansion import rag_fusion_search, hyde_search

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))
retriever = Retriever(chunks)

def show(label, results, is_fused=False):
    print(f"\n=== {label} ===")
    for item in results:
        if is_fused:
            idx, score = item
        else:
            score, idx = item
        print(f"Score: {score:.4f} | Category: {chunks[idx].category}")
        print(f"  {chunks[idx].content[:100]}")


# TEST 1: A vague question where expansion SHOULD help
vague_query = "getting kicked off my plan"
print(f"\n{'='*60}\nTEST 1 (vague phrasing): '{vague_query}'\n{'='*60}")

baseline = retriever.hybrid_search(vague_query, k=5)
show("Baseline hybrid search (no expansion)", baseline)

fused, variants = rag_fusion_search(vague_query, retriever, k=5)
print(f"\nGenerated query variants: {variants}")
show("RAG-Fusion (RRF)", fused, is_fused=True)

hyde_results, hypothetical = hyde_search(vague_query, retriever, k=5)
print(f"\nHyDE's hypothetical answer: {hypothetical[:200]}")
show("HyDE", hyde_results)


# TEST 2: An already-precise question where expansion might just add noise
precise_query = "1.2 age requirement"
print(f"\n\n{'='*60}\nTEST 2 (already precise): '{precise_query}'\n{'='*60}")

baseline2 = retriever.hybrid_search(precise_query, k=3)
show("Baseline hybrid search (no expansion)", baseline2)

fused2, variants2 = rag_fusion_search(precise_query, retriever, k=3)
print(f"\nGenerated query variants: {variants2}")
show("RAG-Fusion (RRF)", fused2, is_fused=True)

print(f"\nOriginal query: 'getting kicked off my plan'")
if result['source'] == 'web_fallback':
    print(f"Contextualized web query: '{result['contextualized_query']}'")