from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.retrieval.ensemble import Retriever
from app.agents.self_correction import (
    crag_retrieve, generate_with_self_critique, classify_query_complexity
)

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))
retriever = Retriever(chunks)

print("=== Adaptive RAG: complexity classification ===")
for q in ["Hi there!", "What's the age requirement to use Netflix?", "hello"]:
    complexity = classify_query_complexity(q)
    print(f"'{q}' -> {complexity}")

print("\n\n=== CRAG: the query that FAILED in Unit 9 ===")
result = crag_retrieve("getting kicked off my plan", retriever, chunks, max_retries=2)
print(f"Source used: {result['source']}")
print(f"Number of retry attempts: {len(result['attempts'])}")
for i, a in enumerate(result['attempts']):
    print(f"  Attempt {i+1}: query='{a['query']}' -> {a['grade']['raw']}")

if result['source'] == 'documents':
    print("\nTop retrieved chunk:")
    score, idx = result['results'][0]
    print(f"  Category: {chunks[idx].category}")
    print(f"  {chunks[idx].content[:150]}")
else:
    print(f"\nContextualized web query: '{result['contextualized_query']}'")
    print("Fell back to web search results:")
    for r in result['results']:
        print(f"  {r[:150]}")

print("\n\n=== Self-RAG: critique in action ===")
context = [chunks[idx].content for score, idx in retriever.hybrid_search("age requirement", k=3)]
result = generate_with_self_critique("What is the age requirement?", context)
print(f"Was the answer revised? {result['revised']}")
print(f"Final answer: {result['answer']}")

from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.retrieval.ensemble import Retriever

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))
retriever = Retriever(chunks)

results = retriever.bm25_search("Cancellation", k=5)
for score, idx in results:
    print(f"Score: {score:.2f} | Category: {chunks[idx].category}")
    print(f"  {chunks[idx].content[:150]}\n")