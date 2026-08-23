from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.retrieval.ensemble import Retriever
from app.retrieval.advanced import (
    contextual_compression, ParentDocumentRetriever, self_query,
    multi_query, rewrite_query, step_back_prompt, decompose_query
)

pdf_docs = load_pdf("data/netflix.pdf")  # these are the "parent" pages
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))  # small chunks
retriever = Retriever(chunks)

print("=== 1. Contextual Compression ===")
results = retriever.hybrid_search("Can minors use Netflix?", k=2)
chunks_with_scores = [(score, chunks[idx]) for score, idx in results]
compressed = contextual_compression("Can minors use Netflix?", chunks_with_scores)
for c in compressed:
    print(f"- {c}")

print("\n=== 2. Parent Document Retriever (small-to-big) ===")
pdr = ParentDocumentRetriever(chunks, pdf_docs)
parents = pdr.retrieve("Can minors use Netflix?", k=1)
print(f"Returned full page ({len(parents[0])} chars) instead of a small chunk:")
print(parents[0][:300], "...")

print("\n=== 3. Self-Query Retriever ===")
result = self_query("What does the Age Limitation clause say?")
print(result)

print("\n=== 4. Multi-Query Retriever ===")
variants = multi_query("Can minors use Netflix on their own?")
for v in variants:
    print(f"- {v}")

print("\n=== 5. Query Rewriting ===")
rewritten = rewrite_query("can kids use this")
print(f"Original: 'can kids use this' -> Rewritten: '{rewritten}'")

print("\n=== 6. Step-Back Prompting ===")
stepback = step_back_prompt("Can a 15 year old with their own payment card sign up alone?")
print(f"Step-back question: {stepback}")

print("\n=== 7. Query Decomposition ===")
sub_qs = decompose_query("What's the age requirement and can I share my account with family?")
for q in sub_qs:
    print(f"- {q}")