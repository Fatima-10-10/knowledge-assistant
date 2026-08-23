from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents

pdf_docs = load_pdf("data/netflix.pdf")

# Combine all pages into one flat piece of text length, just to see total size
total_chars = sum(len(d.content) for d in pdf_docs)
print(f"Total PDF content: {total_chars} characters across {len(pdf_docs)} pages\n")

# Try a SMALL chunk size first
small_chunks = chunk_documents(pdf_docs, chunk_size=200, chunk_overlap=20)
print(f"With chunk_size=200: {len(small_chunks)} chunks created")
print(f"Example chunk: {repr(small_chunks[5].content)}\n")

# Try a LARGER chunk size
large_chunks = chunk_documents(pdf_docs, chunk_size=800, chunk_overlap=100)
print(f"With chunk_size=800: {len(large_chunks)} chunks created")
print(f"Example chunk: {repr(large_chunks[2].content)}\n")

# Confirm metadata survived splitting
print(f"First small chunk's source: {small_chunks[0].source_name}, page {small_chunks[0].page}")

from app.ingestion.splitters import chunk_semantically, tag_categories

print("\n=== Semantic chunking ===")
semantic_chunks = chunk_semantically(pdf_docs)
print(f"Created {len(semantic_chunks)} semantic chunks")
for c in semantic_chunks[:3]:
    print(f"--- chunk (page {c.page}) ---")
    print(c.content[:250])
    print()

print("\n=== Category tagging ===")
tagged_chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=800))
for c in tagged_chunks[:6]:
    print(f"Category: {c.category} | Page: {c.page} | Preview: {c.content[:80]}")