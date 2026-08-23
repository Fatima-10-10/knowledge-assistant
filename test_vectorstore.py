from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.vectorstore.store import ChromaStore

# Reuse the same chunks from earlier units
pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=800))

# NAMESPACE DESIGN: separate collections keep document sets isolated.
# Here we use one collection for this project, but in a bigger system
# you might have "netflix_docs", "stripe_docs" as separate namespaces.
store = ChromaStore(collection_name="netflix_terms")

print("=== INSERT (Create) ===")
store.add(chunks)
print(f"Inserted {store.count()} chunks")

print("\n=== QUERY (Read) — no filter ===")
results = store.query("Can minors use Netflix on their own?", n_results=3)
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"Distance: {dist:.3f} | Category: {meta['category']}")
    print(f"Content: {doc[:150]}\n")

print("=== QUERY with METADATA FILTERING ===")
# Only search chunks tagged "Account Sharing" — proves filtering works
filtered_results = store.query(
    "What are the rules here?",
    n_results=2,
    where={"category": "Account Sharing"}
)
for doc, meta in zip(filtered_results["documents"][0], filtered_results["metadatas"][0]):
    print(f"Category: {meta['category']} (filtered search)")
    print(f"Content: {doc[:150]}\n")

print("=== UPDATE ===")
first_id = store.collection.get(limit=1)["ids"][0]
store.update(
    doc_id=first_id,
    new_content="This is a manually updated test chunk for Unit 5.",
    metadata={"source_type": "test", "source_name": "manual_edit", "page": 0, "category": "test"}
)
updated = store.collection.get(ids=[first_id])
print(f"Updated chunk content: {updated['documents'][0]}")

print("\n=== DELETE ===")
store.delete(first_id)
print(f"Chunks remaining after delete: {store.count()}")
