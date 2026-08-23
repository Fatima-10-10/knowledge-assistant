from dotenv import load_dotenv
load_dotenv()

import time
from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.vectorstore.pinecone_store import PineconeStore

# ... rest of the file stays the same
import time
from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.vectorstore.pinecone_store import PineconeStore

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=800))

store = PineconeStore()

print("=== INSERT ===")
store.add(chunks)
print("Waiting for Pinecone to index (cloud writes aren't instant)...")
time.sleep(10)  # Pinecone indexing is near-real-time but not synchronous
print(f"Total vectors in index: {store.count()}")

print("\n=== QUERY ===")
results = store.query("Can minors use Netflix on their own?", n_results=3)
for match in results["matches"]:
    print(f"Score: {match['score']:.3f} | Category: {match['metadata']['category']}")
    print(f"Content: {match['metadata']['text'][:150]}\n")

print("=== QUERY with FILTER ===")
filtered = store.query(
    "What are the rules here?",
    n_results=2,
    filter={"category": {"$eq": "Account Sharing"}}
)
for match in filtered["matches"]:
    print(f"Category: {match['metadata']['category']} (filtered)")
    print(f"Content: {match['metadata']['text'][:150]}\n")