from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.embeddings.embedder import embed_text, embed_texts, cosine_similarity

# Load and chunk the same way as Unit 3
pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=800))

# Embed every chunk once
chunk_texts = [c.content for c in chunks]
chunk_embeddings = embed_texts(chunk_texts)

# Now embed a QUESTION and see which chunks are most similar
question = "Can minors use Netflix on their own?"
question_embedding = embed_text(question)

# Compare the question against every chunk
scores = []
for i, chunk_emb in enumerate(chunk_embeddings):
    sim = cosine_similarity(question_embedding, chunk_emb)
    scores.append((sim, i))

# Sort by similarity, highest first
scores.sort(reverse=True)

print(f"Question: {question}\n")
print("Top 3 most similar chunks:\n")
for sim, idx in scores[:3]:
    print(f"Similarity: {sim:.3f} | Category: {chunks[idx].category}")
    print(f"Content: {chunks[idx].content[:200]}")
    print()

print("\n=== Dimensionality & model comparison ===")

from sentence_transformers import SentenceTransformer
import time

models_to_compare = [
    "all-MiniLM-L6-v2",     # small, fast
    "all-mpnet-base-v2",    # larger, generally higher quality
]

for model_name in models_to_compare:
    model = SentenceTransformer(model_name)
    start = time.time()
    vec = model.encode(question)
    elapsed = time.time() - start
    print(f"{model_name}: dimensions={len(vec)}, encode_time={elapsed:.3f}s")

print("\n=== Does bigger = better retrieval? ===")
big_model = SentenceTransformer("all-mpnet-base-v2")
big_chunk_embeddings = big_model.encode(chunk_texts)
big_question_embedding = big_model.encode(question)

big_scores = []
for i, chunk_emb in enumerate(big_chunk_embeddings):
    sim = cosine_similarity(big_question_embedding, chunk_emb)
    big_scores.append((sim, i))
big_scores.sort(reverse=True)

print("Top match with all-mpnet-base-v2:")
sim, idx = big_scores[0]
print(f"Similarity: {sim:.3f} | Category: {chunks[idx].category}")
print(f"Content: {chunks[idx].content[:150]}")