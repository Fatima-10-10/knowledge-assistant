import time
from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.retrieval.ensemble import Retriever
from app.retrieval.reranker import rerank

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))
retriever = Retriever(chunks)

query = "What happens if I share my Netflix account outside my household?"

print("=== STAGE 1: First-stage retrieval (bi-encoder, fast) ===")
start = time.time()
first_stage = retriever.hybrid_search(query, k=10)  # get more candidates than we need
stage1_time = time.time() - start
print(f"Time: {stage1_time:.4f}s")
for score, idx in first_stage[:5]:
    print(f"  Score: {score:.3f} | Category: {chunks[idx].category} | {chunks[idx].content[:80]}")

print("\n=== STAGE 2: Re-ranking (cross-encoder, slower but sharper) ===")
candidate_texts = [chunks[idx].content for score, idx in first_stage]
candidate_indices = [idx for score, idx in first_stage]

reranked, rerank_time = rerank(query, candidate_texts, top_k=3)
print(f"Time: {rerank_time:.4f}s")
for score, local_idx in reranked:
    original_idx = candidate_indices[local_idx]
    print(f"  Score: {score:.3f} | Category: {chunks[original_idx].category} | {chunks[original_idx].content[:80]}")

print(f"\n=== COMPARISON ===")
print(f"First-stage top-3 (before reranking):")
for score, idx in first_stage[:3]:
    print(f"  Category: {chunks[idx].category}")
print(f"\nAfter reranking, top-3:")
for score, local_idx in reranked:
    original_idx = candidate_indices[local_idx]
    print(f"  Category: {chunks[original_idx].category}")

print(f"\nAdded latency from reranking: {rerank_time:.4f}s ({rerank_time/stage1_time:.1f}x the first-stage time)")