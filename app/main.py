from fastapi import FastAPI
from app.models.schemas import QueryRequest, QueryResponse
from app.core.pipeline import QueryPipeline
from app.core.startup import build_knowledge_base

app = FastAPI(title="Knowledge Assistant API")

# Build the knowledge base ONCE when the server starts, not per-request
print("Building knowledge base...")
chunks = build_knowledge_base()
pipeline = QueryPipeline(chunks)
print(f"Ready. {len(chunks)} chunks loaded.")


@app.get("/")
def root():
    return {"status": "running", "chunks_loaded": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return pipeline.answer(request.question)