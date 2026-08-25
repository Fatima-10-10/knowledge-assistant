import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.models.schemas import QueryRequest, QueryResponse
from app.core.pipeline import QueryPipeline
from app.core.registry import init_db, add_document, list_documents, delete_document
from app.ingestion.loaders import load_any
from app.ingestion.splitters import chunk_documents, tag_categories
from app.vectorstore.store import ChromaStore

app = FastAPI(title="Cortex")
app.mount("/app", StaticFiles(directory="static", html=True), name="static")

os.makedirs("data/uploads", exist_ok=True)
init_db()

store = ChromaStore(collection_name="cortex_docs")
pipeline = QueryPipeline()


class QueryRequestWithScope(BaseModel):
    question: str
    doc_id: str | None = None  # None = search ALL documents


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/documents")
def get_documents():
    return list_documents()


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    save_path = f"data/uploads/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        raw_docs = load_any(save_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fix: use the clean original filename for citations, not the disk path
    for d in raw_docs:
        d.source_name = file.filename

    chunks = tag_categories(chunk_documents(raw_docs, chunk_size=500))

    doc_id = add_document(filename=file.filename, source_type=file.filename.split(".")[-1], chunk_count=len(chunks))

    for c in chunks:
        c.doc_id = doc_id

    store.add(chunks)

    return {"doc_id": doc_id, "filename": file.filename, "chunks_added": len(chunks)}

@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    store.delete_by_doc_id(doc_id)
    delete_document(doc_id)
    return {"deleted": doc_id}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequestWithScope):
    return pipeline.answer(request.question, doc_id=request.doc_id)


from pydantic import BaseModel as PydanticBaseModel

class URLRequest(PydanticBaseModel):
    url: str


@app.post("/documents/add-url")
def add_url_document(request: URLRequest):
    from app.ingestion.loaders import load_webpage

    try:
        raw_docs = load_webpage(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load URL: {e}")

    if not raw_docs or not raw_docs[0].content.strip():
        raise HTTPException(status_code=400, detail="No content found at that URL")

    # source_name is already the URL itself from load_webpage — good, that's
    # exactly what a citation should show for a web source
    chunks = tag_categories(chunk_documents(raw_docs, chunk_size=500))

    doc_id = add_document(filename=request.url, source_type="web", chunk_count=len(chunks))

    for c in chunks:
        c.doc_id = doc_id

    store.add(chunks)

    return {"doc_id": doc_id, "filename": request.url, "chunks_added": len(chunks)}