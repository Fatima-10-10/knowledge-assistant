from pydantic import BaseModel


class Citation(BaseModel):
    source_type: str      # "pdf", "web", "csv"
    source_name: str      # filename or URL
    page: int | None = None
    category: str | None = None
    excerpt: str           # the actual text this claim came from


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]