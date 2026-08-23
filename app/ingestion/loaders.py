from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
import csv


@dataclass
@dataclass
class SourceDocument:
    content: str
    source_type: str
    source_name: str
    page: int | None = None
    category: str | None = None   # NEW: e.g. "Age Limitation", "Offers"
def load_pdf(pdf_path: str) -> list[SourceDocument]:
    """Load a PDF, keeping each page as a separate, traceable document."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    docs = []
    for i, page in enumerate(pages):
        docs.append(SourceDocument(
            content=page.page_content,
            source_type="pdf",
            source_name=pdf_path,
            page=i + 1  # human-friendly page number
        ))
    return docs


def load_webpage(url: str) -> list[SourceDocument]:
    """Load a webpage as a single traceable document."""
    loader = WebBaseLoader(url)
    web_docs = loader.load()

    docs = []
    for d in web_docs:
        docs.append(SourceDocument(
            content=d.page_content,
            source_type="web",
            source_name=url,
            page=None
        ))
    return docs

def load_csv(csv_path: str) -> list[SourceDocument]:
    """Load a CSV, turning each row into its own traceable document."""
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Turn the row into readable text, not just raw dict
            row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
            docs.append(SourceDocument(
                content=row_text,
                source_type="csv",
                source_name=csv_path,
                page=i + 1  # row number
            ))
    return docs