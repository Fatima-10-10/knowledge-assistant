from app.ingestion.loaders import load_pdf, load_webpage, load_csv
from app.ingestion.splitters import chunk_documents, tag_categories


def build_knowledge_base():
    """
    Load and chunk all our sources into one combined list.
    In a bigger system this would loop over many files; for now
    it's our known Netflix sources.
    """
    all_chunks = []

    pdf_docs = load_pdf("data/netflix.pdf")
    all_chunks.extend(tag_categories(chunk_documents(pdf_docs, chunk_size=500)))

    try:
        web_docs = load_webpage("https://www.netflix.com/signup/planform")
        all_chunks.extend(chunk_documents(web_docs, chunk_size=500))
    except Exception as e:
        print(f"Warning: could not load webpage ({e}), skipping")

    csv_docs = load_csv("data/pricing.csv")
    all_chunks.extend(csv_docs)  # CSV rows are already small, no need to re-chunk

    return all_chunks