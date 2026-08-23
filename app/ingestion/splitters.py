from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.loaders import SourceDocument

import re


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation as the boundary."""
    # Split AFTER '.', '!', or '?' followed by a space/newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_documents(
    docs: list[SourceDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[SourceDocument]:
    """
    Group whole sentences together up to chunk_size, WITHOUT ever
    cutting a sentence in half. A chunk may end up slightly over
    chunk_size if a single sentence itself is long — that's the
    tradeoff for never producing a broken sentence.
    """
    chunked_docs = []

    for doc in docs:
        sentences = split_into_sentences(doc.content)
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk_size AND we
            # already have content, close off the current chunk first.
            if current_chunk and len(current_chunk) + len(sentence) > chunk_size:
                chunked_docs.append(SourceDocument(
                    content=current_chunk.strip(),
                    source_type=doc.source_type,
                    source_name=doc.source_name,
                    page=doc.page
                ))
                current_chunk = sentence
            else:
                current_chunk = (current_chunk + " " + sentence).strip()

        # Don't forget the last chunk
        if current_chunk:
            chunked_docs.append(SourceDocument(
                content=current_chunk.strip(),
                source_type=doc.source_type,
                source_name=doc.source_name,
                page=doc.page
            ))

    return chunked_docs

from sentence_transformers import SentenceTransformer
import numpy as np

_embedder = None

def get_embedder():
    """Load the embedding model once and reuse it (loading it is slow)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def chunk_semantically(
    docs: list[SourceDocument],
    similarity_threshold: float = 0.5,
    max_chunk_size: int = 800
) -> list[SourceDocument]:
    """
    Group sentences by topic similarity instead of just punctuation.
    When two consecutive sentences are semantically dissimilar (below
    similarity_threshold), that's treated as a topic boundary and we
    start a new chunk there.
    """
    embedder = get_embedder()
    chunked_docs = []

    for doc in docs:
        sentences = split_into_sentences(doc.content)
        if not sentences:
            continue

        embeddings = embedder.encode(sentences)

        current_chunk_sentences = [sentences[0]]
        for i in range(1, len(sentences)):
            # Cosine similarity between this sentence and the previous one
            sim = np.dot(embeddings[i], embeddings[i - 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1])
            )

            current_text = " ".join(current_chunk_sentences)
            would_exceed_size = len(current_text) + len(sentences[i]) > max_chunk_size

            if sim < similarity_threshold or would_exceed_size:
                # Topic shifted (or chunk got too big) — close this chunk
                chunked_docs.append(SourceDocument(
                    content=" ".join(current_chunk_sentences),
                    source_type=doc.source_type,
                    source_name=doc.source_name,
                    page=doc.page
                ))
                current_chunk_sentences = [sentences[i]]
            else:
                current_chunk_sentences.append(sentences[i])

        if current_chunk_sentences:
            chunked_docs.append(SourceDocument(
                content=" ".join(current_chunk_sentences),
                source_type=doc.source_type,
                source_name=doc.source_name,
                page=doc.page
            ))

    return chunked_docs    

def tag_categories(docs: list[SourceDocument]) -> list[SourceDocument]:
    """
    Extract a category label for each chunk based on numbered clause
    titles like '1.2. Age Limitation.' -> category = 'Age Limitation'.
    Chunks with no detectable clause title are left uncategorized.
    """
    clause_title_pattern = re.compile(r'\d+\.\d+\.\s*([A-Z][a-zA-Z\s]{2,40}?)\.')

    for doc in docs:
        match = clause_title_pattern.search(doc.content)
        if match:
            doc.category = match.group(1).strip()
    return docs

def tag_categories(docs: list[SourceDocument]) -> list[SourceDocument]:
    """
    Extract a category label for each chunk based on numbered clause
    titles like '1.2. Age Limitation.' -> category = 'Age Limitation'.
    Chunks with no detectable clause title are left uncategorized.
    """
    clause_title_pattern = re.compile(r'\d+\.\d+\.\s*([A-Z][a-zA-Z\s]{2,40}?)\.')

    for doc in docs:
        match = clause_title_pattern.search(doc.content)
        if match:
            doc.category = match.group(1).strip()
    return docs        