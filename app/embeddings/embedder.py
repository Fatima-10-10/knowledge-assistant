from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model():
    """Load the embedding model once and reuse it across the app."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Turn a single piece of text into a vector (list of numbers)."""
    model = get_model()
    return model.encode(text)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed many texts at once (faster than one-by-one)."""
    model = get_model()
    return model.encode(texts)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Measure how similar two vectors are, from -1 (opposite) to 1 (identical).
    This is the core math behind "does this chunk match this question?"
    """
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

from langchain_huggingface import HuggingFaceEmbeddings

def get_langchain_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    LangChain's standard embeddings interface — swapping models (or even
    switching to a proprietary provider like OpenAI later) is just
    changing this one function, without touching any retrieval code
    that depends on it.
    """
    return HuggingFaceEmbeddings(model_name=f"sentence-transformers/{model_name}")