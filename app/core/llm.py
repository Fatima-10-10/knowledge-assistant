from langchain_groq import ChatGroq
from app.core.config import GROQ_MODEL

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm