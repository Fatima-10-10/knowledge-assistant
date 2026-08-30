from langchain_groq import ChatGroq
from app.core.config import GROQ_MODEL
import time

def invoke_with_retry(llm, prompt, max_retries=3, delay=10):
    """
    Retry an LLM call if we hit a rate limit, waiting between attempts
    instead of crashing immediately.
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                print(f"Rate limited, waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                raise

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm