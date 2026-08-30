from app.core.llm import get_llm
from app.retrieval.ensemble import Retriever


def grade_documents(query: str, chunks_content: list[str], llm=None) -> dict:
    """
    Ask the LLM to judge: are these retrieved chunks ACTUALLY relevant
    to the query, or not? Uses a STRICT grading prompt -- being on a
    related topic is not enough, the documents must specifically
    address what was asked.
    """
    llm = llm or get_llm()
    joined = "\n---\n".join(chunks_content)

    prompt = f"""You are a STRICT grader. The retrieved documents must
DIRECTLY and SPECIFICALLY answer the question -- not just be loosely
related or mention similar topics.

QUESTION: {query}

RETRIEVED DOCUMENTS:
{joined}

Do these documents contain the SPECIFIC information needed to directly
answer the question? Being on a related topic is NOT enough -- they
must actually address what was asked.
Respond with ONLY one word: RELEVANT or NOT_RELEVANT
"""
    response = llm.invoke(prompt).content.strip().upper()
    is_relevant = "NOT_RELEVANT" not in response and "RELEVANT" in response
    return {"relevant": is_relevant, "raw": response}


def contextual_rewrite_query(original_query: str, document_anchor: str, llm=None) -> str:
    """
    Rewrite the query for a retry, GROUNDED in a reliable anchor of the
    actual document (its first chunk) -- not a possibly-irrelevant
    retrieved chunk. Prevents the rewrite from drifting into an
    unrelated domain (e.g. turning a Netflix question into a health
    insurance one).
    """
    llm = llm or get_llm()
    prompt = f"""The previous search for this question did not find a
good match in the document. Rewrite the question to search better.
Use the document excerpt below to understand what this document is
actually about, and try DIFFERENT specific terminology than the
original question -- including likely exact words the document itself
would use (e.g. "cancel", "terminate", "refund").

DOCUMENT EXCERPT (identifies what this document is about): {document_anchor[:400]}

ORIGINAL QUESTION: {original_query}

Respond with ONLY the rewritten question, staying on the same topic
as the document above and using its likely terminology.
"""
    return llm.invoke(prompt).content.strip()


def contextualize_web_query(original_query: str, document_anchor: str, llm=None) -> str:
    """
    Before falling back to the web, rewrite the query to include context
    about what document/domain we were actually searching. Uses a
    reliable document anchor (not a possibly-bad retrieved chunk) so
    the fallback query doesn't hallucinate an unrelated company/domain.
    """
    llm = llm or get_llm()
    prompt = f"""A user asked a question about a specific document, but
that document didn't have a good answer. Rewrite their question into a
web search query, using the document excerpt below to correctly
identify what company/topic/domain this is about. Do NOT invent or
guess a different company -- use only what the excerpt tells you.

DOCUMENT EXCERPT: {document_anchor[:400]}

USER'S ORIGINAL QUESTION: {original_query}

Respond with ONLY the rewritten web search query.
"""
    return llm.invoke(prompt).content.strip()


def web_search_fallback(query: str, max_results: int = 3) -> list[str]:
    """
    Fall back to live web search when our own documents don't have a
    good answer -- the actual CRAG fallback behavior.
    """
    from ddgs import DDGS
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"{r['title']}: {r['body']} (source: {r['href']})")
    except Exception as e:
        results.append(f"Web search unavailable: {e}")
    return results


def crag_retrieve(query: str, retriever: Retriever, chunks, k: int = 5, max_retries: int = 2, llm=None):
    """
    Corrective RAG: retrieve -> grade -> if not relevant, REWRITE the
    query (grounded in a reliable document anchor) and retry -> if
    still bad after retries, fall back to web search, also grounded in
    the same reliable anchor -- never in a possibly-irrelevant
    retrieved chunk, which was causing domain hallucination.
    """
    llm = llm or get_llm()
    current_query = query
    attempts = []

    # Reliable anchor: the actual first chunk of the loaded document,
    # NOT a retrieved result (which might itself be irrelevant/misleading).
    document_anchor = chunks[0].content if chunks else ""

    for attempt in range(max_retries + 1):
        results = retriever.hybrid_search(current_query, k=k)
        chunk_texts = [chunks[idx].content for score, idx in results]

        grade = grade_documents(query, chunk_texts, llm=llm)
        attempts.append({"query": current_query, "grade": grade})

        if grade["relevant"]:
            return {
                "source": "documents",
                "results": results,
                "chunks": chunks,
                "attempts": attempts
            }

        current_query = contextual_rewrite_query(current_query, document_anchor, llm=llm)

    contextualized_query = contextualize_web_query(query, document_anchor, llm=llm)
    web_results = web_search_fallback(contextualized_query)
    return {
        "source": "web_fallback",
        "results": web_results,
        "contextualized_query": contextualized_query,
        "attempts": attempts
    }


def generate_with_self_critique(query: str, context_chunks: list[str], llm=None, max_revisions: int = 1):
    """
    Self-RAG: generate a draft answer, then CRITIQUE it against the
    actual source text before showing it to the user. Catches cases
    where the model drifted from what the sources actually say.
    """
    llm = llm or get_llm()
    context_text = "\n\n".join(f"[Source {i+1}] {c}" for i, c in enumerate(context_chunks))

    draft_prompt = f"""Answer the question using ONLY the sources below.

{context_text}

QUESTION: {query}
"""
    draft = llm.invoke(draft_prompt).content

    for _ in range(max_revisions):
        critique_prompt = f"""Review this draft answer against the sources.
Is every claim in the answer actually supported by the sources? If yes,
respond with exactly: APPROVED
If no, respond with a CORRECTED version of the answer that only states
what the sources actually support.

SOURCES:
{context_text}

DRAFT ANSWER:
{draft}
"""
        critique_result = llm.invoke(critique_prompt).content.strip()

        if critique_result == "APPROVED":
            return {"answer": draft, "revised": False}
        else:
            draft = critique_result

    return {"answer": draft, "revised": True}


def classify_query_complexity(query: str, llm=None) -> str:
    """
    Adaptive RAG: decide if a query is SIMPLE (pure greeting/small talk,
    doesn't need retrieval) or COMPLEX (needs the full retrieval
    pipeline -- any question asking for a specific fact, rule, or
    detail, even if it sounds like general knowledge).
    """
    llm = llm or get_llm()
    prompt = f"""Classify this question as SIMPLE or COMPLEX.

SIMPLE = pure greetings or small talk with no factual question at all
(e.g. "hi", "thanks", "how are you").

COMPLEX = ANY question asking for a specific fact, rule, number, or
policy detail -- even if it sounds like something you might already
know. If the answer could differ by company or document, it is COMPLEX.

QUESTION: {query}

Respond with ONLY one word: SIMPLE or COMPLEX
"""
    response = llm.invoke(prompt).content.strip().upper()
    return "SIMPLE" if "SIMPLE" in response else "COMPLEX"