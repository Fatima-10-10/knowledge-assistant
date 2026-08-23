import json
from app.core.llm import get_llm
from app.ingestion.loaders import SourceDocument
from app.retrieval.ensemble import Retriever


def contextual_compression(query: str, chunks_with_scores: list, llm=None) -> list[str]:
    """
    Trim each retrieved chunk down to only the sentences relevant to
    the query, before sending it to the final answer-generation step.
    Saves tokens and reduces noise the LLM has to wade through.
    """
    llm = llm or get_llm()
    compressed = []

    for score, doc in chunks_with_scores:
        prompt = f"""Extract ONLY the sentence(s) from the text below that are
directly relevant to answering the question. If nothing is relevant,
respond with exactly: NOT_RELEVANT

QUESTION: {query}

TEXT:
{doc.content}
"""
        response = llm.invoke(prompt).content.strip()
        if response != "NOT_RELEVANT":
            compressed.append(response)

    return compressed


class ParentDocumentRetriever:
    """
    Small-to-big retrieval: search over small, precise CHUNKS (good for
    matching), but return the full PAGE they came from (good for
    context) — so the LLM sees the whole surrounding clause, not just
    a fragment.
    """

    def __init__(self, chunks: list[SourceDocument], parent_docs: list[SourceDocument]):
        self.retriever = Retriever(chunks)
        self.chunks = chunks
        # Map each (source_name, page) -> the full parent page content
        self.parent_lookup = {
            (d.source_name, d.page): d.content for d in parent_docs
        }

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        results = self.retriever.hybrid_search(query, k=k)
        parents = []
        seen = set()
        for score, idx in results:
            chunk = self.chunks[idx]
            key = (chunk.source_name, chunk.page)
            if key not in seen:
                seen.add(key)
                parent_text = self.parent_lookup.get(key, chunk.content)
                parents.append(parent_text)
        return parents


def self_query(query: str, llm=None) -> dict:
    """
    Turn a natural-language question into a structured filter + a
    cleaned semantic query. E.g. "what does the Age Limitation clause
    say" -> {"category_filter": "Age Limitation", "semantic_query": "age requirement"}
    """
    llm = llm or get_llm()
    prompt = f"""Analyze this question about a Netflix Terms of Use document.
Extract a category filter if the question mentions a specific topic
(like "Age Limitation", "Account Sharing", "Offers"), otherwise null.
Also produce a cleaned-up semantic search query.

Respond ONLY with valid JSON in this exact format:
{{"category_filter": "<category or null>", "semantic_query": "<cleaned query>"}}

QUESTION: {query}
"""
    response = llm.invoke(prompt).content.strip()
    # Strip markdown code fences if the model adds them
    response = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"category_filter": None, "semantic_query": query}


def multi_query(query: str, llm=None) -> list[str]:
    """
    Generate 3 paraphrased versions of the question to widen retrieval
    coverage — helps when the user's exact phrasing doesn't match the
    document's wording.
    """
    llm = llm or get_llm()
    prompt = f"""Generate 3 different ways to ask this same question,
using different wording. Respond with ONLY the 3 questions, one per
line, no numbering.

QUESTION: {query}
"""
    response = llm.invoke(prompt).content.strip()
    variants = [line.strip() for line in response.split("\n") if line.strip()]
    return [query] + variants[:3]  # include the original too


def rewrite_query(query: str, llm=None) -> str:
    """Clean up a vague/poorly-worded query into a clearer search query."""
    llm = llm or get_llm()
    prompt = f"""Rewrite this question to be clearer and more specific for
searching a legal document. Respond with ONLY the rewritten question.

QUESTION: {query}
"""
    return llm.invoke(prompt).content.strip()


def step_back_prompt(query: str, llm=None) -> str:
    """
    Generate a more general, higher-level question first — helps
    retrieve broader context before drilling into specifics.
    E.g. "can a 15 year old with their own card sign up?" ->
    "what are Netflix's age requirements?"
    """
    llm = llm or get_llm()
    prompt = f"""Given this specific question, write a more general,
"step back" question that captures the broader topic. Respond with
ONLY the general question.

SPECIFIC QUESTION: {query}
"""
    return llm.invoke(prompt).content.strip()


def decompose_query(query: str, llm=None) -> list[str]:
    """
    Break a complex, multi-part question into simpler sub-questions
    that can each be retrieved independently, then combined.
    """
    llm = llm or get_llm()
    prompt = f"""If this question has multiple parts, break it into
separate simple sub-questions, one per line, no numbering. If it's
already simple, just repeat it as-is on one line.

QUESTION: {query}
"""
    response = llm.invoke(prompt).content.strip()
    return [line.strip() for line in response.split("\n") if line.strip()]