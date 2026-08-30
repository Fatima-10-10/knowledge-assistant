from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_groq import ChatGroq

from app.core.config import GROQ_MODEL
from app.retrieval.langchain_retriever import LangChainCompatibleRetriever
from app.retrieval.ensemble import Retriever as OurRetriever
from app.retrieval.reranker import rerank


# ---- Structured output schema (the "output parser" piece) ----
class StructuredAnswer(BaseModel):
    answer: str = Field(description="The answer to the question, grounded in the sources")
    confidence: str = Field(description="One of: high, medium, low")
    sources_used: list[int] = Field(description="Which source numbers were actually used, e.g. [1, 3]")


def build_lcel_chain(chunks):
    """
    Builds the retrieval-to-answer flow using LangChain's LCEL
    composition syntax instead of plain function calls.
    """
    our_retriever = OurRetriever(chunks)
    lc_retriever = LangChainCompatibleRetriever(our_retriever=our_retriever, chunks=chunks, k=10)

    parser = PydanticOutputParser(pydantic_object=StructuredAnswer)

    prompt = ChatPromptTemplate.from_template("""Answer the question using ONLY the sources below.
Note which source numbers you actually used.

{format_instructions}

SOURCES:
{context}

QUESTION: {question}
""")

    # PRIMARY model
    primary_llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    # FALLBACK model -- used automatically if the primary fails (rate
    # limit, timeout, etc.) -- this is the "fallbacks" piece.
    fallback_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

    def rerank_step(inputs: dict) -> dict:
        """Custom step: rerank retrieved docs before building context."""
        docs = inputs["docs"]
        question = inputs["question"]
        texts = [d.page_content for d in docs]
        reranked, _ = rerank(question, texts, top_k=min(3, len(texts)))

        context_blocks = []
        for i, (score, local_idx) in enumerate(reranked):
            context_blocks.append(f"[Source {i+1}] {texts[local_idx]}")
        return {
            "context": "\n\n".join(context_blocks),
            "question": question,
            "format_instructions": parser.get_format_instructions()
        }

    # RunnableParallel: fetch docs AND pass the question through
    # SIMULTANEOUSLY, rather than sequentially -- this is the
    # "parallelism" piece of the curriculum.
    retrieval_step = RunnableParallel(
        docs=lc_retriever,
        question=RunnablePassthrough()
    )

    chain = (
        retrieval_step
        | RunnableLambda(rerank_step)
        | prompt
        | llm_with_fallback
        | parser
    )

    return chain

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# In-memory store of chat histories, keyed by a session id
_session_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def build_conversational_chain(chunks):
    """
    Same idea as build_lcel_chain, but wrapped with memory -- so a
    follow-up question like "what about for kids under 13?" can use
    context from the PREVIOUS question automatically.
    """
    our_retriever = OurRetriever(chunks)
    lc_retriever = LangChainCompatibleRetriever(our_retriever=our_retriever, chunks=chunks, k=5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the provided context. Use conversation history for follow-up context."),
        ("placeholder", "{history}"),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}")
    ])

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)

    def retrieve_and_format(inputs: dict) -> dict:
        docs = lc_retriever.invoke(inputs["question"])
        context = "\n\n".join(d.page_content for d in docs)
        return {"context": context, "question": inputs["question"], "history": inputs.get("history", [])}

    base_chain = RunnableLambda(retrieve_and_format) | prompt | llm

    chain_with_memory = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    return chain_with_memory

from langchain_core.tools import tool


def make_search_tool(chunks):
    """
    Define a TOOL the LLM can choose to call -- this is "function
    calling inside a RAG pipeline." Instead of us always forcing
    retrieval, the LLM decides IF and WHEN to search.
    """
    our_retriever = OurRetriever(chunks)

    @tool
    def search_netflix_documents(query: str) -> str:
        """Search the Netflix Terms of Use and related documents for information relevant to the query."""
        results = our_retriever.hybrid_search(query, k=3)
        return "\n\n".join(chunks[idx].content for score, idx in results)

    return search_netflix_documents