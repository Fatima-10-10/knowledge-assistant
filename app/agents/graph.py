from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

from app.retrieval.ensemble import Retriever
from app.retrieval.reranker import rerank
from app.agents.self_correction import (
    grade_documents, contextual_rewrite_query, contextualize_web_query,
    web_search_fallback, generate_with_self_critique
)
from app.core.llm import get_llm


class AgentState(TypedDict):
    """
    The shared state that flows through every node in the graph.
    This is what fixes Unit 12's memory gap -- 'history' carries
    forward automatically, so retrieval nodes can see it too, not
    just the final generation step.
    """
    question: str
    original_question: str
    history: list[str]          # previous Q&A pairs in this session
    retrieved_chunks: list[str]
    grade_result: dict
    retry_count: int
    max_retries: int
    source: str                  # "documents" or "web_fallback"
    answer: str
    revised: bool

def make_graph(chunks):
    """
    Builds the LangGraph agent: retrieve -> grade -> (rewrite -> retry
    loop) -> generate -> critique -> return. This IS the cyclic
    workflow the curriculum describes -- grade/rewrite/re-retrieve as
    an actual loop with conditional routing, not just sequential code.
    """
    retriever = Retriever(chunks)
    llm = get_llm()

    # ---- NODE: Retrieve ----
    def retrieve_node(state: AgentState) -> dict:
        # Use conversation history to enrich the search query -- this
        # is the Unit 12 memory-retrieval fix: fold prior context IN
        # before searching, not just at generation time.
        search_query = state["question"]
        if state.get("history"):
            search_query = f"{' '.join(state['history'][-2:])} {state['question']}"

        results = retriever.hybrid_search(search_query, k=10)
        chunk_texts = [chunks[idx].content for score, idx in results]
        return {"retrieved_chunks": chunk_texts}

    # ---- NODE: Grade ----
    def grade_node(state: AgentState) -> dict:
        grade = grade_documents(state["original_question"], state["retrieved_chunks"], llm=llm)
        return {"grade_result": grade}

    # ---- CONDITIONAL EDGE: decide what to do based on the grade ----
    def route_after_grade(state: AgentState) -> str:
        if state["grade_result"]["relevant"]:
            return "generate"
        if state["retry_count"] >= state["max_retries"]:
            return "web_fallback"
        return "rewrite"

    # ---- NODE: Rewrite (grounded in document anchor, per our Unit 10 fix) ----
    def rewrite_node(state: AgentState) -> dict:
        anchor = chunks[0].content if chunks else ""
        new_query = contextual_rewrite_query(state["question"], anchor, llm=llm)
        return {"question": new_query, "retry_count": state["retry_count"] + 1}

    # ---- NODE: Web fallback ----
    def web_fallback_node(state: AgentState) -> dict:
        anchor = chunks[0].content if chunks else ""
        contextualized = contextualize_web_query(state["original_question"], anchor, llm=llm)
        web_results = web_search_fallback(contextualized)
        return {"retrieved_chunks": web_results, "source": "web_fallback"}

    # ---- NODE: Generate (with Self-RAG critique built in) ----
    def generate_node(state: AgentState) -> dict:
        print(f"DEBUG: retrieved_chunks count = {len(state['retrieved_chunks'])}")
        print(f"DEBUG: first chunk preview = {state['retrieved_chunks'][0][:100] if state['retrieved_chunks'] else 'EMPTY'}")
        result = generate_with_self_critique(
            state["original_question"], state["retrieved_chunks"], llm=llm
        )
        print(f"DEBUG: draft answer = {repr(result['answer'][:200])}")
        source = state.get("source", "documents")
        return {"answer": result["answer"], "revised": result["revised"], "source": source}
    # ---- Build the graph ----
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("web_fallback", web_fallback_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")

    # This is the CYCLIC part -- grade routes to rewrite, which loops
    # back to retrieve, forming an actual cycle in the graph.
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "web_fallback": "web_fallback"}
    )
    graph.add_edge("rewrite", "retrieve")  # <-- the loop
    graph.add_edge("web_fallback", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

class PlannerState(TypedDict):
    question: str
    sub_questions: list[str]
    sub_answers: Annotated[list[str], operator.add]
    final_answer: str


def make_planner_graph(chunks):
    """
    Multi-agent pattern: a PLANNER breaks a complex question into
    sub-questions, RETRIEVER agents answer each independently, a
    CRITIC/synthesizer combines them into one final answer.
    """
    llm = get_llm()
    retriever = Retriever(chunks)

    def planner_node(state: PlannerState) -> dict:
        prompt = f"""Break this question into 2-3 simple sub-questions
if it has multiple parts. If it's already simple, return it as the
only sub-question. One per line, no numbering.

QUESTION: {state['question']}
"""
        response = llm.invoke(prompt).content.strip()
        sub_qs = [line.strip() for line in response.split("\n") if line.strip()]
        return {"sub_questions": sub_qs}

    def retriever_node(state: PlannerState) -> dict:
        # Answer ALL sub-questions (in this simple version, sequentially)
        answers = []
        for sub_q in state["sub_questions"]:
            results = retriever.hybrid_search(sub_q, k=3)
            context = "\n\n".join(chunks[idx].content for score, idx in results)
            prompt = f"Answer using only this context:\n{context}\n\nQuestion: {sub_q}"
            answer = llm.invoke(prompt).content
            answers.append(f"Q: {sub_q}\nA: {answer}")
        return {"sub_answers": answers}

    def critic_node(state: PlannerState) -> dict:
        combined = "\n\n".join(state["sub_answers"])
        prompt = f"""Combine these sub-answers into one clear, coherent
final answer to the original question.

ORIGINAL QUESTION: {state['question']}

SUB-ANSWERS:
{combined}

Write ONE combined answer.
"""
        final = llm.invoke(prompt).content
        return {"final_answer": final}

    graph = StateGraph(PlannerState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")
    graph.add_edge("critic", END)

    return graph.compile()