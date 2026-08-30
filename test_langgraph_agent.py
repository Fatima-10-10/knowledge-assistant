import time
from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.agents.graph import make_graph, make_planner_graph

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))

agent = make_graph(chunks)

print("=== Main agent graph: the failing Unit 9 query ===")
result = agent.invoke({
    "question": "getting kicked off my plan",
    "original_question": "getting kicked off my plan",
    "history": [],
    "retrieved_chunks": [],
    "grade_result": {},
    "retry_count": 0,
    "max_retries": 2,
    "source": "documents",
    "answer": "",
    "revised": False
})
print(f"Source used: {result['source']}")
print(f"Retries: {result['retry_count']}")
print(f"Answer: {result['answer']}")

# print("\nWaiting 20s to avoid rate limits...")
# time.sleep(20)

# print("\n\n=== Main agent graph: testing the Unit 12 memory gap fix ===")
# result2 = agent.invoke({
#     "question": "What about for Extra Members specifically?",
#     "original_question": "What about for Extra Members specifically?",
#     "history": ["What's the age requirement to use Netflix?", "You must be 18+."],
#     "retrieved_chunks": [],
#     "grade_result": {},
#     "retry_count": 0,
#     "max_retries": 2,
#     "source": "documents",
#     "answer": "",
#     "revised": False
# })
# print(f"Answer (should now use history for retrieval): {result2['answer']}")

# print("\nWaiting 20s to avoid rate limits...")
# time.sleep(20)

# print("\n\n=== Planner/Retriever/Critic multi-agent graph ===")
# planner_graph = make_planner_graph(chunks)
# result3 = planner_graph.invoke({
#     "question": "What's the age requirement and can I share my account with family?",
#     "sub_questions": [],
#     "sub_answers": [],
#     "final_answer": ""
# })
# print(f"Sub-questions generated: {result3['sub_questions']}")
# print(f"\nFinal combined answer: {result3['final_answer']}")

# print("\n\n=== DONE ===")