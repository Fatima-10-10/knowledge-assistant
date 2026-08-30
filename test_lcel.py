from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.core.lcel_pipeline import build_lcel_chain, build_conversational_chain, make_search_tool

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))

print("=== LCEL chain with structured output + reranking + fallback model ===")
chain = build_lcel_chain(chunks)
result = chain.invoke("Can minors use Netflix on their own?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources used: {result.sources_used}")

print("\n\n=== Conversational chain with memory ===")
conv_chain = build_conversational_chain(chunks)
config = {"configurable": {"session_id": "test-session-1"}}

r1 = conv_chain.invoke({"question": "What's the age requirement to use Netflix?"}, config=config)
print(f"Q1 Answer: {r1.content}")

r2 = conv_chain.invoke({"question": "What about for Extra Members specifically?"}, config=config)
print(f"Q2 Answer (should use context from Q1): {r2.content}")

print("\n\n=== Tool / function calling ===")
search_tool = make_search_tool(chunks)
print(f"Tool name: {search_tool.name}")
print(f"Tool description: {search_tool.description}")
tool_result = search_tool.invoke("cancellation policy")
print(f"Tool result (first 300 chars): {tool_result[:300]}")