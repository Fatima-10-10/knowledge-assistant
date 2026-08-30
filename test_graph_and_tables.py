from app.ingestion.loaders import load_pdf
from app.ingestion.splitters import chunk_documents, tag_categories
from app.graph.knowledge_graph import KnowledgeGraph
from app.ingestion.tables import extract_tables_from_pdf, table_to_text

pdf_docs = load_pdf("data/netflix.pdf")
chunks = tag_categories(chunk_documents(pdf_docs, chunk_size=500))

print("=== Building knowledge graph (this makes several LLM calls, may take a bit) ===")
kg = KnowledgeGraph()
kg.build_from_chunks(chunks, max_chunks=15)  # limit for speed during testing

summary = kg.summary()
print(f"Entities found: {summary['num_entities']}")
print(f"Relationships found: {summary['num_relationships']}")
print(f"Entity names: {summary['entities']}")

if summary['entities']:
    test_entity = summary['entities'][0]
    print(f"\n=== Relationships for '{test_entity}' ===")
    for rel in kg.get_relationships_for(test_entity):
        print(f"  {rel['source']} --[{rel['relation']}]--> {rel['target']}")

    print(f"\n=== Entities within 1 hop of '{test_entity}' ===")
    related = kg.get_related_entities(test_entity, max_hops=1)
    print(f"  {related}")

print("\n\n=== Table extraction ===")
tables = extract_tables_from_pdf("data/netflix.pdf")
print(f"Found {len(tables)} table(s) in the PDF")
for t in tables[:2]:
    print(f"\nTable on page {t['page']}:")
    print(f"Headers: {t['headers']}")
    print(table_to_text(t)[:300])