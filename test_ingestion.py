from app.ingestion.loaders import load_pdf, load_webpage, load_csv
pdf_docs = load_pdf("data/netflix.pdf")
print(f"Loaded {len(pdf_docs)} pages from PDF")
print(f"First page source: {pdf_docs[0].source_name}, page {pdf_docs[0].page}")
print(f"First 200 chars: {pdf_docs[0].content[:200]}")

print("\n---\n")

web_docs = load_webpage("https://www.netflix.com/signup/planform")
print(f"Loaded {len(web_docs)} document(s) from webpage")
print(f"Source: {web_docs[0].source_name}")
print(f"First 200 chars: {web_docs[0].content[:200]}")

csv_docs = load_csv("data/pricing.csv")
print(f"\nLoaded {len(csv_docs)} rows from CSV")
for doc in csv_docs:
    print(f"Row {doc.page}: {doc.content}")