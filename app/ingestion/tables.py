import pdfplumber


def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract tables as STRUCTURED data (rows/columns), not flattened
    text -- so a table's rows/columns don't get mangled by our text
    chunker, which would otherwise lose the table structure entirely.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()
            for t_idx, table in enumerate(page_tables):
                if table and len(table) > 1:  # has header + at least 1 row
                    tables.append({
                        "page": page_num,
                        "table_index": t_idx,
                        "headers": table[0],
                        "rows": table[1:]
                    })
    return tables


def table_to_text(table: dict) -> str:
    """Turn a structured table into readable text for embedding/search."""
    headers = table["headers"]
    lines = []
    for row in table["rows"]:
        row_text = ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)
        lines.append(row_text)
    return "\n".join(lines)