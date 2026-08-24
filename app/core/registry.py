import sqlite3
import uuid
from datetime import datetime

DB_PATH = "data/documents.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            source_type TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            chunk_count INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_document(filename: str, source_type: str, chunk_count: int) -> str:
    doc_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO documents (id, filename, source_type, uploaded_at, chunk_count) VALUES (?, ?, ?, ?, ?)",
        (doc_id, filename, source_type, datetime.utcnow().isoformat(), chunk_count)
    )
    conn.commit()
    conn.close()
    return doc_id


def list_documents() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document(doc_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()