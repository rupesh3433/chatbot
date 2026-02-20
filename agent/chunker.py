"""
chunker.py — Paragraph-aware document chunker.
Groups whole paragraphs into chunks ≤ CHUNK_MAX_CHARS.
Never splits mid-sentence.
"""
import re
from datetime import datetime, timezone
from agent.config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS
from agent.db import col_documents, col_chunks


def _split_paragraphs(text: str):
    paras = re.split(r"\n{2,}", text.strip())
    result = []
    for p in paras:
        p = p.strip()
        if len(p) >= CHUNK_MIN_CHARS:
            result.append(p)
    return result


def _group_into_chunks(paragraphs):
    chunks, current = [], ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= CHUNK_MAX_CHARS:
            current += "\n\n" + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def ensure_chunks_for_all_docs() -> int:
    created = 0
    now = datetime.now(timezone.utc)
    for doc in col_documents().find({}):
        doc_id = str(doc["_id"])
        title  = doc.get("title", "untitled")
        text   = doc.get("content", doc.get("text", ""))
        if not text:
            continue
        existing = col_chunks().count_documents({"doc_id": doc_id})
        if existing > 0:
            continue
        paragraphs = _split_paragraphs(text)
        chunks     = _group_into_chunks(paragraphs)
        if not chunks:
            continue
        col_chunks().insert_many([
            {"doc_id": doc_id, "title": title,
             "chunk_text": c, "chunk_index": i, "created_at": now}
            for i, c in enumerate(chunks)
        ])
        print(f"[CHUNKER] '{title}' → {len(chunks)} chunks")
        created += len(chunks)
    return created