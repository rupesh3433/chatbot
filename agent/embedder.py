"""
embedder.py — Local CPU embedding via sentence-transformers/all-MiniLM-L6-v2.
Lazy singleton. Batch-processes missing chunks. Unit-normalised for cosine sim.
"""
from datetime import datetime, timezone
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from agent.config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBED_BATCH_SIZE
from agent.db import col_chunks, col_embeddings

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model

def _embed(texts: List[str]) -> np.ndarray:
    return _get_model().encode(
        texts, batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)

def ensure_embeddings_for_all_chunks() -> int:
    existing = {d["chunk_id"] for d in col_embeddings().find({}, {"chunk_id":1,"_id":0})}
    pending  = [{"id": str(c["_id"]), "text": c["chunk_text"]}
                for c in col_chunks().find({}, {"chunk_text":1})
                if str(c["_id"]) not in existing]
    if not pending:
        return 0
    now = datetime.now(timezone.utc)
    created = 0
    for i in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[i:i+EMBED_BATCH_SIZE]
        vecs  = _embed([b["text"] for b in batch])
        col_embeddings().insert_many([
            {"chunk_id": b["id"], "embedding": vecs[j].tolist(),
             "embedding_dim": EMBEDDING_DIM, "updated_at": now}
            for j, b in enumerate(batch)
        ], ordered=False)
        created += len(batch)
    return created

def embed_query(query: str) -> np.ndarray:
    return _embed([query])[0]