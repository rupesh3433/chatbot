"""
retriever.py — Cosine similarity retrieval over MongoDB embeddings.
Memory-safe: streaming cursor + min-heap, never loads all vectors at once.
"""
import heapq
from typing import List
import numpy as np
from bson import ObjectId
from agent.config import TOP_K, SIM_THRESHOLD, MAX_CHUNKS_TO_SEARCH
from agent.db import col_embeddings, col_chunks


def retrieve_top_chunks(query_vec: np.ndarray, top_k: int = TOP_K) -> List[dict]:
    heap, seen = [], 0
    for doc in col_embeddings().find({}, {"chunk_id":1,"embedding":1,"_id":0}):
        if seen >= MAX_CHUNKS_TO_SEARCH:
            break
        seen += 1
        score    = float(np.dot(query_vec, np.array(doc["embedding"], dtype=np.float32)))
        chunk_id = doc["chunk_id"]
        if len(heap) < top_k:
            heapq.heappush(heap, (score, chunk_id))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, chunk_id))

    results = sorted(heap, reverse=True, key=lambda x: x[0])
    enriched = []
    for score, cid in results:
        if score < SIM_THRESHOLD:
            continue
        try:
            oid = ObjectId(cid)
        except Exception:
            oid = cid
        chunk = col_chunks().find_one({"_id": oid}, {"doc_id":1,"title":1,"chunk_text":1})
        if chunk:
            enriched.append({
                "chunk_id":   cid,
                "doc_id":     str(chunk.get("doc_id", "")),
                "title":      chunk.get("title", ""),
                "chunk_text": chunk.get("chunk_text", ""),
                "score":      round(score, 4),
            })
    return enriched