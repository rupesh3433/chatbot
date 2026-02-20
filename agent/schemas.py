"""
schemas.py — Pydantic request/response models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(..., example="user_abc123")
    message:    str = Field(..., example="What services do you offer?")

class ChunkMeta(BaseModel):
    chunk_id:   str
    doc_id:     str
    title:      str
    similarity: float
    preview:    str

class ChatResponse(BaseModel):
    session_id:         str
    answer_text:        str
    intent_mode:        str
    emotion_label:      str
    emotion_confidence: float
    used_sources:       int
    top_chunks:         List[ChunkMeta]
    llm_used:           bool = False