"""
app.py — FastAPI entry point for JinniChirag MUA AI Chatbot.
Version 3.0 — Ollama LLM + NLU + Fuzzy + Memory + Trainer

COMPLETE PIPELINE per request:
  1.  Load session memory (MongoDB)
  2.  Fuzzy-correct the query (typos, slang)
  3.  Smart intent detection (NLU model → rule-based fallback)
  4.  Emotion detection (VADER + rules)
  5.  Memory meta-question? → answer from history
  6.  Greeting / small-talk? → canned reply
  7.  Expand vague query for better retrieval
  8.  Ensure KB is chunked + embedded
  9.  Retrieve top-k chunks by cosine similarity
 10.  Call local LLM (Ollama) with KB context + conversation history
 11.  Save turn to memory
 12.  Log interaction for continuous learning
 13.  Return structured ChatResponse

SETUP (one-time):
  pip install -r requirements.txt
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llama3.2        # ~2 GB, fully offline after this
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

ADMIN ENDPOINTS:
  GET  /health            → system health check
  GET  /admin/gaps        → unanswered KB gaps to add content for
  GET  /admin/logs        → recent interaction logs
  POST /admin/retrain     → force NLU retraining
  POST /admin/correct     → label a misclassified message
"""

import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent.schemas       import ChatRequest, ChatResponse, ChunkMeta
from agent.config        import TOP_K, SIM_THRESHOLD, OLLAMA_MODEL
from trainnlp.fuzzy_match   import correct_query
from agent.utils         import detect_intent_smart, is_conversational, extract_keywords, safe_truncate
from agent.emotion       import detect_emotion
from agent.memory        import load_memory, save_memory, get_history_for_llm, is_memory_question, answer_memory_question, expand_with_memory
from agent.chunker       import ensure_chunks_for_all_docs
from agent.embedder      import ensure_embeddings_for_all_chunks, embed_query
from agent.retriever     import retrieve_top_chunks
from trainnlp.llm_engine    import generate_answer, check_ollama_health
from agent.conversational import get_conversational_reply
from trainnlp.trainer       import log_interaction, init_trainer, add_correction, force_retrain, get_kb_gaps, get_recent_logs


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JinniChirag MUA — AI Chatbot",
    description="Intelligent local LLM chatbot with RAG, memory, and NLU.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup():
    init_trainer()
    print(f"[STARTUP] Model: {OLLAMA_MODEL}")
    h = check_ollama_health()
    if h["running"]:
        print(f"[STARTUP] Ollama running ✓  model_available={h['model_available']}")
        print(f"[STARTUP] Available models: {h['models']}")
    else:
        print("[STARTUP] ⚠️  Ollama not running. Start with: ollama serve")


@app.get("/", include_in_schema=False)
async def serve_ui():
    p = STATIC_DIR / "index.html"
    if not p.exists():
        return JSONResponse({"status": "ok", "message": "JinniChirag AI Chatbot API v3.0"})
    return FileResponse(str(p), media_type="text/html")


# ══════════════════════════════════════════════════════════════════════════
# POST /chat — main pipeline
# ══════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id.strip()
    raw_msg    = req.message.strip()
    if not raw_msg:
        raise HTTPException(400, "Message cannot be empty.")

    # ── 1. Load memory ─────────────────────────────────────────────────────
    memory = load_memory(session_id)

    # ── 2. Fuzzy correction ────────────────────────────────────────────────
    corrected = correct_query(raw_msg)

    # ── 3. Intent detection (NLU + rules) ─────────────────────────────────
    intent_result  = detect_intent_smart(corrected)
    intent_mode    = intent_result["intent_mode"]
    nlu_label      = intent_result["nlu_label"]
    nlu_confidence = intent_result["nlu_confidence"]

    # ── 4. Emotion detection ───────────────────────────────────────────────
    em_result     = detect_emotion(raw_msg)
    emotion_label = em_result["label"]
    emotion_conf  = em_result["confidence"]

    print(f"\n[PIPELINE] msg='{raw_msg[:60]}'")
    print(f"           corrected='{corrected[:60]}'")
    print(f"           intent={intent_mode}  nlu={nlu_label}({nlu_confidence:.2f})")
    print(f"           emotion={emotion_label}({emotion_conf:.2f})")

    # ── 5. Memory meta-questions ───────────────────────────────────────────
    if intent_mode == "memory" or is_memory_question(raw_msg):
        answer = answer_memory_question(raw_msg, memory)
        _finish(session_id, raw_msg, answer, memory)
        return _resp(session_id, answer, "memory", emotion_label, emotion_conf, 0, [], False)

    # ── 6. Greeting / small-talk ───────────────────────────────────────────
    if is_conversational(intent_mode):
        answer = get_conversational_reply(intent_mode, emotion_label)
        _finish(session_id, raw_msg, answer, memory)
        return _resp(session_id, answer, intent_mode, emotion_label, emotion_conf, 0, [], False)

    # ── 7. Query expansion for retrieval ──────────────────────────────────
    search_query = _expand_query(corrected, memory)
    print(f"           search='{search_query[:60]}'")

    # ── 8. Chunking + embedding (lazy) ────────────────────────────────────
    ensure_chunks_for_all_docs()
    ensure_embeddings_for_all_chunks()

    # ── 9. Retrieval ───────────────────────────────────────────────────────
    query_vec    = embed_query(search_query)
    chunks       = retrieve_top_chunks(query_vec, top_k=TOP_K)
    best_score   = chunks[0]["score"] if chunks else 0.0
    used_sources = len(chunks)
    was_fallback = best_score < SIM_THRESHOLD or not chunks

    print(f"           chunks={used_sources}  best={best_score:.4f}  fallback={was_fallback}")

    # ── 10. LLM generation ─────────────────────────────────────────────────
    history_pairs = memory.get("history_pairs", [])

    # Get NLU content hint
    nlu_hint = None
    try:
        from trainnlp.nlu import predict, content_hint
        nlu_hint = content_hint(predict(corrected))
    except Exception:
        pass

    llm_result = generate_answer(
        user_message  = raw_msg,
        kb_chunks     = chunks,
        history_pairs = history_pairs,
        emotion       = emotion_label,
        intent_mode   = intent_mode,
        nlu_hint      = nlu_hint,
    )
    answer   = llm_result["answer"]
    llm_used = llm_result["llm_used"]

    # ── 11. Save memory ────────────────────────────────────────────────────
    _finish(session_id, raw_msg, answer, memory)

    # ── 12. Log interaction ────────────────────────────────────────────────
    _log(session_id, raw_msg, corrected, intent_mode, nlu_label,
         nlu_confidence, emotion_label, used_sources, best_score,
         was_fallback, llm_used, answer)

    # ── 13. Build response ─────────────────────────────────────────────────
    chunk_metas = [
        ChunkMeta(
            chunk_id=c["chunk_id"], doc_id=c["doc_id"], title=c["title"],
            similarity=c["score"], preview=c["chunk_text"][:120],
        )
        for c in chunks
    ]
    return _resp(session_id, answer, intent_mode, emotion_label,
                 emotion_conf, used_sources, chunk_metas, llm_used)


# ══════════════════════════════════════════════════════════════════════════
# Admin + health endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    ollama = check_ollama_health()
    return {
        "status":          "ok",
        "ollama_running":  ollama["running"],
        "model_available": ollama.get("model_available", False),
        "model":           OLLAMA_MODEL,
        "available_models":ollama.get("models", []),
    }


@app.get("/admin/gaps")
async def admin_gaps(limit: int = 20):
    return {"gaps": get_kb_gaps(limit), "count": len(get_kb_gaps(limit))}


@app.get("/admin/logs")
async def admin_logs(limit: int = 50):
    return {"logs": get_recent_logs(limit)}


@app.post("/admin/retrain")
async def admin_retrain():
    ok = force_retrain()
    return {"status": "ok" if ok else "error", "retrained": ok}


class CorrectionBody(BaseModel):
    message:        str
    correct_intent: str

@app.post("/admin/correct")
async def admin_correct(body: CorrectionBody):
    ok = add_correction(body.message, body.correct_intent)
    return {"status": "ok" if ok else "invalid_intent", "saved": ok}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _resp(session_id, answer, intent_mode, emotion_label, emotion_conf,
          used_sources, top_chunks, llm_used) -> ChatResponse:
    return ChatResponse(
        session_id=session_id, answer_text=answer,
        intent_mode=intent_mode, emotion_label=emotion_label,
        emotion_confidence=emotion_conf, used_sources=used_sources,
        top_chunks=top_chunks, llm_used=llm_used,
    )


def _finish(session_id: str, user_msg: str, answer: str, memory: dict):
    save_memory(session_id, user_msg, answer)


def _log(session_id, raw_msg, corrected, intent_mode, nlu_label,
         nlu_conf, emotion_label, used_sources, best_score,
         was_fallback, llm_used, answer):
    try:
        log_interaction(
            session_id=session_id, user_message=raw_msg,
            corrected_query=corrected, intent_mode=intent_mode,
            nlu_label=nlu_label, nlu_confidence=nlu_conf,
            emotion_label=emotion_label, used_sources=used_sources,
            best_score=best_score, was_fallback=was_fallback,
            llm_used=llm_used, answer_preview=answer,
        )
    except Exception:
        pass


_VAGUE_EXPANSIONS = [
    (r"\bwhat can you do\b",              "services bridal party makeup henna booking pricing"),
    (r"\bwhat do you (do|offer|provide)\b","services bridal party makeup henna"),
    (r"\bwhat (are|is) (your )?services\b","services bridal party makeup henna mehendi"),
    (r"\bwho are you\b",                  "about JinniChirag Makeup Artist Chirag Sharma experience"),
    (r"\btell me about (your)?self\b",    "about JinniChirag Makeup Artist Chirag Sharma"),
    (r"\bhow much\b",                     "pricing cost bridal party makeup packages"),
    (r"\bprice\b",                        "pricing cost bridal party makeup packages"),
    (r"\bhow (to |do i )?book\b",         "booking appointment process steps OTP confirmation"),
    (r"\bare you (free|available)\b",     "booking availability appointment slot schedule"),
    (r"\bcontact\b",                      "contact phone email WhatsApp Instagram"),
    (r"\byour (phone|number|email)\b",    "contact phone email WhatsApp Instagram"),
    (r"\bfollower\b",                     "social media followers Instagram YouTube TikTok"),
    (r"\binstagram\b",                    "social media Instagram link followers"),
]


def _expand_query(corrected: str, memory: dict) -> str:
    lower = corrected.lower()
    for pattern, expansion in _VAGUE_EXPANSIONS:
        if re.search(pattern, lower):
            return expansion
    return expand_with_memory(corrected, memory)