"""
config.py — All settings, thresholds, and constants.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ── MongoDB ────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGO_URI", os.getenv("MONGODB_URI"))
DB_NAME     = os.getenv("MONGODB_DB_NAME", os.getenv("DB_NAME"))
COL_DOCUMENTS  = "knowledge_base"
COL_CHUNKS     = "kb_chunks"
COL_EMBEDDINGS = "kb_embeddings"
COL_SESSIONS   = "sessions"

# ── Embedding model (local, CPU, ~90MB) ───────────────────────────────────
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM    = 384
EMBED_BATCH_SIZE = 16

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K                = 6
SIM_THRESHOLD        = 0.18
MAX_CHUNKS_TO_SEARCH = 500
MAX_CONTEXT_CHARS    = 10000

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_MAX_CHARS = 900
CHUNK_MIN_CHARS = 60

# ── Session / history ─────────────────────────────────────────────────────
MAX_HISTORY_TURNS     = 8     # user+bot pairs kept for LLM context window
MAX_USER_MESSAGES     = 20
MAX_ASSISTANT_ANSWERS = 10
SESSION_TTL_MINUTES   = 120

# ── Ollama local LLM ──────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:39025")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT   = 120
LLM_MAX_TOKENS   = 600
LLM_TEMPERATURE  = 0.25   # low = factual, focused

# ── NLU classifier ────────────────────────────────────────────────────────
NLU_MIN_CONFIDENCE = 0.42

# ── VADER thresholds ──────────────────────────────────────────────────────
VADER_ANGRY_THRESHOLD = -0.50
VADER_SAD_UPPER       = -0.20
VADER_HAPPY_THRESHOLD =  0.30

# ── Continuous learning ───────────────────────────────────────────────────
RETRAIN_THRESHOLD  = 15
MIN_FALLBACK_COUNT = 4