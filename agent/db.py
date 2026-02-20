"""
db.py — MongoDB connection singleton + collection helpers.
"""
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from agent.config import MONGODB_URI, DB_NAME, COL_DOCUMENTS, COL_CHUNKS, COL_EMBEDDINGS, COL_SESSIONS

_client = None
_db     = None

def get_db() -> Database:
    global _client, _db
    if _db is None:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000)
        _db = _client[DB_NAME]
        _ensure_indexes(_db)
    return _db

def _ensure_indexes(db):
    db[COL_CHUNKS].create_index([("doc_id", ASCENDING)], background=True)
    db[COL_EMBEDDINGS].create_index([("chunk_id", ASCENDING)], unique=True, background=True)
    db[COL_SESSIONS].create_index([("session_id", ASCENDING)], unique=True, background=True)
    db[COL_SESSIONS].create_index([("expires_at", ASCENDING)], expireAfterSeconds=0, background=True)
    # Interaction logs
    try:
        db["interaction_logs"].create_index([("created_at", ASCENDING)], background=True)
        db["kb_gaps"].create_index([("query", ASCENDING)], unique=True, background=True)
        db["nlu_corrections"].create_index([("used_for_training", ASCENDING)], background=True)
    except Exception:
        pass

def col_documents()  -> Collection: return get_db()[COL_DOCUMENTS]
def col_chunks()     -> Collection: return get_db()[COL_CHUNKS]
def col_embeddings() -> Collection: return get_db()[COL_EMBEDDINGS]
def col_sessions()   -> Collection: return get_db()[COL_SESSIONS]