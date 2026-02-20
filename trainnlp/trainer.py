"""
trainer.py — Interaction logging + continuous NLU retraining.

Logs every chat turn to MongoDB.
Tracks repeated fallback queries as KB gaps.
Admin can label misclassified messages → retrain NLU.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from agent.config import RETRAIN_THRESHOLD, MIN_FALLBACK_COUNT

logger = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def _col(name: str):
    from agent.db import get_db
    return get_db()[name]


# ── Interaction log ────────────────────────────────────────────────────────

def log_interaction(
    session_id:      str,
    user_message:    str,
    corrected_query: str,
    intent_mode:     str,
    nlu_label:       Optional[str],
    nlu_confidence:  Optional[float],
    emotion_label:   str,
    used_sources:    int,
    best_score:      float,
    was_fallback:    bool,
    llm_used:        bool,
    answer_preview:  str,
) -> None:
    try:
        _col("interaction_logs").insert_one({
            "session_id":      session_id,
            "user_message":    user_message,
            "corrected_query": corrected_query,
            "intent_mode":     intent_mode,
            "nlu_label":       nlu_label,
            "nlu_confidence":  nlu_confidence,
            "emotion_label":   emotion_label,
            "used_sources":    used_sources,
            "best_score":      best_score,
            "was_fallback":    was_fallback,
            "llm_used":        llm_used,
            "answer_preview":  answer_preview[:250],
            "created_at":      _now(),
        })
        if was_fallback:
            _track_gap(user_message)
    except Exception as e:
        logger.warning(f"[TRAINER] log error: {e}")


def _track_gap(query: str) -> None:
    try:
        result = _col("kb_gaps").find_one_and_update(
            {"query": query.lower()[:200]},
            {"$inc": {"count": 1}, "$set": {"last_seen": _now()},
             "$setOnInsert": {"query": query.lower()[:200], "created_at": _now(), "flagged": False}},
            upsert=True, return_document=True,
        )
        if result and result.get("count", 0) >= MIN_FALLBACK_COUNT and not result.get("flagged"):
            _col("kb_gaps").update_one(
                {"query": query.lower()[:200]},
                {"$set": {"flagged": True, "flagged_at": _now()}}
            )
            print(f"[TRAINER] KB gap flagged: '{query[:60]}'")
    except Exception as e:
        logger.warning(f"[TRAINER] gap track error: {e}")


# ── Correction API ────────────────────────────────────────────────────────

VALID_INTENTS = {
    "greeting", "small_talk", "services", "pricing", "booking",
    "contact", "about", "social", "tips", "events", "faq",
    "short", "steps", "detailed", "memory", "general",
}


def add_correction(message: str, correct_intent: str) -> bool:
    if correct_intent not in VALID_INTENTS:
        return False
    try:
        _col("nlu_corrections").insert_one({
            "message": message, "correct_intent": correct_intent,
            "used_for_training": False, "created_at": _now(),
        })
        retrain_if_due()
        return True
    except Exception as e:
        logger.warning(f"[TRAINER] correction error: {e}")
        return False


def retrain_if_due() -> bool:
    try:
        count = _col("nlu_corrections").count_documents({"used_for_training": False})
        if count >= RETRAIN_THRESHOLD:
            return force_retrain()
    except Exception:
        pass
    return False


def force_retrain() -> bool:
    try:
        docs  = list(_col("nlu_corrections").find({}))
        extra: List[Tuple[str, str]] = [(d["message"], d["correct_intent"]) for d in docs]
        from trainnlp.nlu import retrain
        retrain(extra)
        pending_ids = [d["_id"] for d in docs if not d.get("used_for_training")]
        if pending_ids:
            _col("nlu_corrections").update_many(
                {"_id": {"$in": pending_ids}},
                {"$set": {"used_for_training": True, "trained_at": _now()}}
            )
        print(f"[TRAINER] Retrained with {len(extra)} corrections")
        return True
    except Exception as e:
        logger.error(f"[TRAINER] retrain error: {e}")
        return False


def get_kb_gaps(limit: int = 20) -> list:
    try:
        return [
            {"query": d.get("query"), "count": d.get("count", 0), "flagged_at": d.get("flagged_at")}
            for d in _col("kb_gaps").find({"flagged": True}).sort("count", -1).limit(limit)
        ]
    except Exception:
        return []


def get_recent_logs(limit: int = 50) -> list:
    try:
        return list(_col("interaction_logs").find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
    except Exception:
        return []


def init_trainer():
    """Called at startup — ensures indexes exist and NLU model is loaded."""
    try:
        from trainnlp.nlu import _load
        _load()
        print("[TRAINER] NLU model ready")
    except Exception as e:
        logger.warning(f"[TRAINER] init: {e}")