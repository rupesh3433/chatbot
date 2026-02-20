"""
memory.py — Session conversation memory stored in MongoDB.

Keeps last MAX_HISTORY_TURNS user+bot pairs for LLM context injection.
TTL index on expires_at auto-deletes stale sessions.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from agent.config import MAX_HISTORY_TURNS, MAX_USER_MESSAGES, MAX_ASSISTANT_ANSWERS, SESSION_TTL_MINUTES
from agent.db import col_sessions

def _now(): return datetime.now(timezone.utc)
def _expiry(): return _now() + timedelta(minutes=SESSION_TTL_MINUTES)


def load_memory(session_id: str) -> dict:
    doc = col_sessions().find_one({"session_id": session_id})
    if doc is None:
        return {"session_id": session_id, "user_messages": [], "assistant_messages": [], "history_pairs": []}
    return {
        "session_id":         session_id,
        "user_messages":      doc.get("user_messages", []),
        "assistant_messages": doc.get("assistant_messages", []),
        "history_pairs":      doc.get("history_pairs", []),   # [{user, bot}, ...]
    }


def save_memory(session_id: str, user_msg: str, bot_msg: str) -> None:
    mem = load_memory(session_id)

    user_msgs = (mem["user_messages"] + [user_msg])[-MAX_USER_MESSAGES:]
    asst_msgs = (mem["assistant_messages"] + [bot_msg])[-MAX_ASSISTANT_ANSWERS:]

    # Keep full turn pairs for LLM context
    pairs = mem.get("history_pairs", [])
    pairs.append({"user": user_msg, "bot": bot_msg[:400]})   # truncate long bot answers
    pairs = pairs[-MAX_HISTORY_TURNS:]

    col_sessions().update_one(
        {"session_id": session_id},
        {"$set": {
            "session_id":         session_id,
            "user_messages":      user_msgs,
            "assistant_messages": asst_msgs,
            "history_pairs":      pairs,
            "updated_at":         _now(),
            "expires_at":         _expiry(),
        }},
        upsert=True,
    )


def get_history_for_llm(memory: dict) -> List[Dict]:
    """
    Return last N turns as a list of {"role": ..., "content": ...} dicts
    suitable for injection into the LLM prompt.
    """
    messages = []
    for pair in memory.get("history_pairs", []):
        messages.append({"role": "user",      "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["bot"]})
    return messages


def expand_with_memory(message: str, memory: dict) -> str:
    """Prepend last user message for short follow-up queries."""
    hist = memory.get("user_messages", [])
    words = message.split()
    is_short    = len(words) <= 4
    follow_words = {"this","that","it","its","they","them","those","these"}
    is_follow   = any(w in follow_words for w in words[:3])
    if (is_short or is_follow) and hist:
        return f"{hist[-1]} {message}"
    return message


def answer_memory_question(message: str, memory: dict) -> str:
    """Answer questions about the user's own chat history."""
    hist = memory.get("user_messages", [])
    if not hist:
        return "I don't have any previous questions recorded in this session yet. Feel free to ask me anything! 😊"

    low = re.sub(r"\bout\b", "our", message.lower())
    import re as _re
    if "first" in low:
        return f"Your first question in this session was:\n  ❓ \"{hist[0]}\""
    if any(w in low for w in ("last", "previous", "recent")):
        return f"Your previous question was:\n  ❓ \"{hist[-2] if len(hist) >= 2 else hist[0]}\""
    lines = ["Here are your recent questions:"]
    for i, q in enumerate(hist[-5:], 1):
        lines.append(f"  {i}. \"{q}\"")
    return "\n".join(lines)


import re
_MEM_PAT = re.compile(
    r"\b(what|which).{0,15}(first|last|previous|recent).{0,20}(chat|question|ask|message)\b"
    r"|\b(first|last|previous).{0,10}(chat|question|message)\b"
    r"|\bwhat (is|was|were) (my|our|out).{0,15}(chat|question|ask)\b"
    r"|\bfirst chat\b|\bprevious question\b|\bfirst question\b|\bmy (first|last) question\b",
    re.IGNORECASE,
)

def is_memory_question(message: str) -> bool:
    normalised = re.sub(r"\bout\b", "our", message.lower())
    return bool(_MEM_PAT.search(normalised))