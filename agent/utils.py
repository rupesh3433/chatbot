"""
utils.py — Smart intent detection: NLU model → rule-based fallback.

detect_intent_smart() is the primary entry point for app.py.
It runs NLU → if confident, uses it; otherwise falls back to rules.
"""
import re
from typing import List, Optional

# ── Rule-based constants ───────────────────────────────────────────────────
_GREETINGS = [
    "hi", "hello", "hey", "howdy", "hiya", "greetings",
    "good morning", "good afternoon", "good evening", "good night", "sup", "yo",
]
_SMALL_TALK = [
    "how are you", "how r u", "how are u", "how do you do",
    "are you a bot", "are you an ai", "are you real",
    "nice to meet you", "thank you", "thanks", "thank u", "thx", "ty",
    "ok thanks", "okay thanks", "great thanks", "bye", "goodbye",
    "see you", "take care", "cya", "what's up", "whats up",
    "good", "okay", "ok", "cool", "nice", "great", "awesome",
    "not bad", "i'm fine", "im fine", "i am fine", "noted", "got it",
]
_KB_OVERRIDES = [
    "what can you do", "what do you offer", "what do you provide",
    "what services", "can you help", "help me", "i need help",
    "are you free", "are you available", "what are your",
    "tell me about", "show me", "i want to", "i would like",
    "i need", "do you do", "can you do", "how much",
    "what is the price", "book", "appointment", "schedule",
    "your service", "your services", "your price", "give me", "tell me",
]
_SHORT_KW  = ["in short","brief","summarise","summarize","tldr","briefly",
               "1 line","one line","2 lines","in 2 lines","quick","short answer"]
_STEPS_KW  = ["step by step","steps to","how to","how do i","procedure",
               "guide me","walk me through","show me how","explain how"]
_DETAIL_KW = ["detailed","explain fully","elaborate","in detail","full detail",
               "complete details","explain everything","tell me everything"]
_BIZ_WORDS = {"book","service","services","price","cost","free","available",
               "appointment","makeup","bridal","party","henna","offer","package"}


def _rule_based(message: str) -> str:
    lower = message.lower().strip()
    clean = re.sub(r"[^\w\s]", "", lower)
    words = clean.split()

    if any(p in lower for p in _KB_OVERRIDES):
        if any(k in lower for k in _SHORT_KW):  return "short"
        if any(k in lower for k in _DETAIL_KW): return "detailed"
        if any(k in lower for k in _STEPS_KW):  return "steps"
        return "default"

    if len(words) <= 3 and (clean in _GREETINGS or
            any(clean == g or clean.startswith(g+" ") for g in _GREETINGS)):
        return "greeting"

    if any(p == lower or p == clean for p in _SMALL_TALK):
        return "small_talk"
    if (len(words) <= 4 and any(lower.startswith(p) for p in _SMALL_TALK)
            and not any(bw in lower for bw in _BIZ_WORDS)):
        return "small_talk"

    if any(k in lower for k in _SHORT_KW):  return "short"
    if any(k in lower for k in _DETAIL_KW): return "detailed"
    if any(k in lower for k in _STEPS_KW):  return "steps"
    return "default"


def detect_intent_smart(message: str) -> dict:
    """
    Full pipeline: NLU model → rule-based fallback.
    Returns dict with intent_mode, nlu_label, nlu_confidence, nlu_used.
    """
    # Step 1: NLU model
    nlu_label, nlu_conf, nlu_mode = None, 0.0, None
    try:
        from agent.nlu import predict, to_intent_mode
        nlu_result  = predict(message)
        nlu_label   = nlu_result["label"]
        nlu_conf    = nlu_result["confidence"]
        nlu_mode    = to_intent_mode(nlu_result)
    except Exception:
        pass

    # Step 2: rule-based fallback
    rule_mode = _rule_based(message)

    # NLU wins if confident; else rules win
    final_mode = nlu_mode if nlu_mode is not None else rule_mode

    return {
        "intent_mode":    final_mode,
        "nlu_label":      nlu_label,
        "nlu_confidence": nlu_conf,
        "nlu_used":       nlu_mode is not None,
    }


def is_conversational(intent_mode: str) -> bool:
    return intent_mode in ("greeting", "small_talk")


def extract_keywords(text: str) -> List[str]:
    STOP = {"i","me","my","we","our","you","your","the","a","an","is","are","was","were",
            "be","been","do","does","did","have","has","had","will","would","could","should",
            "may","might","can","to","of","in","on","at","for","with","and","or","but","not",
            "what","which","who","this","that","it","its","so","if","about","how","why","when",
            "tell","show","give","get","need","want","please","all","just","any","some","help"}
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOP and len(w) > 2]


def safe_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    for sep in (". ", "! ", "? ", "\n"):
        idx = text.rfind(sep, 0, max_chars)
        if idx > max_chars // 2:
            return text[:idx+1].strip()
    idx = text.rfind(" ", 0, max_chars)
    return text[:idx].strip() if idx > 0 else text[:max_chars]