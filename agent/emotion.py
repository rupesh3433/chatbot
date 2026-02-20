"""
emotion.py — Emotion detection: rule overrides + VADER sentiment.

Labels: angry | sad | confused | excited | happy | neutral
"""
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from agent.config import VADER_ANGRY_THRESHOLD, VADER_SAD_UPPER, VADER_HAPPY_THRESHOLD

_vader = SentimentIntensityAnalyzer()

# Business/service words → skip confusion detection (no false positives)
_SVC_RE = re.compile(
    r"\b(service|services|book|booking|price|cost|offer|provide|available|"
    r"appointment|bridal|makeup|henna|party|chirag|jinni|artist|contact|"
    r"phone|email|instagram|youtube|about|give me|tell me|what can|"
    r"what do|brief|short|intro|your|follower|social)\b",
    re.IGNORECASE,
)
_CONFUSED_RE = re.compile(
    r"\b(why|explain|confused|clarify|unclear|not sure|not working|"
    r"not understanding|makes no sense|don't understand|dont understand|"
    r"could you explain|can you explain)\b",
    re.IGNORECASE,
)
_EXCITED_RE = re.compile(
    r"\b(excited|can't wait|cant wait|looking forward|amazing|wonderful|"
    r"love it|awesome|fantastic|wow|yay|thrilled|great news|so happy)\b",
    re.IGNORECASE,
)


def detect_emotion(message: str) -> dict:
    # 1. Excitement
    if _EXCITED_RE.search(message):
        return {"label": "excited", "confidence": 0.82}

    # 2. Service override → skip confusion for business queries
    if _SVC_RE.search(message):
        c = _vader.polarity_scores(message)["compound"]
        if c <= VADER_ANGRY_THRESHOLD:
            return {"label": "angry",   "confidence": round(min(1.0, abs(c)), 2)}
        if c >= VADER_HAPPY_THRESHOLD:
            return {"label": "happy",   "confidence": round(min(1.0, c), 2)}
        return {"label": "neutral", "confidence": 0.60}

    # 3. Confusion keywords
    if _CONFUSED_RE.search(message):
        return {"label": "confused", "confidence": 0.85}

    # 4. VADER
    c = _vader.polarity_scores(message)["compound"]
    if c <= VADER_ANGRY_THRESHOLD:
        return {"label": "angry",   "confidence": round(min(1.0, abs(c)), 2)}
    if c <= VADER_SAD_UPPER:
        return {"label": "sad",     "confidence": round(min(1.0, abs(c) * 0.8), 2)}
    if c >= VADER_HAPPY_THRESHOLD:
        return {"label": "happy",   "confidence": round(min(1.0, c), 2)}
    return {"label": "neutral", "confidence": 0.60}