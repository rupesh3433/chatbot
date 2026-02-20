"""
emotion_model.py — DistilBERT-based multi-class emotion detection.

WHY THIS EXISTS:
  VADER is a lexicon-based tool designed for social media sentiment.
  It only produces positive/negative/neutral scores and misclassifies
  many makeup business queries.
  A fine-tuned DistilBERT model is far more accurate for nuanced
  emotions like confused, excited, angry, sad — and still runs on CPU
  within ~300 MB RAM.

MODEL USED:
  j-hartmann/emotion-english-distilroberta-base
  • Trained on 6 datasets (GoEmotions, ISEAR, etc.)
  • Labels: anger, disgust, fear, joy, neutral, sadness, surprise
  • Size: ~330 MB, loads in ~3 seconds on CPU
  • Inference: ~50ms per query on CPU

FALLBACK:
  If the model fails to load (RAM, network, etc.), falls back to
  VADER + rule-based detection automatically.

RAM COST: ~330 MB for the model weights
CPU COST: ~50ms per inference
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Emotion label mappings ─────────────────────────────────────────────────
# Maps DistilRoBERTa labels → our system labels
_DISTILROBERTA_MAP = {
    "anger":    "angry",
    "disgust":  "angry",     # treat disgust as angry for our use case
    "fear":     "sad",       # treat fear as sad
    "joy":      "happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "excited",   # treat surprise as excited
}

# ── Alternative model (smaller, if memory is very tight) ──────────────────
_PRIMARY_MODEL   = "j-hartmann/emotion-english-distilroberta-base"
_FALLBACK_MODEL  = "bhadresh-savani/distilbert-base-uncased-emotion"

# ── Singleton classifier ───────────────────────────────────────────────────
_classifier = None
_model_name  = None


def _load_model():
    """
    Lazily load the emotion classifier.
    Tries primary model first, then fallback, then returns None.
    """
    global _classifier, _model_name

    if _classifier is not None:
        return _classifier

    for model_id in (_PRIMARY_MODEL, _FALLBACK_MODEL):
        try:
            from transformers import pipeline as hf_pipeline
            print(f"[EMOTION_MODEL] Loading {model_id}...")
            _classifier = hf_pipeline(
                "text-classification",
                model     = model_id,
                top_k     = None,       # get all label scores
                device    = -1,         # CPU only
                truncation= True,
                max_length= 128,
            )
            _model_name = model_id
            print(f"[EMOTION_MODEL] Loaded: {model_id}")
            return _classifier
        except Exception as e:
            logger.warning(f"[EMOTION_MODEL] Failed to load {model_id}: {e}")
            continue

    logger.warning("[EMOTION_MODEL] All models failed to load. Using VADER fallback.")
    return None


# ══════════════════════════════════════════════════════════════════════════
# A) DistilBERT-based detection
# ══════════════════════════════════════════════════════════════════════════

def _predict_with_model(message: str) -> Optional[dict]:
    """
    Run the DistilRoBERTa classifier and return our standardised result.
    Returns None if model is unavailable.
    """
    clf = _load_model()
    if clf is None:
        return None

    try:
        results = clf(message[:512])[0]   # truncate to model max length
        # results is a list of {"label": str, "score": float}
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        top = sorted_results[0]

        raw_label  = top["label"].lower()
        mapped     = _DISTILROBERTA_MAP.get(raw_label, "neutral")
        confidence = round(float(top["score"]), 3)

        return {"label": mapped, "confidence": confidence}

    except Exception as e:
        logger.warning(f"[EMOTION_MODEL] Inference error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# B) VADER fallback (same logic as original emotion.py)
# ══════════════════════════════════════════════════════════════════════════

def _vader_fallback(message: str) -> dict:
    """
    VADER + rule-based emotion detection.
    Used when DistilBERT model is unavailable.
    """
    import re
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _analyser = SentimentIntensityAnalyzer()

    # Confusion keywords — tightly scoped
    CONFUSION_KEYWORDS = [
        "why", "explain", "confused",
        "i don't understand", "i dont understand",
        "clarify", "meaning", "means", "define",
        "unclear", "not sure", "not working", "not understanding",
        "makes no sense", "doesn't make sense",
    ]
    CONFUSION_PATTERN = re.compile(
        r'\b(' + '|'.join(re.escape(kw) for kw in CONFUSION_KEYWORDS) + r')\b',
        re.IGNORECASE,
    )
    SERVICE_OVERRIDE = re.compile(
        r'\b(service|services|book|booking|price|cost|offer|provide|'
        r'available|appointment|bridal|makeup|henna|party|chirag|'
        r'give me|tell me|what can|what do|about you|brief|short)\b',
        re.IGNORECASE,
    )
    EXCITEMENT_PATTERN = re.compile(
        r'\b(excited|cant wait|looking forward|amazing|wonderful|'
        r'love it|awesome|fantastic|wow|yay|so happy|thrilled)\b',
        re.IGNORECASE,
    )

    if EXCITEMENT_PATTERN.search(message):
        return {"label": "excited", "confidence": 0.80}

    if SERVICE_OVERRIDE.search(message):
        scores = _analyser.polarity_scores(message)
        c = scores["compound"]
        if c <= -0.55: return {"label": "angry",   "confidence": round(min(1.0, abs(c)), 2)}
        if c >= 0.35:  return {"label": "happy",   "confidence": round(min(1.0, c), 2)}
        return {"label": "neutral", "confidence": 0.60}

    if CONFUSION_PATTERN.search(message):
        return {"label": "confused", "confidence": 0.85}

    scores = _analyser.polarity_scores(message)
    c = scores["compound"]
    if c <= -0.55: return {"label": "angry",   "confidence": round(min(1.0, abs(c)), 2)}
    if c <= -0.25: return {"label": "sad",     "confidence": round(min(1.0, abs(c) * 0.8), 2)}
    if c >= 0.35:  return {"label": "happy",   "confidence": round(min(1.0, c), 2)}
    return {"label": "neutral", "confidence": 0.60}


# ══════════════════════════════════════════════════════════════════════════
# C) Main public function
# ══════════════════════════════════════════════════════════════════════════

def detect_emotion_ml(message: str) -> dict:
    """
    Detect emotion using DistilBERT model with VADER fallback.

    Returns:
        {"label": str, "confidence": float}
        label ∈ {angry, sad, confused, excited, happy, neutral}
    """
    # Try ML model first
    result = _predict_with_model(message)
    if result is not None:
        # Override: rule-based confusion/excitement detection is still more
        # reliable for domain-specific messages ("why is it not working" etc.)
        import re
        CONFUSION_PATTERN = re.compile(
            r'\b(why|explain|confused|clarify|unclear|not sure|'
            r'not working|not understanding|makes no sense)\b',
            re.IGNORECASE,
        )
        SERVICE_OVERRIDE = re.compile(
            r'\b(service|services|book|booking|price|cost|offer|'
            r'available|appointment|bridal|makeup|henna|chirag|'
            r'give me|tell me|about you|brief)\b',
            re.IGNORECASE,
        )
        EXCITEMENT_PATTERN = re.compile(
            r'\b(excited|cant wait|amazing|wow|yay|thrilled|looking forward)\b',
            re.IGNORECASE,
        )

        # High-confidence rule overrides (more reliable for domain inputs)
        if EXCITEMENT_PATTERN.search(message):
            return {"label": "excited", "confidence": 0.85}
        if CONFUSION_PATTERN.search(message) and not SERVICE_OVERRIDE.search(message):
            return {"label": "confused", "confidence": 0.85}

        return result

    # Fallback to VADER
    logger.debug("[EMOTION_MODEL] Using VADER fallback")
    return _vader_fallback(message)


# ══════════════════════════════════════════════════════════════════════════
# D) Check if model is available (for health checks)
# ══════════════════════════════════════════════════════════════════════════

def is_model_available() -> bool:
    """Return True if the DistilBERT emotion model loaded successfully."""
    return _load_model() is not None


# ══════════════════════════════════════════════════════════════════════════
# E) CLI test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Emotion Detection Test (Ctrl+C to quit)\n")
    test_sentences = [
        "I'm so excited to book my bridal makeup!",
        "Why is it not understanding my question?",
        "What is your phone number?",
        "I am very angry about the service.",
        "Give me 1 line about you.",
        "What are your services?",
        "I don't understand the booking process.",
        "How much does bridal makeup cost?",
        "Thank you so much!",
        "This is terrible, nothing is working.",
    ]
    for sentence in test_sentences:
        result = detect_emotion_ml(sentence)
        print(f"  '{sentence[:50]}'")
        print(f"    → {result['label']} ({result['confidence']:.2f})\n")

    print("\nInteractive mode (Ctrl+C to quit):")
    while True:
        try:
            msg = input("You: ").strip()
            if msg:
                r = detect_emotion_ml(msg)
                print(f"  → {r['label']} ({r['confidence']:.2f})")
        except KeyboardInterrupt:
            break