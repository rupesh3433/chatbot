"""
nlu.py — Trainable TF-IDF + LinearSVC intent classifier.

Generalises to unseen phrasings, slang, and typos via char+word n-grams.
Trains in <5 seconds on CPU. Uses ~30 MB RAM.
Saved to models/nlu_model.pkl — reloaded on restart.

INTENT LABELS:
  greeting, small_talk, services, pricing, booking, contact,
  about, social, tips, events, faq, short, steps, detailed, memory, general
"""
import os, re, pickle, logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/nlu_model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)
MIN_CONFIDENCE = 0.42

# ── Training data ──────────────────────────────────────────────────────────
TRAIN: List[Tuple[str, str]] = [
    # greeting
    ("hi", "greeting"), ("hello", "greeting"), ("hey there", "greeting"),
    ("good morning", "greeting"), ("good evening", "greeting"),
    ("howdy", "greeting"), ("hiya", "greeting"), ("yo", "greeting"),
    ("sup", "greeting"), ("hey!", "greeting"), ("heya", "greeting"),
    ("greetings", "greeting"), ("good afternoon", "greeting"),

    # small_talk
    ("how are you", "small_talk"), ("how r u", "small_talk"),
    ("are you a bot", "small_talk"), ("are you real", "small_talk"),
    ("thank you", "small_talk"), ("thanks", "small_talk"), ("thx", "small_talk"),
    ("bye", "small_talk"), ("goodbye", "small_talk"), ("take care", "small_talk"),
    ("nice to meet you", "small_talk"), ("okay cool", "small_talk"),
    ("got it", "small_talk"), ("noted", "small_talk"), ("awesome", "small_talk"),
    ("ok thanks", "small_talk"), ("great thanks", "small_talk"),

    # services
    ("what can you do", "services"), ("what do you offer", "services"),
    ("what services do you provide", "services"), ("what are your services", "services"),
    ("tell me about your services", "services"), ("list your services", "services"),
    ("what kind of makeup do you do", "services"), ("what do you specialize in", "services"),
    ("show me your services", "services"), ("can you help me with makeup", "services"),
    ("what types of makeup do you offer", "services"), ("do you do bridal makeup", "services"),
    ("do you offer party makeup", "services"), ("what all do you do", "services"),
    ("wut services u got", "services"), ("services plz", "services"),
    ("services list", "services"), ("gimme ur services", "services"),
    ("what can you help with", "services"), ("what makeup do you do", "services"),

    # pricing
    ("how much does it cost", "pricing"), ("what is the price", "pricing"),
    ("what are your charges", "pricing"), ("pricing details please", "pricing"),
    ("how much for bridal makeup", "pricing"), ("what is the cost of party makeup", "pricing"),
    ("can you tell me the rates", "pricing"), ("what are the fees", "pricing"),
    ("how much do you charge", "pricing"), ("give me the price list", "pricing"),
    ("what is your pricing", "pricing"), ("how much for henna", "pricing"),
    ("what does bridal cost", "pricing"), ("cost of makeup", "pricing"),
    ("hw much", "pricing"), ("price?", "pricing"), ("cost?", "pricing"),
    ("rates?", "pricing"), ("tell me pricing", "pricing"), ("package prices", "pricing"),

    # booking
    ("how do i book", "booking"), ("how to book an appointment", "booking"),
    ("i want to book", "booking"), ("can i book makeup", "booking"),
    ("how can i schedule", "booking"), ("book bridal makeup", "booking"),
    ("are you available", "booking"), ("are you free today", "booking"),
    ("do you have any slots", "booking"), ("how to reserve a slot", "booking"),
    ("what is the booking process", "booking"), ("explain booking steps", "booking"),
    ("can i make an appointment", "booking"), ("how do i confirm my booking", "booking"),
    ("how do i get otp", "booking"), ("wanna book", "booking"),
    ("book pls", "booking"), ("how 2 book", "booking"), ("slot available", "booking"),
    ("free slot", "booking"), ("book party makeup", "booking"),
    ("i want to schedule", "booking"), ("available for booking", "booking"),

    # contact
    ("what is your phone number", "contact"), ("how can i contact you", "contact"),
    ("what is your email", "contact"), ("give me your contact details", "contact"),
    ("how do i reach you", "contact"), ("whatsapp number please", "contact"),
    ("how to get in touch", "contact"), ("contact information", "contact"),
    ("how do i message you", "contact"), ("give me your number", "contact"),
    ("phone number", "contact"), ("ur number", "contact"), ("contact?", "contact"),
    ("your email id", "contact"), ("how 2 contact", "contact"),

    # about
    ("who are you", "about"), ("tell me about yourself", "about"),
    ("what is jinnichirag", "about"), ("who is chirag sharma", "about"),
    ("introduce yourself", "about"), ("tell me about jinni chirag", "about"),
    ("what is this brand about", "about"), ("background of chirag sharma", "about"),
    ("give me 1 line about you", "about"), ("brief intro about you", "about"),
    ("who is the makeup artist", "about"), ("about jinnichirag", "about"),
    ("whos chirag", "about"), ("abt u", "about"), ("describe yourself", "about"),
    ("tell me abt urself", "about"), ("how many years experience", "about"),

    # social
    ("how many followers do you have", "social"), ("what is your instagram", "social"),
    ("your youtube channel", "social"), ("social media links", "social"),
    ("how many subscribers", "social"), ("tiktok account", "social"),
    ("how many people follow you", "social"), ("combined followers", "social"),
    ("instagram followers count", "social"), ("ur insta", "social"),
    ("ig link", "social"), ("yt channel link", "social"), ("social media presence", "social"),
    ("how big is your following", "social"),

    # tips
    ("give me makeup tips", "tips"), ("any advice on bridal makeup", "tips"),
    ("how to apply makeup", "tips"), ("tips for long lasting makeup", "tips"),
    ("skincare before makeup", "tips"), ("professional makeup advice", "tips"),
    ("how to prepare for wedding makeup", "tips"), ("tips plz", "tips"),
    ("advice for makeup", "tips"), ("what should i do before makeup session", "tips"),

    # events
    ("are there upcoming events", "events"), ("do you conduct workshops", "events"),
    ("makeup class available", "events"), ("any seminars", "events"),
    ("how to attend your event", "events"), ("workshop schedule", "events"),

    # faq
    ("show me the faq", "faq"), ("common questions", "faq"),
    ("what are frequently asked questions", "faq"), ("what do people usually ask", "faq"),

    # short
    ("give me 1 line about your services", "short"), ("brief answer please", "short"),
    ("in short", "short"), ("summarize this", "short"), ("tldr", "short"),
    ("one line answer", "short"), ("quick summary", "short"), ("briefly explain", "short"),
    ("2 lines about booking", "short"), ("just give me a brief", "short"),
    ("short description", "short"), ("quick answer", "short"), ("in 2 lines", "short"),

    # steps
    ("explain step by step how to book", "steps"), ("walk me through the process", "steps"),
    ("step by step booking guide", "steps"), ("show me the steps", "steps"),
    ("procedure for booking", "steps"), ("guide me through booking", "steps"),
    ("how to do it step by step", "steps"), ("detailed steps please", "steps"),

    # detailed
    ("give me detailed information", "detailed"), ("explain everything in detail", "detailed"),
    ("complete details about services", "detailed"), ("full explanation please", "detailed"),
    ("elaborate on bridal makeup", "detailed"), ("tell me everything about booking", "detailed"),
    ("complete answer", "detailed"), ("explain fully", "detailed"),

    # memory
    ("what was my first question", "memory"), ("what did i ask first", "memory"),
    ("my previous question", "memory"), ("what is our first chat", "memory"),
    ("remind me what i asked", "memory"), ("what have i asked so far", "memory"),
    ("what was my last question", "memory"), ("show my chat history", "memory"),
    ("what did i say before", "memory"), ("previous messages", "memory"),
    ("what is out first chats", "memory"), ("first question", "memory"),

    # general
    ("can you help me", "general"), ("i have a question", "general"),
    ("need some information", "general"), ("just wondering", "general"),
]

# ── Map NLU label → intent_mode used by the pipeline ──────────────────────
_TO_MODE = {
    "greeting":  "greeting",   "small_talk": "small_talk",
    "short":     "short",      "steps":      "steps",
    "detailed":  "detailed",   "memory":     "memory",
    "services":  "default",    "pricing":    "default",
    "booking":   "default",    "contact":    "default",
    "about":     "default",    "social":     "default",
    "tips":      "default",    "events":     "default",
    "faq":       "default",    "general":    "default",
}
# KB content-type labels (hint passed to summarizer)
KB_LABELS = {"services","pricing","booking","contact","about","social","tips","events","faq"}


def _preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text)


def _build_pipeline():
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    char_vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
                                 max_features=35000, sublinear_tf=True,
                                 strip_accents="unicode", min_df=1)
    word_vect = TfidfVectorizer(analyzer="word", ngram_range=(1,3),
                                 max_features=25000, sublinear_tf=True,
                                 strip_accents="unicode", min_df=1)
    combined = FeatureUnion([("char", char_vect), ("word", word_vect)])
    clf = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"), cv=3
    )
    return Pipeline([("features", combined), ("clf", clf)])


def train(extra: List[Tuple[str, str]] = None):
    from sklearn.model_selection import cross_val_score
    data   = list(TRAIN) + (extra or [])
    texts  = [_preprocess(t) for t, _ in data]
    labels = [l for _, l in data]
    pipe   = _build_pipeline()
    pipe.fit(texts, labels)
    cv = cross_val_score(pipe, texts, labels, cv=3, scoring="accuracy")
    print(f"[NLU] Trained {len(texts)} examples | CV acc {cv.mean():.2f}±{cv.std():.2f}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"[NLU] Saved → {MODEL_PATH}")
    return pipe


_pipe = None

def _load():
    global _pipe
    if _pipe is not None:
        return _pipe
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _pipe = pickle.load(f)
        print(f"[NLU] Loaded from {MODEL_PATH}")
    else:
        print("[NLU] No model found — training now...")
        _pipe = train()
    return _pipe


def predict(message: str) -> dict:
    pipe = _load()
    proc = _preprocess(message)
    proba   = pipe.predict_proba([proc])[0]
    classes = pipe.classes_
    idx     = int(np.argmax(proba))
    return {
        "label":      classes[idx],
        "confidence": round(float(proba[idx]), 3),
        "trusted":    float(proba[idx]) >= MIN_CONFIDENCE,
    }


def to_intent_mode(result: dict) -> Optional[str]:
    if not result["trusted"]:
        return None
    return _TO_MODE.get(result["label"], "default")


def content_hint(result: dict) -> Optional[str]:
    if not result["trusted"]:
        return None
    return result["label"] if result["label"] in KB_LABELS else None


def retrain(extra: List[Tuple[str, str]] = None):
    global _pipe
    _pipe = train(extra)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        _load()
        print("NLU test — type messages (Ctrl+C to quit):")
        while True:
            try:
                msg = input("→ ").strip()
                if msg:
                    r = predict(msg)
                    print(f"  label={r['label']}  conf={r['confidence']}  trusted={r['trusted']}")
            except KeyboardInterrupt:
                break