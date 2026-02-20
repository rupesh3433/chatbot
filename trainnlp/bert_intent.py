"""
bert_intent.py — Fine‑tuned DistilBERT for intent classification.

WHY:
  The previous TF‑IDF+SVM model was limited in handling unseen phrasings.
  DistilBERT captures semantic meaning and generalises far better.

TRAINING:
  Uses the same labeled data as `nlu.py` but fine‑tunes a transformer.
  Training is slower (requires GPU ideally), but inference is fast enough on CPU.

USAGE:
  from trainnlp.bert_intent import predict_intent, retrain
  result = predict_intent("how much for bridal makeup")
  # returns {"intent": "pricing", "confidence": 0.97, "trusted": True}
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

BERT_MODEL_PATH = MODEL_DIR / "bert_intent"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# Same training data as in nlu.py (copied here for independence)
TRAINING_DATA: List[Tuple[str, str]] = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey there", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),
    ("good afternoon", "greeting"),
    ("hiya", "greeting"),
    ("howdy", "greeting"),
    ("yo", "greeting"),
    ("sup", "greeting"),
    ("hey!", "greeting"),
    ("hi there", "greeting"),
    ("hello!", "greeting"),
    ("greetings", "greeting"),
    ("heya", "greeting"),
    ("how are you", "small_talk"),
    ("how r u", "small_talk"),
    ("how are u doing", "small_talk"),
    ("are you a bot", "small_talk"),
    ("are you an ai", "small_talk"),
    ("are you real", "small_talk"),
    ("thank you", "small_talk"),
    ("thanks", "small_talk"),
    ("thank u", "small_talk"),
    ("thx", "small_talk"),
    ("ok thanks", "small_talk"),
    ("great thanks", "small_talk"),
    ("bye", "small_talk"),
    ("goodbye", "small_talk"),
    ("see you later", "small_talk"),
    ("take care", "small_talk"),
    ("nice to meet you", "small_talk"),
    ("you're welcome", "small_talk"),
    ("no problem", "small_talk"),
    ("that's great", "small_talk"),
    ("okay cool", "small_talk"),
    ("awesome", "small_talk"),
    ("noted", "small_talk"),
    ("got it", "small_talk"),
    ("i see", "small_talk"),
    ("what can you do", "services"),
    ("what do you offer", "services"),
    ("what services do you provide", "services"),
    ("what are your services", "services"),
    ("tell me about your services", "services"),
    ("what is available", "services"),
    ("list your services", "services"),
    ("what kind of makeup do you do", "services"),
    ("what do you specialize in", "services"),
    ("which services are available", "services"),
    ("show me your services", "services"),
    ("can you help me with makeup", "services"),
    ("what types of makeup do you offer", "services"),
    ("do you do bridal makeup", "services"),
    ("do you offer party makeup", "services"),
    ("what makeup services do you have", "services"),
    ("tell me about the makeup packages", "services"),
    ("i want to know about your services", "services"),
    ("what can you help me with", "services"),
    ("what all do you do", "services"),
    ("wut services u got", "services"),
    ("yo what do u offer", "services"),
    ("services plz", "services"),
    ("services list", "services"),
    ("gimme ur services", "services"),
    ("how much does it cost", "pricing"),
    ("what is the price", "pricing"),
    ("what are your charges", "pricing"),
    ("pricing details please", "pricing"),
    ("how much for bridal makeup", "pricing"),
    ("what is the cost of party makeup", "pricing"),
    ("can you tell me the rates", "pricing"),
    ("what are the fees", "pricing"),
    ("how much do you charge", "pricing"),
    ("give me the price list", "pricing"),
    ("what is your pricing", "pricing"),
    ("is it expensive", "pricing"),
    ("tell me about packages and prices", "pricing"),
    ("how much is henna", "pricing"),
    ("what does bridal cost", "pricing"),
    ("cost of makeup", "pricing"),
    ("price of bridal", "pricing"),
    ("how much for party", "pricing"),
    ("hw much", "pricing"),
    ("price?", "pricing"),
    ("cost?", "pricing"),
    ("rates?", "pricing"),
    ("how do i book", "booking"),
    ("how to book an appointment", "booking"),
    ("i want to book", "booking"),
    ("can i book makeup", "booking"),
    ("how can i schedule an appointment", "booking"),
    ("book bridal makeup", "booking"),
    ("are you available", "booking"),
    ("are you free today", "booking"),
    ("do you have any slots available", "booking"),
    ("how to reserve a slot", "booking"),
    ("what is the booking process", "booking"),
    ("explain the booking steps", "booking"),
    ("can i make an appointment", "booking"),
    ("how do i confirm my booking", "booking"),
    ("what information is needed to book", "booking"),
    ("i want to schedule", "booking"),
    ("is there availability this weekend", "booking"),
    ("book appointment for wedding", "booking"),
    ("how do i get otp", "booking"),
    ("how to book online", "booking"),
    ("wanna book", "booking"),
    ("book pls", "booking"),
    ("how 2 book", "booking"),
    ("slot available?", "booking"),
    ("free date?", "booking"),
    ("what is your phone number", "contact"),
    ("how can i contact you", "contact"),
    ("what is your email", "contact"),
    ("give me your contact details", "contact"),
    ("how do i reach you", "contact"),
    ("can i call you", "contact"),
    ("whatsapp number please", "contact"),
    ("how to get in touch", "contact"),
    ("contact information", "contact"),
    ("what is your address", "contact"),
    ("how do i message you", "contact"),
    ("give me your number", "contact"),
    ("your email id", "contact"),
    ("contact details please", "contact"),
    ("phone number?", "contact"),
    ("ur number?", "contact"),
    ("contact?", "contact"),
    ("how 2 contact", "contact"),
    ("who are you", "about"),
    ("tell me about yourself", "about"),
    ("what is jinnichirag", "about"),
    ("who is chirag sharma", "about"),
    ("introduce yourself", "about"),
    ("tell me about jinni chirag", "about"),
    ("what is this brand about", "about"),
    ("background of chirag sharma", "about"),
    ("how many years of experience", "about"),
    ("give me 1 line about you", "about"),
    ("brief intro about you", "about"),
    ("tell me about the artist", "about"),
    ("who is the makeup artist", "about"),
    ("about jinnichirag makeup artist", "about"),
    ("what can you tell me about yourself", "about"),
    ("describe yourself", "about"),
    ("whos chirag", "about"),
    ("abt u", "about"),
    ("tell me abt urself", "about"),
    ("how many followers do you have", "social"),
    ("what is your instagram", "social"),
    ("your youtube channel", "social"),
    ("social media links", "social"),
    ("how many subscribers", "social"),
    ("tiktok account", "social"),
    ("facebook page", "social"),
    ("what is your social media", "social"),
    ("how many people follow you", "social"),
    ("total following", "social"),
    ("combined followers", "social"),
    ("instagram followers count", "social"),
    ("social media presence", "social"),
    ("follow on instagram", "social"),
    ("ur insta?", "social"),
    ("ig link", "social"),
    ("yt channel link", "social"),
    ("give me makeup tips", "tips"),
    ("any advice on bridal makeup", "tips"),
    ("how to apply makeup", "tips"),
    ("tips for long lasting makeup", "tips"),
    ("skincare before makeup", "tips"),
    ("how to prepare for wedding makeup", "tips"),
    ("professional makeup advice", "tips"),
    ("makeup tips for beginners", "tips"),
    ("what should i do before makeup session", "tips"),
    ("how to make makeup last", "tips"),
    ("tips plz", "tips"),
    ("advice for makeup", "tips"),
    ("are there any upcoming events", "events"),
    ("do you conduct workshops", "events"),
    ("makeup class available", "events"),
    ("any seminars", "events"),
    ("how to attend your event", "events"),
    ("event details", "events"),
    ("workshop schedule", "events"),
    ("what are the frequently asked questions", "faq"),
    ("show me the faq", "faq"),
    ("common questions", "faq"),
    ("what do people usually ask", "faq"),
    ("give me 1 line about your services", "short"),
    ("brief answer please", "short"),
    ("in short", "short"),
    ("summarize this", "short"),
    ("tldr", "short"),
    ("one line answer", "short"),
    ("quick summary", "short"),
    ("briefly explain", "short"),
    ("2 lines about booking", "short"),
    ("short description", "short"),
    ("just give me a brief", "short"),
    ("quick answer", "short"),
    ("explain step by step how to book", "steps"),
    ("walk me through the process", "steps"),
    ("step by step booking guide", "steps"),
    ("how do i do it step by step", "steps"),
    ("show me the steps", "steps"),
    ("procedure for booking", "steps"),
    ("guide me through booking", "steps"),
    ("detailed steps please", "steps"),
    ("give me detailed information", "detailed"),
    ("explain everything in detail", "detailed"),
    ("complete details about services", "detailed"),
    ("full explanation please", "detailed"),
    ("elaborate on bridal makeup", "detailed"),
    ("tell me everything about booking", "detailed"),
    ("complete answer", "detailed"),
    ("explain fully", "detailed"),
    ("what was my first question", "memory"),
    ("what did i ask first", "memory"),
    ("my previous question", "memory"),
    ("what is our first chat", "memory"),
    ("remind me what i asked", "memory"),
    ("what have i asked so far", "memory"),
    ("what was my last question", "memory"),
    ("show my chat history", "memory"),
    ("what did i say before", "memory"),
    ("previous messages", "memory"),
    ("what is out first chats", "memory"),
    ("my first question was what", "memory"),
    ("can you help me", "general"),
    ("i have a question", "general"),
    ("need some information", "general"),
    ("just wondering", "general"),
    ("random question", "general"),
    ("not sure what to ask", "general"),
]

# All possible intent labels (for LabelEncoder)
INTENT_LABELS = sorted({
    "greeting", "small_talk", "services", "pricing", "booking",
    "contact", "about", "social", "tips", "events", "faq",
    "short", "steps", "detailed", "memory", "general",
})

# Minimum confidence to trust the classifier
MIN_CONFIDENCE = 0.45

# Global variables for lazy loading
_tokenizer = None
_model = None
_label_encoder = None


def _load_or_train():
    """Load the model and tokenizer from disk, or train if not present."""
    global _tokenizer, _model, _label_encoder

    if _model is not None:
        return

    if BERT_MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
        print("[BERT_INTENT] Loading model from disk...")
        _tokenizer = DistilBertTokenizer.from_pretrained(str(BERT_MODEL_PATH))
        _model = DistilBertForSequenceClassification.from_pretrained(str(BERT_MODEL_PATH))
        with open(LABEL_ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)
        _model.eval()
        print("[BERT_INTENT] Model loaded.")
    else:
        print("[BERT_INTENT] No saved model found. Training now...")
        train()
        _load_or_train()  # recursive call after training


def train(extra_data: Optional[List[Tuple[str, str]]] = None):
    """
    Fine-tune DistilBERT on the training data + extra corrections.
    Saves model and label encoder to disk.
    """
    global _tokenizer, _model, _label_encoder

    data = list(TRAINING_DATA)
    if extra_data:
        data.extend(extra_data)

    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    # Split for validation (optional)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, y, test_size=0.1, random_state=42, stratify=y
    )

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )

    # Use eval_strategy (compatible with older transformers) instead of evaluation_strategy
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "training"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(MODEL_DIR / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        fp16=False,  # CPU friendly
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},
    )

    trainer.train()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    # Save model and tokenizer
    model.save_pretrained(BERT_MODEL_PATH)
    tokenizer.save_pretrained(BERT_MODEL_PATH)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"[BERT_INTENT] Training complete. Model saved to {BERT_MODEL_PATH}")

    # Update globals
    _tokenizer = tokenizer
    _model = model
    _label_encoder = label_encoder
    _model.eval()


def predict_intent(message: str) -> dict:
    """
    Predict intent using the fine-tuned BERT model.

    Returns:
        {
            "intent":     str,
            "confidence": float,
            "trusted":    bool   (True if confidence >= MIN_CONFIDENCE)
        }
    """
    _load_or_train()

    inputs = _tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])
    intent = _label_encoder.inverse_transform([pred_id])[0]

    return {
        "intent": intent,
        "confidence": round(confidence, 3),
        "trusted": confidence >= MIN_CONFIDENCE,
    }


def nlu_intent_to_mode(nlu_result: dict) -> Optional[str]:
    """
    Convert BERT intent to system intent_mode.
    Same mapping as in old nlu.py.
    """
    if not nlu_result["trusted"]:
        return None

    label = nlu_result["intent"]

    # Direct mapping
    direct_map = {
        "greeting":  "greeting",
        "small_talk":"small_talk",
        "short":     "short",
        "steps":     "steps",
        "detailed":  "detailed",
        "memory":    "memory",
        "general":   "default",
    }
    if label in direct_map:
        return direct_map[label]

    # KB content intents -> "default" (they go to KB)
    kb_intents = {
        "services", "pricing", "booking", "contact",
        "about", "social", "tips", "events", "faq",
    }
    if label in kb_intents:
        return "default"

    return "default"


def nlu_content_hint(nlu_result: dict) -> Optional[str]:
    """
    If the NLU classified a KB content-type intent with high confidence,
    return it as a hint to summarizer.detect_content_type().
    """
    if not nlu_result["trusted"]:
        return None
    label = nlu_result["intent"]
    kb_intents = {
        "services", "pricing", "booking", "contact",
        "about", "social", "tips", "events", "faq",
    }
    return label if label in kb_intents else None


def retrain(extra_data: List[Tuple[str, str]] = None):
    """Public API: retrain the model and refresh cache."""
    global _model, _tokenizer, _label_encoder
    _model = _tokenizer = _label_encoder = None
    train(extra_data)
    _load_or_train()


# For command-line training
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        # Interactive test
        _load_or_train()
        print("BERT Intent Classifier ready. Type a message (Ctrl+C to quit)")
        while True:
            msg = input("You: ").strip()
            if msg:
                res = predict_intent(msg)
                print(f"  Intent: {res['intent']} (conf={res['confidence']:.3f}, trusted={res['trusted']})")