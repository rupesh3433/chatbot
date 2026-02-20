"""
summarizer.py — Content-type-aware extractive summarisation.

v3.0 — Accepts optional NLU content_hint to improve content-type detection.

UPGRADE:
  extractive_summarise() now accepts an optional content_hint parameter.
  When the NLU classifier predicts a KB content type with high confidence
  (e.g. "pricing", "services"), that hint is used to override or confirm
  the heuristic detect_content_type() logic. This reduces misclassification.
"""

import re
from typing import List, Dict, Any, Optional

from agent.config import MAX_CONTEXT_CHARS, MAX_SUMMARY_SENTENCES
from agent.utils import extract_keywords, safe_truncate


# ══════════════════════════════════════════════════════════════════════════
# A) Content-type detection
# ══════════════════════════════════════════════════════════════════════════

def detect_content_type(
    chunks: List[dict],
    query: str,
    hint: Optional[str] = None,
) -> str:
    """
    Determine content type from query + chunks.
    If hint is provided (from NLU classifier), it takes priority
    when the heuristic result agrees or is 'general'.
    """
    combined = " ".join(c.get("chunk_text", "") for c in chunks).lower()
    q = query.lower()

    def asks(*words): return any(w in q for w in words)
    def has(*words):  return any(w in combined for w in words)

    # Heuristic detection
    heuristic = _heuristic_content_type(q, combined, asks, has)

    # Apply hint: if NLU gave a specific KB label and heuristic is vague
    valid_hints = {
        "services", "pricing", "booking", "contact",
        "about", "social", "tips", "events", "faq",
    }
    if hint and hint in valid_hints:
        if heuristic == "general" or heuristic == hint:
            return hint

    return heuristic


def _heuristic_content_type(q, combined, asks, has) -> str:
    """Pure heuristic content-type detection (unchanged from v2.3)."""

    # Social
    if asks("follower", "followers", "following", "how many people", "how many follow",
            "social media count", "subscriber", "reach", "fan", "fans", "audience"):
        return "social"
    if has("1.4 million", "combined following", "million followers") and \
       asks("follow", "social", "instagram", "youtube"):
        return "social"

    # Pricing
    if asks("price", "cost", "charge", "fee", "how much", "rate", "pricing",
            "package", "₹", "inr") and \
       has("₹", "inr", "rs.", "price", "cost", "charge"):
        return "pricing"

    # Booking
    if asks("book", "appointment", "schedule", "reserve", "how to book",
            "slot", "availability", "otp", "confirmation",
            "are you free", "are you available", "free today", "available today",
            "when can i", "can i book", "free slot", "open slot"):
        return "booking"

    # Contact
    if asks("contact", "phone", "email", "reach", "call", "whatsapp",
            "message", "talk", "number", "address"):
        return "contact"

    # Services — BEFORE faq
    if asks("service", "services", "offer", "provide", "do you do",
            "what can", "available", "list", "what do you",
            "tell me about your", "what services",
            "what can you do", "what do you have", "capabilities",
            "what do you offer", "what can you help",
            "services offered", "services available"):
        return "services"

    # About
    if asks("about", "who are you", "tell me about yourself", "background",
            "experience", "profile", "introduce", "biography", "history",
            "what can you tell", "complete details", "yourself", "chirag",
            "about you", "1 line about", "one line about", "brief about",
            "tell me about jinni", "tell me about chirag"):
        return "about"

    # Tips
    if asks("tip", "tips", "advice", "how to apply", "skin", "skincare",
            "prepare", "preparation", "makeup advice"):
        return "tips"

    # Events
    if asks("event", "events", "workshop", "class", "seminar", "attend"):
        return "events"

    # FAQ — only when explicitly requested
    if asks("faq", "frequently", "common question") and \
       has("q:", "a:", "frequently asked"):
        return "faq"

    # Content-based fallback
    if has("₹"):                                          return "pricing"
    if has("1.4 million", "million followers"):            return "social"
    if has("instagram", "youtube", "tiktok") and \
       has("+977", "email", "jinnie.chirag"):              return "contact"
    if has("bridal", "party makeup", "henna"):             return "services"
    if has("otp", "booking confirmed", "advance"):         return "booking"

    return "general"


# ══════════════════════════════════════════════════════════════════════════
# B) Structured extractors (unchanged from v2.3)
# ══════════════════════════════════════════════════════════════════════════

def _extract_pricing(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    categories, current_cat, current_items = [], None, []
    for line in lines:
        has_price = bool(re.search(r'₹[\d,]+', line))
        is_header = (
            line.endswith(":") or
            re.match(r'^(bridal|party|henna|mehendi|engagement|pre.wedding|'
                     r'reception|luxury|signature|makeup services)', line, re.I)
        ) and not has_price
        is_item = bool(re.match(r'^[-•*]\s|^\d+[.)]\s', line)) or has_price
        if is_header:
            if current_cat and current_items:
                categories.append({"name": current_cat, "items": current_items})
            current_cat, current_items = line.rstrip(":"), []
        elif is_item and current_cat:
            clean = re.sub(r'^[-•*\d.)]+\s*', '', line).strip()
            if clean: current_items.append(clean)
    if current_cat and current_items:
        categories.append({"name": current_cat, "items": current_items})
    if not categories:
        price_lines = [l.strip() for l in text.splitlines() if re.search(r'₹[\d,]+', l)]
        if price_lines:
            categories.append({"name": "Price List", "items": price_lines})
    notes = [l.strip().lstrip("-• ") for l in text.splitlines()
             if l.strip() and any(k in l.lower() for k in
                                   ["exclud", "vari", "final", "advance", "confirm"])]
    return {"categories": categories, "notes": notes[:4]}


def _extract_services(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    services, seen = [], set()
    svc_keywords   = r'bridal|party|henna|mehendi|engagement|pre.wedding|reception|signature|luxury'
    skip_patterns  = re.compile(
        r'^(q:|a:|yes,|no,|there are|the booking|clients|for immediate|'
        r'available by|service is|services are|all services)',
        re.IGNORECASE
    )
    for line in lines:
        if skip_patterns.match(line) or len(line) > 100:
            continue
        if re.search(svc_keywords, line, re.I) and len(line) > 8:
            clean = re.sub(r'^[-•*\d.)]+\s*', '', line).strip().rstrip(":")
            clean = re.sub(r'\s*[\(₹].*$', '', clean).strip()
            key   = clean.lower()[:40]
            if clean and key not in seen and len(clean) > 5:
                services.append(clean)
                seen.add(key)
    for line in lines:
        if re.match(r'^\d+\.\s', line) and not skip_patterns.match(line):
            clean = re.sub(r'^\d+\.\s*', '', line).strip()
            clean = re.sub(r'\s*[\(₹].*$', '', clean).strip()
            key   = clean.lower()[:40]
            if key not in seen and 5 < len(clean) < 80:
                services.insert(0, clean)
                seen.add(key)
    return {"services": list(dict.fromkeys(services))[:10]}


def _extract_booking(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    steps, required, seen_steps = [], [], set()
    in_required = False
    for line in lines:
        lower = line.lower()
        if any(w in lower for w in ["required information", "information required",
                                     "information you'll need", "details needed"]):
            in_required = True
            continue
        if re.match(r'^[-•*]\s|^\d+[.)]\s', line):
            clean = re.sub(r'^[-•*\d.)]+\s*', '', line).strip()
            key   = clean.lower()[:60]
            if in_required:
                if key not in seen_steps: required.append(clean); seen_steps.add(key)
            else:
                if key not in seen_steps and len(clean) > 10:
                    steps.append(clean); seen_steps.add(key)
        elif any(w in lower for w in ["method 1", "method 2", "step",
                                       "first", "next", "then", "finally", "after"]):
            key = lower[:60]
            if key not in seen_steps: steps.append(line); seen_steps.add(key)
    return {"steps": [s for s in steps if len(s) > 15][:7], "required": required[:7]}


def _extract_contact(text: str) -> Dict[str, Any]:
    info  = {}
    for line in text.splitlines():
        line  = line.strip(); lower = line.lower()
        if not line: continue
        if re.search(r'\+\d[\d\s\-]{8,}', line):
            m = re.search(r'\+[\d\s\-]+', line)
            if m: info.setdefault("phone", m.group().strip())
        if re.search(r'[\w.\-]+@[\w.\-]+\.\w+', line):
            m = re.search(r'[\w.\-]+@[\w.\-]+\.\w+', line)
            if m: info.setdefault("email", m.group())
        for platform in ["instagram", "youtube", "facebook", "tiktok"]:
            if platform in lower:
                m = re.search(r'https?://\S+', line)
                if m: info.setdefault(platform, m.group().rstrip(".,)"))
        if "whatsapp" in lower and "whatsapp" not in info:
            m = re.search(r'\+[\d\s\-]+', line)
            if m: info["whatsapp"] = m.group().strip()
    return {"contact": info}


def _extract_social(text: str) -> Dict[str, Any]:
    info = {}
    m = re.search(r'([\d.]+\s*million\+?|over\s+[\d.]+\s*million|[\d,]+\+?\s*followers)',
                  text, re.I)
    if m: info["total_followers"] = m.group().strip()
    for line in text.splitlines():
        lower = line.lower()
        for platform in ["instagram", "youtube", "facebook", "tiktok"]:
            if platform in lower:
                url_m   = re.search(r'https?://\S+', line)
                count_m = re.search(r'[\d.,]+\s*(k|m|million|thousand|lakh|followers)?', line, re.I)
                entry   = {}
                if url_m: entry["url"] = url_m.group().rstrip(".,)")
                if count_m and any(u in line.lower() for u in ["k", "m", "million", "lakh", "followers"]):
                    entry["followers"] = count_m.group().strip()
                if entry: info.setdefault(platform, entry)
    return {"social": info}


def _extract_about(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    facts, contact, social = [], {}, {}
    for line in lines:
        lower = line.lower()
        if len(line) < 5: continue
        if re.search(r'\+\d[\d\s\-]{8,}', line):
            m = re.search(r'\+[\d\s\-]+', line)
            if m: contact.setdefault("phone", m.group().strip())
        if re.search(r'[\w.\-]+@[\w.\-]+\.\w+', line):
            m = re.search(r'[\w.\-]+@[\w.\-]+\.\w+', line)
            if m: contact.setdefault("email", m.group())
        for platform in ["instagram", "youtube", "facebook", "tiktok"]:
            if platform in lower:
                url_m = re.search(r'https?://\S+', line)
                if url_m: social[platform] = url_m.group().rstrip(".,)")
        if (re.match(r'^[-•*]\s|^\d+[.)]\s', line) or
                any(k in lower for k in ["year", "experience", "speciali", "known for",
                                          "follower", "million", "brand", "founded",
                                          "premium", "luxury", "celebrity", "artist"])):
            clean = re.sub(r'^[-•*\d.)]+\s*', '', line).strip()
            if len(clean) > 15: facts.append(clean)
    return {"facts": list(dict.fromkeys(facts))[:10], "contact": contact, "social": social}


def _extract_tips(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sections, current_sec, current_tips = [], None, []
    for line in lines:
        if line.endswith(":") and len(line) < 50:
            if current_sec and current_tips:
                sections.append({"name": current_sec, "tips": current_tips})
            current_sec, current_tips = line.rstrip(":"), []
        elif re.match(r'^[-•*]\s|^\d+[.)]\s', line):
            clean = re.sub(r'^[-•*\d.)]+\s*', '', line).strip()
            if clean: current_tips.append(clean)
        elif current_sec and line:
            current_tips.append(line)
    if current_sec and current_tips:
        sections.append({"name": current_sec, "tips": current_tips})
    if not sections:
        tips = [re.sub(r'^[-•*\d.)]+\s*', '', l).strip()
                for l in lines if re.match(r'^[-•*]\s|^\d+[.)]\s', l)]
        if tips: sections.append({"name": "Makeup Tips", "tips": tips})
    return {"sections": sections}


def _extract_faq(text: str, query: str) -> Dict[str, Any]:
    pairs, current_q, current_a = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if re.match(r'^Q:\s', line, re.I):
            if current_q and current_a:
                pairs.append({"q": current_q, "a": " ".join(current_a)})
            current_q = re.sub(r'^Q:\s*', '', line, flags=re.I); current_a = []
        elif re.match(r'^A:\s', line, re.I):
            current_a = [re.sub(r'^A:\s*', '', line, flags=re.I)]
        elif current_a is not None and not re.match(r'^Q:\s', line, re.I):
            current_a.append(line)
    if current_q and current_a:
        pairs.append({"q": current_q, "a": " ".join(current_a)})
    keywords = extract_keywords(query)
    if keywords:
        def sc(p): return sum(1 for k in keywords if k in (p["q"] + p["a"]).lower())
        pairs.sort(key=sc, reverse=True)
    return {"qa_pairs": pairs[:4]}


def _extract_general(text: str, query: str) -> Dict[str, Any]:
    parts     = re.split(r'(?<=[.!?])\s+', text)
    sentences = [p.strip() for p in parts if len(p.strip()) >= 20]
    keywords  = extract_keywords(query)
    def sc(s): return sum(1 for k in keywords if k in s.lower())
    scored = sorted(sentences, key=sc, reverse=True)
    seen, result = set(), []
    for s in scored:
        key = s.lower()[:50]
        if key not in seen: seen.add(key); result.append(s)
    return {"sentences": result[:MAX_SUMMARY_SENTENCES]}


# ══════════════════════════════════════════════════════════════════════════
# C) Main public function  — now accepts content_hint
# ══════════════════════════════════════════════════════════════════════════

def extractive_summarise(
    query:        str,
    chunks:       List[dict],
    content_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect content type and extract structured data from retrieved chunks.

    Args:
        query:        Original user message (used for content-type heuristics).
        chunks:       Retrieved KB chunks.
        content_hint: Optional NLU-predicted content-type label to use as override.

    Returns a rich dict consumed by formatter.build_answer().
    """
    if not chunks:
        return {"content_type": "general", "structured": {},
                "all_text": "", "chunk_titles": [], "sentences": []}

    chunk_titles = list(dict.fromkeys(c.get("title", "") for c in chunks))
    all_text     = "\n\n".join(
        c.get("chunk_text", "").strip() for c in chunks if c.get("chunk_text")
    )
    all_text = safe_truncate(all_text, MAX_CONTEXT_CHARS)

    content_type = detect_content_type(chunks, query, hint=content_hint)

    dispatch = {
        "pricing":  lambda: _extract_pricing(all_text),
        "services": lambda: _extract_services(all_text),
        "booking":  lambda: _extract_booking(all_text),
        "contact":  lambda: _extract_contact(all_text),
        "social":   lambda: _extract_social(all_text),
        "about":    lambda: _extract_about(all_text),
        "tips":     lambda: _extract_tips(all_text),
        "faq":      lambda: _extract_faq(all_text, query),
        "events":   lambda: _extract_general(all_text, query),
        "general":  lambda: _extract_general(all_text, query),
    }

    structured = dispatch.get(content_type, dispatch["general"])()
    sentences  = _extract_general(all_text, query).get("sentences", [])

    return {
        "content_type": content_type,
        "structured":   structured,
        "all_text":     all_text,
        "chunk_titles": chunk_titles,
        "sentences":    sentences,
    }