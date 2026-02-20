"""
formatter.py — Content-type-aware, emotion-driven answer builder.

v2.3 — Complete unified version.

FIXES ACROSS VERSIONS:
  • "Give me 1 line about you" → short intent → single sentence response
  • "What is your services?"   → clean numbered list, no FAQ Q&A content
  • "Are you free today?"      → booking renderer with availability note
  • Services renderer: skips FAQ content, shows only service names
  • About renderer: short mode returns one crisp sentence
  • General renderer: short mode returns 1 sentence
  • excited emotion fully supported across all openers/closers
  • Source line deduplication — unique titles only
"""

from typing import List, Dict, Any


# ══════════════════════════════════════════════════════════════════════════
# A) Openers and closers — emotion × content type
# ══════════════════════════════════════════════════════════════════════════

_OPENING: Dict[str, Dict[str, str]] = {
    "happy": {
        "pricing":  "Great news — here's the complete pricing breakdown for you! 😊",
        "services": "Absolutely! Here's everything JinniChirag Makeup Artist offers 😊",
        "booking":  "Exciting! Let me walk you through how to book 🎉",
        "contact":  "Of course! Here are all the ways to get in touch 😊",
        "social":   "Great question! Here are the social media details 😊",
        "about":    "Happy to share! Here's a bit about JinniChirag Makeup Artist 😊",
        "tips":     "Love it! Here are some pro tips from Chirag Sharma 😊",
        "faq":      "Great question! Here are the answers 😊",
        "events":   "Exciting! Here's all the info about events 😊",
        "general":  "Great question! Here's what I found for you 😊",
    },
    "excited": {
        "pricing":  "Amazing! 🎉 Here's the complete pricing just for you!",
        "services": "Woohoo! Here's everything we offer at JinniChirag Makeup Artist 🌟",
        "booking":  "Let's get you booked! 🎊 Here's how it works:",
        "contact":  "Sure thing! 🌟 Here's how to reach us:",
        "social":   "Yay! 🎉 Here are all our social media links:",
        "about":    "We're thrilled you asked! 🎉 Here's all about JinniChirag:",
        "tips":     "Love your enthusiasm! ✨ Here are top tips from Chirag Sharma:",
        "faq":      "Great energy! 🌟 Here are the answers you need:",
        "events":   "Exciting times! 🎊 Here's all the event info:",
        "general":  "Love the excitement! 🎉 Here's what I found:",
    },
    "neutral": {
        "pricing":  "Here is the complete pricing information:",
        "services": "Here is an overview of services offered by JinniChirag Makeup Artist:",
        "booking":  "Here is how to book a service with JinniChirag Makeup Artist:",
        "contact":  "Here are the contact details for JinniChirag Makeup Artist:",
        "social":   "Here are the social media details for JinniChirag Makeup Artist:",
        "about":    "Here is information about JinniChirag Makeup Artist:",
        "tips":     "Here are professional makeup tips from JinniChirag Makeup Artist:",
        "faq":      "Here are the answers to your question:",
        "events":   "Here is information about JinniChirag events:",
        "general":  "Here is the information from our knowledge base:",
    },
    "sad": {
        "pricing":  "I understand — here's the pricing information that might help you plan:",
        "services": "I'm here to help. Here's what JinniChirag Makeup Artist offers:",
        "booking":  "Don't worry — I'll walk you through the booking process:",
        "contact":  "Of course — here's how you can reach us directly:",
        "social":   "Sure — here are the social media details:",
        "about":    "I'm here to help. Here's everything about us:",
        "tips":     "Here are some helpful tips from Chirag Sharma:",
        "faq":      "I understand. Let me answer your question clearly:",
        "events":   "Here is the events information:",
        "general":  "I understand, and I'm here to help. Here's what I found:",
    },
    "angry": {
        "pricing":  "I sincerely apologise for any confusion. Here is the correct pricing:",
        "services": "I apologise for any inconvenience. Here are the exact services provided:",
        "booking":  "I'm sorry for the trouble. Here is the exact booking process:",
        "contact":  "I apologise. Here are the direct contact details:",
        "social":   "I apologise. Here are the social media details:",
        "about":    "I'm sorry for any confusion. Here is the correct information:",
        "tips":     "I'm sorry. Here are the makeup tips:",
        "faq":      "I sincerely apologise. Here is the clear answer:",
        "events":   "I apologise. Here is the events information:",
        "general":  "I sincerely apologise. Here is the correct information:",
    },
    "confused": {
        "pricing":  "No worries — let me break down the pricing clearly for you:",
        "services": "No problem at all — here's a clear list of what we offer:",
        "booking":  "No worries! Here's the booking process explained simply:",
        "contact":  "Sure! Here's exactly how you can contact us:",
        "social":   "No problem! Here are the social media details clearly:",
        "about":    "No worries — here's a clear overview of who we are:",
        "tips":     "No worries! Here are easy-to-follow makeup tips:",
        "faq":      "Good question! Let me break down the answer clearly:",
        "events":   "No worries! Here's how events work:",
        "general":  "No worries — let me explain this clearly:",
    },
}

_CLOSING: Dict[str, str] = {
    "happy":   "Hope that helps! Feel free to ask anything else 😊",
    "excited": "Hope you're as excited as we are! Ask me anything else 🌟",
    "neutral": "If you need further clarification, feel free to ask.",
    "sad":     "I'm here whenever you need more help — you've got this! 💙",
    "angry":   "I hope this resolves your concern. Please let me know if I can help further.",
    "confused":"Does that make sense? Ask me anything and I'm happy to clarify!",
}


def _get_opening(emotion: str, content_type: str) -> str:
    em = emotion if emotion in _OPENING else "neutral"
    ct = content_type if content_type in _OPENING[em] else "general"
    return _OPENING[em][ct]


def _get_closing(emotion: str) -> str:
    return _CLOSING.get(emotion, _CLOSING["neutral"])


def _source_line(titles: List[str]) -> str:
    unique = list(dict.fromkeys(t for t in titles if t))
    if not unique:
        return ""
    return "Sources:  " + "  ".join(f"📄 {t}" for t in unique)


# ══════════════════════════════════════════════════════════════════════════
# B) Greeting / small-talk replies (no KB)
# ══════════════════════════════════════════════════════════════════════════

_GREETING = {
    "happy":   "Hey there! 😊 So glad you stopped by! I'm JinniChirag's assistant — ask me anything about makeup services, pricing, bookings, or Chirag Sharma!",
    "excited": "Hey!! 🎉 Welcome to JinniChirag Makeup Artist! I'm so excited to help you today — ask me about services, pricing, or how to book!",
    "neutral": (
        "Hello! 👋 Welcome to JinniChirag Makeup Artist. I can help you with:\n"
        "  • Services and pricing\n"
        "  • Booking an appointment\n"
        "  • Contact and social media\n"
        "  • Makeup tips\n"
        "What would you like to know?"
    ),
    "sad":     "Hello! I'm here for you. 💙 Feel free to ask anything — I'll do my best to help.",
    "angry":   "Hi there. I'm sorry if something has frustrated you — I'm here and ready to help right away.",
    "confused":"Hello! Don't worry at all — I'm here to explain anything about JinniChirag's services. What would you like to know?",
}

_SMALL_TALK = {
    "happy":   "I'm doing great, thanks! 😄 I can help with services, pricing, bookings, and more about JinniChirag Makeup Artist. What's on your mind?",
    "excited": "I'm full of energy and ready to help! 🌟 Ask me about services, pricing, or how to book an appointment!",
    "neutral": (
        "I'm doing well, thank you! I'm JinniChirag's AI assistant. I can help with:\n"
        "  • Services and pricing\n"
        "  • Booking an appointment\n"
        "  • Contact information\n"
        "  • Makeup tips and FAQs\n"
        "What would you like to know?"
    ),
    "sad":     "I'm always here to help! 💙 Just ask me anything about JinniChirag Makeup Artist.",
    "angry":   "I'm here and ready to help. What do you need?",
    "confused":"I'm JinniChirag's AI assistant! I can help with services, pricing, bookings, and more. What would you like to know?",
}


def build_conversational_reply(intent_mode: str, emotion: str) -> str:
    em = emotion if emotion in _GREETING else "neutral"
    return _GREETING[em] if intent_mode == "greeting" else _SMALL_TALK[em]


# ══════════════════════════════════════════════════════════════════════════
# C) Fallback (no KB match)
# ══════════════════════════════════════════════════════════════════════════

def build_fallback_answer(emotion: str, related_titles: List[str]) -> str:
    em = emotion if emotion in _CLOSING else "neutral"
    lines = [
        _get_opening(em, "general"),
        "",
        "I could not find this specific information in our knowledge base.",
        "You may contact us directly for assistance:",
        "  📞 Phone / WhatsApp: +977 970-7613340",
        "  📧 Email: jinnie.chirag.mua101@gmail.com",
    ]
    if related_titles:
        lines += ["", "You might also find these topics helpful:"]
        for t in related_titles[:2]:
            lines.append(f"  • {t}")
    lines += ["", _get_closing(em)]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# D) Specialist content renderers
# ══════════════════════════════════════════════════════════════════════════

def _render_pricing(structured: dict) -> str:
    categories = structured.get("categories", [])
    notes      = structured.get("notes", [])
    lines = []
    for cat in categories:
        name  = cat.get("name", "")
        items = cat.get("items", [])
        if not items:
            continue
        if name:
            lines.append(f"\n{name}:")
        for item in items:
            lines.append(f"  • {item.strip().lstrip('-•* ')}")
    if notes:
        lines.append("\nImportant Notes:")
        for note in notes:
            lines.append(f"  • {note.strip().lstrip('-•* ')}")
    return "\n".join(lines).strip()


def _render_services(structured: dict, intent_mode: str = "default") -> str:
    services = structured.get("services", [])
    default_services = [
        "Bridal Makeup (Signature & Luxury packages)",
        "Party Makeup",
        "Engagement Makeup",
        "Pre-Wedding Shoot Makeup",
        "Reception Makeup",
        "Henna / Mehendi Services",
    ]
    display = services if services else default_services

    # Short / one-line mode
    if intent_mode == "short":
        return ("JinniChirag Makeup Artist offers: "
                + ", ".join(display[:4]) + ", and more. 💄")

    lines = []
    for i, svc in enumerate(display, 1):
        lines.append(f"  {i}. {svc.strip().lstrip('-•* ')}")
    lines.append("\n💬 Ask me about pricing or how to book any of these!")
    return "\n".join(lines)


def _render_booking(structured: dict) -> str:
    steps    = structured.get("steps", [])
    required = structured.get("required", [])
    lines    = ["📅 To check availability and book your slot:\n"]

    if steps:
        lines.append("How to Book:")
        for i, step in enumerate(steps[:6], 1):
            lines.append(f"  {i}. {step.strip().lstrip('-•* ')}")
    else:
        lines += [
            "  1. Say \"I want to book\" in this chat — the assistant guides you step by step.",
            "  2. Or fill out the booking form on the website.",
            "  3. You'll receive an OTP on WhatsApp to confirm your booking.",
            "  4. The admin contacts you within 24 hours to confirm availability.",
        ]

    if required:
        lines += ["", "Information Required:"]
        for item in required[:6]:
            lines.append(f"  • {item.strip().lstrip('-•* ')}")

    return "\n".join(lines)


def _render_contact(structured: dict) -> str:
    info   = structured.get("contact", {})
    labels = {
        "phone":     "📞 Phone / WhatsApp",
        "whatsapp":  "💬 WhatsApp",
        "email":     "📧 Email",
        "instagram": "📸 Instagram",
        "youtube":   "▶️  YouTube",
        "facebook":  "📘 Facebook",
        "tiktok":    "🎵 TikTok",
    }
    lines = [f"  {label}: {info[key]}"
             for key, label in labels.items() if info.get(key)]
    if not lines:
        lines = [
            "  📞 Phone / WhatsApp: +977 970-7613340",
            "  📧 Email: jinnie.chirag.mua101@gmail.com",
            "  📸 Instagram: https://www.instagram.com/_jinniechiragmua/",
            "  ▶️  YouTube: https://www.youtube.com/@jinniechiragmua",
            "  📘 Facebook: https://www.facebook.com/chirag.sharma.5477272/",
            "  🎵 TikTok: https://www.tiktok.com/@_chirag_101",
        ]
    return "\n".join(lines)


def _render_social(structured: dict) -> str:
    info  = structured.get("social", {})
    lines = []
    if "total_followers" in info:
        lines += [f"  🌟 Total combined following: {info['total_followers']}", ""]
    platform_labels = {
        "instagram": "📸 Instagram",
        "youtube":   "▶️  YouTube",
        "facebook":  "📘 Facebook",
        "tiktok":    "🎵 TikTok",
    }
    for key, label in platform_labels.items():
        val = info.get(key)
        if isinstance(val, dict):
            url = val.get("url", "")
            cnt = val.get("followers", "")
            if url:
                lines.append(f"  {label}: {url}" + (f"  ({cnt})" if cnt else ""))
        elif isinstance(val, str):
            lines.append(f"  {label}: {val}")
    if not lines:
        lines = [
            "  🌟 Combined following: over 1.4 million across all platforms",
            "  📸 Instagram: https://www.instagram.com/_jinniechiragmua/",
            "  ▶️  YouTube: https://www.youtube.com/@jinniechiragmua",
            "  📘 Facebook: https://www.facebook.com/chirag.sharma.5477272/",
            "  🎵 TikTok: https://www.tiktok.com/@_chirag_101",
        ]
    return "\n".join(lines)


def _render_about(structured: dict, intent_mode: str = "default") -> str:
    facts   = structured.get("facts", [])
    contact = structured.get("contact", {})
    social  = structured.get("social", {})

    default_facts = [
        "JinniChirag Makeup Artist is a premium makeup brand founded by Chirag Sharma.",
        "Chirag Sharma is a celebrity makeup artist and bridal specialist with over 9 years of experience.",
        "The brand has a combined social media following of over 1.4 million followers.",
        "Chirag specializes in bridal, party, engagement makeup, and henna services.",
        "The brand serves clients across India, Nepal, Pakistan, Bangladesh, and Dubai/UAE.",
    ]
    display_facts = facts if facts else default_facts

    # ── Short / 1-line mode ────────────────────────────────────────────────
    if intent_mode == "short":
        return (display_facts[0] if display_facts else
                "JinniChirag Makeup Artist is a premium bridal & party makeup brand by celebrity "
                "artist Chirag Sharma, with 9+ years of experience and 1.4M+ social media followers. 💄")

    lines = ["About JinniChirag Makeup Artist:"]
    for f in display_facts[:6]:
        lines.append(f"  • {f.strip().lstrip('-•* ')}")

    if contact:
        lines += ["", "Contact:"]
        if contact.get("phone"): lines.append(f"  📞 {contact['phone']}")
        if contact.get("email"): lines.append(f"  📧 {contact['email']}")

    if social:
        lines += ["", "Social Media:"]
        plat = {"instagram": "📸", "youtube": "▶️ ", "facebook": "📘", "tiktok": "🎵"}
        for k, icon in plat.items():
            if social.get(k):
                lines.append(f"  {icon} {social[k]}")

    return "\n".join(lines)


def _render_tips(structured: dict) -> str:
    sections = structured.get("sections", [])
    lines    = []
    for sec in sections:
        name = sec.get("name", "")
        tips = sec.get("tips", [])
        if name:
            lines.append(f"\n{name}:")
        for tip in tips:
            lines.append(f"  • {tip.strip().lstrip('-•* ')}")
    return "\n".join(lines).strip()


def _render_faq(structured: dict) -> str:
    pairs = structured.get("qa_pairs", [])
    lines = []
    for pair in pairs:
        lines += [f"  ❓ {pair['q']}", f"     {pair['a']}", ""]
    return "\n".join(lines).strip()


def _render_general(structured: dict, sentences: List[str],
                     intent_mode: str, all_text: str) -> str:
    if not sentences:
        return all_text[:500] if all_text else ""

    # 1 line / short mode → single sentence
    if intent_mode == "short":
        return sentences[0]

    if intent_mode == "steps":
        return "\n".join(f"  {i}. {s}" for i, s in enumerate(sentences[:7], 1))

    if len(sentences) == 1:
        return sentences[0]

    # Default: 2-sentence intro + bullet overflow for extras
    intro = " ".join(sentences[:2])
    remaining = sentences[2:5]
    if remaining:
        return intro + "\n\n" + "\n".join(f"  • {s}" for s in remaining)
    return intro


# ══════════════════════════════════════════════════════════════════════════
# E) Main public builder
# ══════════════════════════════════════════════════════════════════════════

def build_answer(summary: Dict[str, Any], emotion: str, intent_mode: str) -> str:
    em           = emotion if emotion in _OPENING else "neutral"
    content_type = summary.get("content_type", "general")
    structured   = summary.get("structured", {})
    sentences    = summary.get("sentences", [])
    all_text     = summary.get("all_text", "")
    titles       = summary.get("chunk_titles", [])

    # Pass intent_mode to renderers that support short mode
    renderers = {
        "pricing":  lambda: _render_pricing(structured),
        "services": lambda: _render_services(structured, intent_mode),
        "booking":  lambda: _render_booking(structured),
        "contact":  lambda: _render_contact(structured),
        "social":   lambda: _render_social(structured),
        "about":    lambda: _render_about(structured, intent_mode),
        "tips":     lambda: _render_tips(structured),
        "faq":      lambda: _render_faq(structured),
    }

    body = ""
    if content_type in renderers:
        body = renderers[content_type]()

    # Fallback to general if specialist produced nothing useful
    if not body or len(body.strip()) < 20:
        body = _render_general(structured, sentences, intent_mode, all_text)

    if not body or len(body.strip()) < 10:
        body = ("I found some information but could not structure it clearly. "
                "Please contact us at +977 970-7613340 for direct assistance.")

    parts = [_get_opening(em, content_type), "", body, "", _get_closing(em)]
    src   = _source_line(titles)
    if src:
        parts.append(src)

    return "\n".join(parts)