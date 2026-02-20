"""
conversational.py — Canned greeting / small-talk replies, emotion-aware.
Only used when intent_mode is greeting or small_talk (no KB needed).
"""

_GREETING = {
    "happy":   "Hey there! 😊 Great to see you! I'm JinniChirag's AI assistant — ask me anything about makeup services, pricing, booking, or Chirag Sharma!",
    "excited": "Hey!! 🎉 Welcome to JinniChirag Makeup Artist! I'm SO excited to help you today — ask me about services, pricing, or how to book!",
    "neutral": (
        "Hello! 👋 Welcome to JinniChirag Makeup Artist. I can help you with:\n"
        "  • Services and pricing 💄\n"
        "  • Booking an appointment 📅\n"
        "  • Contact and social media 📱\n"
        "  • Makeup tips and FAQs ✨\n\n"
        "What would you like to know?"
    ),
    "sad":     "Hello! 💙 I'm here for you. Feel free to ask anything — I'll do my best to help you.",
    "angry":   "Hi there. I'm sorry if something has frustrated you — I'm here and ready to help right away.",
    "confused":"Hello! Don't worry at all 😊 — I'm here to explain anything about JinniChirag's services. What would you like to know?",
}

_SMALL_TALK = {
    "happy":   "I'm doing great, thanks! 😄 I can help with services, pricing, bookings, and more. What's on your mind?",
    "excited": "I'm full of energy and ready to help! 🌟 Ask me about services, pricing, or how to book an appointment!",
    "neutral": (
        "I'm doing well, thank you! I'm JinniChirag's AI assistant. I can help with:\n"
        "  • Services and pricing 💄\n"
        "  • Booking an appointment 📅\n"
        "  • Contact information 📱\n"
        "  • Makeup tips ✨\n\n"
        "What would you like to know?"
    ),
    "sad":     "I'm always here to help! 💙 Just ask me anything about JinniChirag Makeup Artist.",
    "angry":   "I'm here and ready to help. What do you need?",
    "confused":"I'm JinniChirag's AI assistant! I can help with services, pricing, bookings, and more. What would you like to know?",
}


def get_conversational_reply(intent_mode: str, emotion: str) -> str:
    em = emotion if emotion in _GREETING else "neutral"
    return _GREETING[em] if intent_mode == "greeting" else _SMALL_TALK[em]