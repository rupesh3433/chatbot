"""
llm_engine.py — Local LLM via Ollama (Llama 3.2 / Phi-3 / Mistral).

Runs 100% locally on your machine. No external API calls during inference.
After `ollama pull llama3.2` (one-time 2 GB download), works fully offline.

WHAT THIS DOES:
  • Builds a rich prompt with:
      - System persona (JinniChirag MUA assistant)
      - Emotion-aware tone instruction
      - Full KB context from retrieved chunks
      - Last N conversation turns (real memory)
      - User's current question
  • Calls Ollama's local HTTP server (localhost:11434)
  • Returns a grounded, human-like answer
  • Falls back gracefully if Ollama is not running

RAM: model weights stay in Ollama's process (~2-5 GB depending on model)
     this Python code uses ~0 MB extra for the LLM itself
"""
import json
import logging
import re
from typing import List, Dict, Optional

import httpx

from agent.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    OLLAMA_TIMEOUT, LLM_MAX_TOKENS, LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# A) System prompt — persona + constraints
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an intelligent AI assistant for JinniChirag Makeup Artist, a premium makeup brand founded by celebrity artist Chirag Sharma with 9+ years of experience and over 1.4 million social media followers.

YOUR ROLE:
- Answer questions about services, pricing, booking, contact info, social media, and the brand
- Use the KNOWLEDGE BASE sections provided to ground every answer in facts
- Never make up prices, dates, or contact details not in the knowledge base
- If something is not in the knowledge base, say so honestly and offer to connect the user with the team directly

TONE RULES (follow the emotion instruction at the top of each user message):
- Happy/Excited user → upbeat, enthusiastic, use emojis 🎉💄
- Sad user → warm, empathetic, gentle tone
- Angry user → calm, apologetic, direct and helpful
- Confused user → clear, step-by-step explanations
- Neutral user → professional, concise, friendly

FORMAT RULES:
- Keep answers concise (3-6 sentences for simple questions)
- Use numbered lists for steps/processes
- Use bullet points for feature lists
- Always end with an offer to help further
- Include relevant emojis to make responses friendly
- For pricing and booking details, be precise and structured

MEMORY: You remember the conversation history. Refer to earlier messages naturally when relevant (e.g., "As I mentioned earlier...", "Building on your earlier question...").

HONESTY: If the knowledge base doesn't contain the answer, say: "I don't have that specific detail in my knowledge base right now. Please contact us directly at +977 970-7613340 or jinnie.chirag.mua101@gmail.com for the most accurate information."
"""

# ── Emotion-to-tone instructions ───────────────────────────────────────────
TONE_INSTRUCTIONS = {
    "happy":   "The user seems happy and positive. Match their energy with an upbeat, cheerful response. Use friendly emojis.",
    "excited": "The user is excited! Be enthusiastic and match their excitement. Use celebratory emojis 🎉✨.",
    "sad":     "The user seems sad or stressed. Be extra warm, empathetic, and reassuring. Be gentle and supportive.",
    "angry":   "The user is frustrated or angry. Stay very calm, apologize for any inconvenience, and be extra helpful and direct.",
    "confused":"The user is confused. Give a clear, step-by-step explanation. Avoid jargon. Be patient and thorough.",
    "neutral": "Keep the response professional, concise, and friendly.",
}


# ══════════════════════════════════════════════════════════════════════════
# B) Prompt builder
# ══════════════════════════════════════════════════════════════════════════

def build_prompt_messages(
    user_message:    str,
    kb_chunks:       List[dict],
    history_pairs:   List[dict],
    emotion:         str,
    intent_mode:     str,
    nlu_hint:        Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build the full message list for the Ollama /api/chat endpoint.

    Structure:
      [system] → [history turns...] → [user with KB context]
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject conversation history (real memory)
    for pair in history_pairs[-6:]:   # last 6 turns max
        messages.append({"role": "user",      "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["bot"]})

    # Build the enriched user message with KB context
    tone  = TONE_INSTRUCTIONS.get(emotion, TONE_INSTRUCTIONS["neutral"])
    parts = [f"[TONE INSTRUCTION]: {tone}"]

    # Intent hint for output format
    if intent_mode == "short":
        parts.append("[FORMAT]: Give a short 1-2 sentence answer only.")
    elif intent_mode == "steps":
        parts.append("[FORMAT]: Give a numbered step-by-step answer.")
    elif intent_mode == "detailed":
        parts.append("[FORMAT]: Give a thorough, detailed answer covering all aspects.")

    # KB context
    if kb_chunks:
        parts.append("\n[KNOWLEDGE BASE — use this to answer the question]:")
        seen_titles = set()
        for i, chunk in enumerate(kb_chunks[:6], 1):
            title = chunk.get("title", "")
            text  = chunk.get("chunk_text", "").strip()
            label = f"Source {i}" + (f" ({title})" if title and title not in seen_titles else "")
            seen_titles.add(title)
            parts.append(f"\n{label}:\n{text}")
        parts.append("\n[END OF KNOWLEDGE BASE]")
    else:
        parts.append("\n[NOTE]: No specific knowledge base content was found for this query. Answer based on your general knowledge of JinniChirag Makeup Artist if possible, or honestly state you don't have this information.")

    parts.append(f"\n[USER QUESTION]: {user_message}")

    messages.append({"role": "user", "content": "\n".join(parts)})
    return messages


# ══════════════════════════════════════════════════════════════════════════
# C) Ollama API caller
# ══════════════════════════════════════════════════════════════════════════

def _call_ollama(messages: List[Dict[str, str]]) -> str:
    """
    POST to Ollama's /api/chat endpoint (local, no internet required).
    Returns the assistant's text response.
    Raises on timeout or connection error.
    """
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
            "top_p":       0.9,
            "repeat_penalty": 1.1,
        },
    }
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()


def _clean_response(text: str) -> str:
    """Remove any LLM artifacts from the response."""
    # Remove markdown code blocks if LLM wrapped the answer
    text = re.sub(r"```[\s\S]*?```", "", text).strip()
    # Remove leading role labels like "Assistant:" that some models add
    text = re.sub(r"^(Assistant|Bot|AI):\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════
# D) Main public function
# ══════════════════════════════════════════════════════════════════════════

def generate_answer(
    user_message:    str,
    kb_chunks:       List[dict],
    history_pairs:   List[dict],
    emotion:         str,
    intent_mode:     str,
    nlu_hint:        Optional[str] = None,
) -> dict:
    """
    Generate a grounded, human-like answer using the local LLM.

    Returns:
        {
            "answer":   str,    # the generated answer
            "llm_used": bool,   # True if LLM was used, False if fallback
            "error":    str,    # empty string if no error
        }
    """
    try:
        messages = build_prompt_messages(
            user_message  = user_message,
            kb_chunks     = kb_chunks,
            history_pairs = history_pairs,
            emotion       = emotion,
            intent_mode   = intent_mode,
            nlu_hint      = nlu_hint,
        )
        raw    = _call_ollama(messages)
        answer = _clean_response(raw)
        print(f"[LLM] Generated {len(answer)} chars via {OLLAMA_MODEL}")
        return {"answer": answer, "llm_used": True, "error": ""}

    except httpx.ConnectError:
        msg = (
            "⚠️ The local AI model (Ollama) is not running. "
            "Please start it with: `ollama serve`\n\n"
            "You can still contact us directly:\n"
            "📞 Phone: +977 970-7613340\n"
            "📧 Email: jinnie.chirag.mua101@gmail.com"
        )
        logger.warning("[LLM] Ollama not running — ConnectError")
        return {"answer": msg, "llm_used": False, "error": "ollama_not_running"}

    except httpx.TimeoutException:
        msg = (
            "⏱️ The AI model took too long to respond. "
            "Please try a shorter question, or contact us directly:\n"
            "📞 Phone: +977 970-7613340"
        )
        logger.warning("[LLM] Ollama timeout")
        return {"answer": msg, "llm_used": False, "error": "timeout"}

    except Exception as e:
        logger.error(f"[LLM] Unexpected error: {e}")
        msg = (
            "I encountered an issue generating a response. "
            "Please contact us directly:\n"
            "📞 Phone: +977 970-7613340\n"
            "📧 Email: jinnie.chirag.mua101@gmail.com"
        )
        return {"answer": msg, "llm_used": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# E) Health check
# ══════════════════════════════════════════════════════════════════════════

def check_ollama_health() -> dict:
    """
    Check if Ollama is running and the configured model is available.
    Returns {"running": bool, "model_available": bool, "models": list}
    """
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            data   = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            model_available = any(
                OLLAMA_MODEL in m or m.startswith(OLLAMA_MODEL.split(":")[0])
                for m in models
            )
            return {"running": True, "model_available": model_available, "models": models}
    except Exception as e:
        return {"running": False, "model_available": False, "models": [], "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# F) CLI test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Ollama connection...")
    health = check_ollama_health()
    print(f"  Running: {health['running']}")
    print(f"  Model '{OLLAMA_MODEL}' available: {health.get('model_available')}")
    print(f"  Available models: {health.get('models', [])}")

    if health["running"]:
        print("\nTest generation:")
        result = generate_answer(
            user_message  = "What bridal makeup services do you offer?",
            kb_chunks     = [],
            history_pairs = [],
            emotion       = "neutral",
            intent_mode   = "default",
        )
        print(f"  LLM used: {result['llm_used']}")
        print(f"  Answer:\n{result['answer']}")