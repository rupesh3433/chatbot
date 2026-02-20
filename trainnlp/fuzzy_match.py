"""
fuzzy_match.py — Typo correction + slang expansion before embedding/LLM.

Pipeline:
  1. Full-query regex patterns  (hw 2 book → how to book)
  2. Word-level slang map       (ur → your, gimme → give me)
  3. RapidFuzz word correction  (bridl → bridal, makup → makeup)

RAM: ~5 MB   Speed: <5 ms per query
"""
import re
import logging
logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process as rfp
    _HAS_RF = True
except ImportError:
    _HAS_RF = False

# ── Domain vocabulary ──────────────────────────────────────────────────────
DOMAIN = {
    "bridal","bride","makeup","henna","mehendi","party","engagement",
    "reception","wedding","prewedding","signature","luxury","premium",
    "booking","book","appointment","schedule","availability","slot",
    "confirm","otp","payment","price","pricing","cost","charge","fee",
    "rate","package","service","services","offer",
    "contact","phone","number","email","whatsapp","instagram","youtube",
    "facebook","tiktok","social","media","follower","followers",
    "chirag","sharma","jinni","artist","celebrity","experience","brand",
    "what","how","where","when","why","who","which","can","do","does",
    "tell","give","show","explain","list","about","your","you","free",
    "short","brief","detailed","complete","full","quick","first","last",
}

# ── Slang → standard ───────────────────────────────────────────────────────
SLANG = {
    "hw":"how","wut":"what","wot":"what","ur":"your","u":"you","r":"are",
    "pls":"please","plz":"please","thx":"thanks","ty":"thank you",
    "abt":"about","info":"information","num":"number","ph":"phone",
    "ig":"instagram","yt":"youtube","fb":"facebook","tt":"tiktok",
    "insta":"instagram","wapp":"whatsapp","wa":"whatsapp",
    "gimme":"give me","gime":"give me","lemme":"let me",
    "wanna":"want to","gonna":"going to","gotta":"got to",
    "dunno":"do not know","dont":"do not","doesnt":"does not",
    "cant":"cannot","wont":"will not","isnt":"is not",
    "2":"to","4":"for","b4":"before",
    "bridl":"bridal","hnna":"henna","mehndi":"mehendi",
    "makup":"makeup","bokng":"booking","servics":"services",
    "pricng":"pricing","folwers":"followers","out":"our",
}

# ── Full-query patterns ────────────────────────────────────────────────────
PATTERNS = [
    (r"\bhw\s+2\s+book\b",              "how to book an appointment"),
    (r"\bhw\s+much\b",                  "how much does it cost"),
    (r"\bwut\s+(can\s+)?u\s+do\b",      "what can you do"),
    (r"\bgimme\s+1\s+line\b",           "give me 1 line"),
    (r"\bwanna\s+book\b",               "i want to book"),
    (r"\bbook\s+pls\b",                 "how do i book"),
    (r"\bprice\s*\?*$",                 "what is the price"),
    (r"\bcost\s*\?*$",                  "what is the cost"),
    (r"\brates\s*\?*$",                 "what are your rates"),
    (r"\bcontact\s*\?*$",               "how can i contact you"),
    (r"\bservices\s+pls\b",             "what are your services"),
    (r"\btips\s+pls\b",                 "give me makeup tips"),
    (r"\babt\s+u\b",                    "about you"),
    (r"\btell\s+me\s+abt\s+urself\b",   "tell me about yourself"),
    (r"\bur\s+insta\b",                 "your instagram link"),
    (r"\bout\s+first\s+(chat|question|message)\b", "our first question"),
    (r"\bfree\s+today\b",               "are you available today for booking"),
]


def correct_query(message: str) -> str:
    original = message.strip()
    text = original.lower()

    # Step 1: full-query patterns
    for pat, rep in PATTERNS:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # Step 2: slang word expansion
    words = text.split()
    text = " ".join(SLANG.get(w.rstrip(".,?!"), w) for w in words)

    # Step 3: fuzzy correction for unknown words
    if _HAS_RF:
        result_words = []
        for w in text.split():
            if len(w) > 3 and w not in DOMAIN:
                match = rfp.extractOne(w, DOMAIN, scorer=fuzz.ratio, score_cutoff=86)
                result_words.append(match[0] if match else w)
            else:
                result_words.append(w)
        text = " ".join(result_words)

    text = re.sub(r"\s+", " ", text).strip()
    if text != original.lower():
        print(f"[FUZZY] '{original}' → '{text}'")
    return text