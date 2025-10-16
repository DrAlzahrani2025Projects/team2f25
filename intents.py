import re

GREETINGS = re.compile(
    r"^(hi|hello|hey|yo|good\s*(morning|afternoon|evening)|what'?s up|how (are|r) (you|u))\b",
    re.I
)

def detect_intent(text: str):
    """Simple rule-based intent detector."""
    if not text or not text.strip():
        return "empty"
    if GREETINGS.search(text):
        return "greeting"
    # look for internship-related words
    if re.search(r"\b(intern(ship)?|role|position|apply|location|remote|stipend|paid|summer|fall|spring)\b", text, re.I):
        return "search_internships"
    return "fallback"
