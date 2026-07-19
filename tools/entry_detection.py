from __future__ import annotations
import re

# Structured entry patterns — require explicit logging intent.
# Removed broad match on generic words like daily|entry|journal|note|log
# that caused false positives on conversational messages.
_ENTRY_PATTERNS = [
    # Explicit logging prefix: "log:", "entry:", "note:", "daily log:"
    re.compile(r"^(?:log|entry|note|daily\s*log|journal\s*entry)\s*[:：]\s*.+", re.I),
    # "I want to log ..." / "let me log ..."
    re.compile(r"\b(?:i|let\s*me)\s+(?:want\s+to\s+|need\s+to\s+|should|will)\s+(?:log|record|note|journal)\b", re.I),
    # "log that ..." / "note that ..."
    re.compile(r"\b(?:log|record|note)\s+that\b", re.I),
]

_EXCLUDE_LEAD_RE = re.compile(
    r"^(/|!|\.|\?| remind| schedule| create| delete| list| show| get| help| ping| todo| task| track| start| stop| new)\b",
    re.I,
)

# Explicitly excluded patterns — things that look like logging but are questions/chat
_QUESTION_RE = re.compile(
    r"\b(what|which|how|who|where|when|why)\b.*\b(have been|has been|is|are|was|were|did|do|does)\b|"
    r"\b(have been|has been)\b.*\b(logged|done|recorded|saved)\b",
    re.I,
)

def classify_entry_intent(text: str):
    message = (text or "").strip()
    if not message or _EXCLUDE_LEAD_RE.search(message):
        return {"is_entry": False, "text": message}
    if _QUESTION_RE.search(message):
        return {"is_entry": False, "text": message}
    for pattern in _ENTRY_PATTERNS:
        if pattern.search(message):
            return {"is_entry": True, "text": message}
    return {"is_entry": False, "text": message}
