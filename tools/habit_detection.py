
from __future__ import annotations

import re

PHRASES = [
    r"\byoga\b",
    r"\byog\b",
    r"\bcssbattle\b",
    r"\bcss battle\b",
    r"\bduolingo\b|\bduo\s*lingo\b",
    r"\bbrushed\b|\bbrushing\b|\bbrush\b",
    r"\boil(?:ing|ed)?(?:\s*hair)?\b|\boil_hair\b",
]

_KEYWORDS = {
    "yoga": re.compile(r"\byoga\b|\byog\b"),
    "cssbattle": re.compile(r"\bcssbattle\b|\bcss battle\b"),
    "duolingo": re.compile(r"\bduolingo\b|\bduo\s*lingo\b"),
    "night_brushing": re.compile(r"\bbrushed\b|\bbrushing\b|\bbrush\b"),
    "oil_hair": re.compile(r"\boil(?:ing|ed)?(?:\s*hair)?\b|\boil_hair\b"),
}

_EXCLUDE_LEAD_RE = re.compile(
    r"^(/|!|\.|\?| remind| schedule| create| delete| list| show| get| help| ping| todo| task| track| start| stop| new)\b",
    re.IGNORECASE,
)


def classify_habit_intent(text: str) -> dict:
    message = (text or "").strip()
    if not message or _EXCLUDE_LEAD_RE.search(message):
        return {"is_habit": False, "habit": None, "has_done": False, "text": message}

    message_lower = message.lower()
    matched_habit = None
    for key, rx in _KEYWORDS.items():
        if rx.search(message_lower):
            matched_habit = key
            break

    return {"is_habit": bool(matched_habit), "habit": matched_habit, "has_done": bool(matched_habit), "text": message}
