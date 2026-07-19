from __future__ import annotations
import re

# Work-log patterns — require structured format, not conversational words.
# Removed broad match on generic words like work|task|meeting|sync|deploy|review
# that caused false positives on everyday chat.
_WORK_PATTERNS = [
    # Project identifier + action: "PROJ-123 note ...", "PROJ-123 time 60 ..."
    re.compile(r"^[A-Za-z]+-\d+\s+(note|time|status)\b", re.I),
    # "spent 60m on PROJ-123 ..." / "spent 1h on ..."
    re.compile(r"\bspent\s+\d+\s*(?:m|min|mins|minutes|h|hr|hrs|hours)?\s+on\b", re.I),
    # Explicit work-log prefix: "work log:", "standup:", "task:"
    re.compile(r"^(?:work\s*log|standup|task|time\s*log)\s*[:：]\s*.+", re.I),
    # "I worked on ... for X hours"
    re.compile(r"\b(?:worked|spent)\s+on\s+.+\s+for\s+\d+\s*(?:min|hour|hr)", re.I),
]

_EXCLUDE_LEAD_RE = re.compile(
    r"^(/|!|\.|\?| remind| schedule| create| delete| list| show| get| help| ping| todo| task| track| start| stop| new)\b",
    re.I,
)

def classify_work_intent(text: str):
    message = (text or "").strip()
    if not message or _EXCLUDE_LEAD_RE.search(message):
        return {"is_work": False, "text": message}
    for pattern in _WORK_PATTERNS:
        if pattern.search(message):
            return {"is_work": True, "text": message}
    return {"is_work": False, "text": message}
