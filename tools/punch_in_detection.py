
import re

PUNCH_IN_RE = re.compile(
    r"\b(?:punch(?:ed|ing)?|clock(?:ed|ing)?)(?:[-\s]+)in\b",
    re.IGNORECASE,
)

def contains_punch_in(text: str) -> bool:
    return bool(PUNCH_IN_RE.search(text or ""))
