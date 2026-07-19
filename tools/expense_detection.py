"""
Expense detection helper for deterministic gateway pre-agent routing.

Returns a simple intent dict so the gateway can decide whether a message
should bypass the agent and recur directly through log_expense.py.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

_EXCLUDE_LEAD_RE = re.compile(
    r"^(/|!|\.|\?| remind| schedule| create| delete| list| show| get| help| ping| todo| task| habit| punch| track| start| stop| new)\b",
    re.IGNORECASE,
)

_MERCHANTISH_RE = re.compile(
    r"""
    (?:
      (?:from|at|to|on)\s+[A-Za-z][\w\s&.'’-]{2,}
     |(?:bought|ordered|got|purchased)\s+(?:\w+\s+)?(?:from|at|on)\s+[A-Za-z][\w\s&.'’-]{2,}
     |\b(?:Blinkit|Swiggy|Zomato|Amazon\.in|Amazon\b|Flipkart|Zepto|BigBasket|Uber|Ola|Netflix|Spotify|Hostinger|OpenRouter|DMart|HP\b|Shell|BookMyShow|PVR|INOX)\b
     |\b(?:petrol|diesel|fuel|dinner|lunch|breakfast|coffee|tea|groceries|grocery|ride|trip|bill|rent|ticket|cab|auto|medicine|meds|subscription|fee|water|electricity|internet|mobile|recharge)\b
     |\b(?:Netflix|Spotify|Prime|Hotstar|Disney|YouTube\s*Premium|Swiggy|Zomato|Amazon|Flipkart|Uber|Ola)\s*(?:sub|annual|monthly|renewal|plan|\d{3,})\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_AMOUNT_RE = re.compile(
    r"""
    (?:
      (?:₹|Rs\.?|INR|rs\.?|inr)\s*\d[\d,]*\.?\d*
     |\b\d[\d,]*\.?\d*\s*(?:Rs\.?|INR|/-)\b
     |\b(?:cost|costs|charged|paid|spent)\s+\d[\d,]*\.?\d*\b
     |\b(\d[\d,]*\.?\d*)\s*(?:petrol|diesel|fuel|dinner|lunch|breakfast|coffee|tea|groceries|grocery|ride|trip|bill|rent|ticket|cab|auto|medicine|meds|subscription|fee|water|electricity|internet|mobile|recharge)\b
     |\b(\d[\d,]*\.?\d*)\b(?=[^\n]{0,60}\b(?:from|at|on|via)\b)
     |\b(?:Netflix|Spotify|Prime|Hotstar|Disney|YouTube\s*Premium|subscription|service)\s+(\d[\d,]*\.?\d*)\b
     |\b(?:dinner|lunch|breakfast|petrol|fuel|ride|trip|bill)\s+(\d[\d,]*\.?\d*)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify_expense_intent(text: str) -> Dict[str, Any]:
    message = (text or "").strip()
    if not message or _EXCLUDE_LEAD_RE.search(message):
        return {"is_expense": False, "has_amount": False, "merchantish": False, "text": message}

    has_amount = bool(_AMOUNT_RE.search(message))
    merchantish = bool(_MERCHANTISH_RE.search(message))
    return {"is_expense": bool(has_amount and merchantish), "has_amount": has_amount, "merchantish": merchantish, "text": message}

