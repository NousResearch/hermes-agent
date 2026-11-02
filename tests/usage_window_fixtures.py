"""Test helper: the server's usage-window gauge list from the old per-window keyword names.

The Portal account and the API send one ``gauges`` list (agent/usage_gauges.py). Tests still read best as "day 76%,
week 40%, top tier full", so this maps those names onto the list, in the server's order.
"""

from __future__ import annotations

import time
from typing import Any, Optional

# A reset well inside parse_gauges' plausible range (no more than a year ahead).
FAR = int(time.time() * 1000) + 30 * 86_400_000
_FAR = FAR


def gauges(*, day_used_bp: int = 0, week_used_bp: int = 0, top_tier_used_bp: Optional[int] = 0,
           month_used_bp: Optional[int] = None, day_resets_at_ms: Optional[int] = _FAR,
           week_resets_at_ms: Optional[int] = _FAR, month_resets_at_ms: Optional[int] = _FAR,
           top_tier_resets_at_ms: Optional[int] = None) -> list[dict[str, Any]]:
    out = [
        {"id": "day", "scope": "all", "label": "Today", "used_bp": day_used_bp, "resets_at_ms": day_resets_at_ms},
        {"id": "week", "scope": "all", "label": "This week", "used_bp": week_used_bp, "resets_at_ms": week_resets_at_ms},
    ]
    if month_used_bp is not None:
        out.append({"id": "month", "scope": "all", "label": "This month", "used_bp": month_used_bp,
                    "resets_at_ms": month_resets_at_ms})
    if top_tier_used_bp is not None:
        out.append({"id": "top_tier", "scope": "top_tier", "label": "Large models (monthly)", "used_bp": top_tier_used_bp,
                    "resets_at_ms": top_tier_resets_at_ms or day_resets_at_ms})
    return out


_GAUGE_KEYS = frozenset({"day_used_bp", "week_used_bp", "top_tier_used_bp", "month_used_bp", "day_resets_at_ms",
                         "week_resets_at_ms", "month_resets_at_ms", "top_tier_resets_at_ms"})


def usage_window(**kw: Any) -> dict[str, Any]:
    """A portal ``usage_window`` block: the gauge keywords become ``gauges``; everything else passes through."""
    gauge_kw = {k: kw.pop(k) for k in list(kw) if k in _GAUGE_KEYS}
    return {"gauges": gauges(**gauge_kw), **kw}
