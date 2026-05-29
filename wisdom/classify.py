"""Deterministic capture trigger and classification rules."""

from __future__ import annotations

import re

from wisdom.models import Category, Classification, SourceType, TriggerMatch


_TRIGGERS: tuple[tuple[str, Category | None, SourceType | None], ...] = (
    ("remember this:", None, None),
    ("remember this ", None, None),
    ("save this:", None, None),
    ("save this thought:", None, None),
    ("note this:", None, None),
    ("business idea:", "business", None),
    ("investing thought:", "investing", None),
    ("health note:", "health", None),
    ("life thought:", "life", None),
    ("book note:", None, "book"),
    ("book:", None, "book"),
    ("podcast note:", None, "podcast"),
    ("podcast idea:", None, "podcast"),
    ("podcast:", None, "podcast"),
)

_KEYWORDS: dict[Category, tuple[str, ...]] = {
    "business": (
        "client",
        "communication",
        "trust",
        "x10x",
        "pms",
        "aif",
        "report",
        "reporting",
        "sales",
        "team",
        "prospect",
        "business",
        "meeting",
        "ops",
        "positioning",
        "decision-making",
    ),
    "investing": (
        "stock",
        "market",
        "option",
        "portfolio",
        "risk",
        "macro",
        "thesis",
        "allocation",
        "trade",
        "sizing",
        "position size",
        "position sizing",
        "liquidity",
        "downside",
        "behavioral edge",
    ),
    "health": (
        "sleep",
        "food",
        "energy",
        "exercise",
        "gym",
        "decision quality",
        "health",
        "lunch",
        "cognition",
        "recovery",
    ),
    "life": (
        "family",
        "relationship",
        "happiness",
        "philosophy",
        "courage",
        "fear",
        "meaning",
        "habit",
        "avoidance",
        "uncomfortable",
        "systems",
    ),
    "inbox": (),
}


def detect_explicit_trigger(text: str | None) -> TriggerMatch | None:
    if not text:
        return None
    if text.lstrip().startswith("/"):
        return None
    leading = len(text) - len(text.lstrip())
    stripped = text[leading:]
    lowered = stripped.lower()
    for prefix, category_hint, source_hint in _TRIGGERS:
        if lowered.startswith(prefix):
            cleaned = stripped[len(prefix):].strip()
            return TriggerMatch(
                prefix=prefix,
                cleaned_text=cleaned,
                category_hint=category_hint,
                source_hint=source_hint,
            )
    return None


def classify_capture(original_text: str, cleaned_text: str | None = None, trigger: TriggerMatch | None = None) -> Classification:
    cleaned = (cleaned_text if cleaned_text is not None else original_text).strip()
    category = trigger.category_hint if trigger and trigger.category_hint else _score_category(cleaned or original_text)
    source_type = trigger.source_hint if trigger and trigger.source_hint else _source_type(cleaned or original_text)
    confidence = 0.78 if trigger and trigger.category_hint else 0.62 if category != "inbox" else 0.45
    if trigger and trigger.source_hint:
        confidence = max(confidence, 0.66)
    return Classification(
        category=category,
        source_type=source_type,
        title=_title_for(cleaned or original_text),
        confidence=confidence,
        importance_score=None,
        novelty_score=None,
        actionability_score=None,
    )


def _score_category(text: str) -> Category:
    lowered = text.lower()
    scores: dict[Category, int] = {category: 0 for category in _KEYWORDS if category != "inbox"}
    for category, keywords in _KEYWORDS.items():
        if category == "inbox":
            continue
        for keyword in keywords:
            if " " in keyword:
                if keyword in lowered:
                    scores[category] += 1
            elif _keyword_matches(lowered, keyword):
                scores[category] += 1
    best_score = max(scores.values()) if scores else 0
    if best_score <= 0:
        return "inbox"
    winners = [category for category, score in scores.items() if score == best_score]
    return winners[0] if len(winners) == 1 else "inbox"


def _source_type(text: str) -> SourceType:
    lowered = text.lower()
    if (
        lowered.startswith("podcast note:")
        or lowered.startswith("podcast:")
        or re.search(r"\bpodcast\s+(episode|note|idea)\b", lowered)
    ):
        return "podcast"
    if (
        lowered.startswith("book note:")
        or lowered.startswith("book:")
        or re.search(r"\bbook\s+(note|idea)\b", lowered)
    ):
        return "book"
    if lowered.startswith("quote") or " quote:" in lowered:
        return "quote"
    if "meeting" in lowered:
        return "meeting"
    if "article" in lowered:
        return "article"
    if "conversation" in lowered:
        return "conversation"
    return "thought"


def _keyword_matches(text: str, keyword: str) -> bool:
    escaped = re.escape(keyword)
    suffix = "s?" if not keyword.endswith("s") else ""
    return re.search(rf"\b{escaped}{suffix}\b", text) is not None


def _title_for(text: str) -> str:
    compact = " ".join(text.strip().split())
    if not compact:
        return "Untitled capture"
    sentence = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0]
    if len(sentence) <= 72:
        return sentence
    return sentence[:69].rstrip() + "..."
