"""Deterministic interpretation for Wisdom captures."""

from __future__ import annotations

from wisdom.db import WisdomDB
from wisdom.models import CaptureRecord, InterpretationRecord


_INSIGHT_BY_CATEGORY = {
    "business": "This may be useful as client language, positioning, or an operating principle.",
    "investing": "This may be useful as a risk, allocation, or decision rule.",
    "health": "This may be useful as a personal experiment or decision-quality reminder.",
    "life": "This may be useful as a principle, reflection, or writing seed.",
    "inbox": "This is worth preserving, but needs later context before strong classification.",
}

_APPLICATION_BY_CATEGORY = {
    "business": "Turn it into a client-facing phrase, principle, or internal task proposal.",
    "investing": "Turn it into an investment rule, checklist item, or decision guardrail.",
    "health": "Turn it into a small health experiment or decision-quality rule.",
    "life": "Turn it into a principle, writing idea, or decision rule.",
    "inbox": "Review later and decide whether it should become a principle or writing idea.",
}


def deterministic_interpretation(capture: CaptureRecord) -> dict[str, object]:
    text = (capture.cleaned_text or capture.original_text).strip()
    summary = _summary(text)
    return {
        "summary": summary,
        "insight": _INSIGHT_BY_CATEGORY.get(capture.category),
        "why_it_matters": "It was explicitly captured, so preserving the original lets later decisions refer back to the user's exact wording.",
        "possible_application": _APPLICATION_BY_CATEGORY.get(capture.category),
        "counterpoint": "This is a lightweight interpretation; revisit it before treating it as a durable rule.",
        "confidence": 0.55 if capture.category != "inbox" else 0.45,
        "method": "deterministic",
        "model_used": None,
        "metadata": {"interpreter_version": 1},
    }


def interpret_capture(db: WisdomDB, capture_id: int, *, create: bool = True) -> InterpretationRecord | None:
    existing = db.get_interpretation(capture_id)
    if existing or not create:
        return existing
    capture = db.get_capture(capture_id)
    if capture is None:
        return None
    data = deterministic_interpretation(capture)
    return db.insert_interpretation(capture_id=capture_id, **data)


def _summary(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 180:
        return compact
    return compact[:177].rstrip() + "..."
