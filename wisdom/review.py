"""Deterministic review, quality, and related-capture helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from wisdom.db import WisdomDB
from wisdom.models import ApplicationRecord, CaptureRecord


@dataclass(frozen=True)
class QualityScore:
    importance: float
    actionability: float
    novelty: float
    reusability: float
    overall: float
    label: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RelatedCapture:
    capture: CaptureRecord
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewItem:
    capture: CaptureRecord
    quality: QualityScore
    suggested_action: str
    related: list[RelatedCapture] = field(default_factory=list)
    application_count: int = 0


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "have", "i", "in", "is", "it", "me", "my", "not", "of", "on", "or",
    "our", "that", "the", "this", "to", "we", "what", "when", "with", "you",
}

_ACTION_TERMS = {
    "add", "apply", "build", "call", "change", "check", "create", "decide",
    "do", "experiment", "make", "measure", "need", "review", "rule", "should",
    "size", "sizing", "test", "track", "turn", "use",
}

_REUSABLE_TERMS = {
    "checklist", "client", "clients", "decision", "experiment", "framework",
    "language", "lesson", "position", "principle", "process", "report", "reports",
    "rule", "size", "sizing", "template", "thesis",
}

_DOMAIN_TERMS = {
    "business": {"client", "clients", "report", "reports", "sales", "x10x", "pms", "aif", "prospect", "team"},
    "investing": {"allocation", "downside", "liquidity", "market", "position", "portfolio", "risk", "size", "sizing", "thesis", "trade"},
    "health": {"cognition", "decision", "energy", "exercise", "food", "health", "sleep"},
    "life": {"avoid", "avoidance", "courage", "decision", "family", "fear", "habit", "meaning", "relationship", "system", "systems"},
    "inbox": set(),
}

_METAPHOR_TERMS = {
    "alpha", "beats", "bridge", "cadence", "mirror", "mirrors", "road",
    "survivability", "windshield", "windshields",
}


def build_review_items(
    db: WisdomDB,
    *,
    category: str | None = None,
    mode: str = "needs_review",
    limit: int = 5,
) -> list[ReviewItem]:
    captures = db.list_captures(limit=250, include_archived=False)
    captures = _filter_captures(captures, category=category, mode=mode)
    items: list[ReviewItem] = []
    for capture in captures:
        applications = db.list_applications(capture.id)
        related = related_captures(db, capture.id, limit=3)
        quality = score_capture(capture, applications=applications, related_count=len(related))
        items.append(
            ReviewItem(
                capture=capture,
                quality=quality,
                suggested_action=suggested_action(capture, applications, quality),
                related=related,
                application_count=len(applications),
            )
        )
    if _mode(mode) == "high_potential":
        items = [item for item in items if item.quality.overall >= 0.55]
    items.sort(key=lambda item: _review_priority(item, mode=mode), reverse=True)
    return items[: max(1, min(50, int(limit)))]


def related_captures(db: WisdomDB, capture_id: int, *, limit: int = 5) -> list[RelatedCapture]:
    target = db.get_capture(capture_id)
    if target is None:
        return []
    target_tokens = _tokens(_capture_text(target))
    if not target_tokens:
        return []

    candidates: dict[int, CaptureRecord] = {}
    fts_query = " ".join(_top_terms(target_tokens, max_terms=8))
    if fts_query:
        for record in db.search(fts_query, limit=50):
            if record.id != target.id:
                candidates[record.id] = record
    for record in db.list_captures(limit=250, include_archived=False):
        if record.id != target.id:
            candidates.setdefault(record.id, record)

    related: list[RelatedCapture] = []
    for candidate in candidates.values():
        score, reasons = _related_score(target, candidate, target_tokens)
        if score >= 0.24:
            related.append(RelatedCapture(capture=candidate, score=_round(score), reasons=reasons))
    related.sort(key=lambda item: (item.score, item.capture.created_at), reverse=True)
    return related[: max(1, min(20, int(limit)))]


def score_capture(
    capture: CaptureRecord,
    *,
    applications: list[ApplicationRecord] | None = None,
    related_count: int = 0,
) -> QualityScore:
    text = _capture_text(capture)
    tokens = _tokens(text)
    token_set = set(tokens)
    applications = applications or []

    domain_hits = len(token_set & _DOMAIN_TERMS.get(capture.category, set()))
    action_hits = len(token_set & _ACTION_TERMS)
    reusable_hits = len(token_set & _REUSABLE_TERMS)
    metaphor_hits = len(token_set & _METAPHOR_TERMS)
    length_bonus = _length_bonus(len(tokens))

    importance = 0.22 + length_bonus
    if capture.category in {"business", "investing"}:
        importance += 0.22
    elif capture.category in {"health", "life"}:
        importance += 0.14
    importance += min(0.18, domain_hits * 0.045)
    importance += min(0.08, related_count * 0.03)

    actionability = 0.22 + min(0.32, action_hits * 0.08) + min(0.22, reusable_hits * 0.055)
    if capture.category in {"business", "investing", "health"}:
        actionability += 0.12
    elif capture.category == "life":
        actionability += 0.06
    if metaphor_hits:
        actionability += 0.06
    if applications:
        actionability -= 0.08

    novelty = 0.20 + min(0.24, metaphor_hits * 0.06) + length_bonus
    if related_count == 0:
        novelty += 0.08
    elif related_count >= 2:
        novelty += 0.03
    if len(token_set) >= 8:
        novelty += 0.06

    reusability = 0.20 + min(0.30, reusable_hits * 0.075)
    if capture.category in {"business", "investing"}:
        reusability += 0.22
    elif capture.category in {"health", "life"}:
        reusability += 0.12
    if related_count:
        reusability += min(0.10, related_count * 0.035)
    if metaphor_hits:
        reusability += 0.06

    if capture.review_status == "accepted":
        importance += 0.08
        reusability += 0.08
    elif capture.review_status == "dismissed":
        importance -= 0.25
        actionability -= 0.20
        reusability -= 0.20
    elif capture.review_status == "applied":
        actionability -= 0.12

    importance = _clamp(importance)
    actionability = _clamp(actionability)
    novelty = _clamp(novelty)
    reusability = _clamp(reusability)
    overall = _clamp(importance * 0.30 + actionability * 0.30 + novelty * 0.12 + reusability * 0.28)

    reasons = _quality_reasons(capture, domain_hits, action_hits, reusable_hits, metaphor_hits, related_count, applications)
    return QualityScore(
        importance=_round(importance),
        actionability=_round(actionability),
        novelty=_round(novelty),
        reusability=_round(reusability),
        overall=_round(overall),
        label=_label(overall),
        reasons=reasons,
    )


def suggested_action(
    capture: CaptureRecord,
    applications: list[ApplicationRecord],
    quality: QualityScore,
) -> str:
    if capture.review_status == "dismissed":
        return "Keep dismissed unless the idea becomes relevant again."
    if capture.review_status == "applied" or applications:
        return "Review existing application proposals; archive if no longer useful."
    if capture.category == "business":
        return "Turn into client language, a principle, or a report/process improvement."
    if capture.category == "investing":
        return "Turn into an investment rule, checklist, or decision rule."
    if capture.category == "health":
        return "Turn into a small health experiment or decision-quality rule."
    if capture.category == "life":
        return "Turn into a principle, writing idea, or personal decision rule."
    if quality.overall >= 0.58:
        return "Accept or apply this; it has enough specificity to revisit."
    return "Dismiss if it still feels vague after review."


def review_counts(db: WisdomDB) -> dict[str, int]:
    captures = db.list_captures(limit=500, include_archived=False)
    applications_by_capture = {capture.id: db.list_applications(capture.id) for capture in captures}
    high_potential = 0
    for capture in captures:
        related = related_captures(db, capture.id, limit=3)
        score = score_capture(capture, applications=applications_by_capture[capture.id], related_count=len(related))
        if score.overall >= 0.55 and capture.review_status not in {"dismissed", "applied", "archived"}:
            high_potential += 1
    return {
        "needs_review": sum(1 for c in captures if c.review_status in {"unreviewed", "reviewed", "accepted"}),
        "high_potential": high_potential,
        "unapplied": sum(1 for c in captures if c.review_status != "dismissed" and not applications_by_capture[c.id]),
        "dismissed": sum(1 for c in captures if c.review_status == "dismissed"),
        "applied": sum(1 for c in captures if c.review_status == "applied" or applications_by_capture[c.id]),
    }


def _filter_captures(captures: list[CaptureRecord], *, category: str | None, mode: str) -> list[CaptureRecord]:
    category = (category or "").strip().lower() or None
    if category:
        captures = [capture for capture in captures if capture.category == category]
    mode = _mode(mode)
    if mode == "all":
        return captures
    if mode == "unapplied":
        return [capture for capture in captures if capture.review_status not in {"dismissed", "applied", "archived"}]
    if mode == "high_potential":
        return [capture for capture in captures if capture.review_status not in {"dismissed", "applied", "archived"}]
    return [capture for capture in captures if capture.review_status in {"unreviewed", "reviewed", "accepted"}]


def _review_priority(item: ReviewItem, *, mode: str) -> float:
    capture = item.capture
    status_bonus = {
        "accepted": 0.22,
        "unreviewed": 0.18,
        "reviewed": 0.06,
        "dismissed": -0.40,
        "applied": -0.25,
        "archived": -1.0,
    }.get(capture.review_status, 0.0)
    application_penalty = 0.18 if item.application_count else 0.0
    mode_bonus = 0.0
    if mode == "unapplied" and not item.application_count:
        mode_bonus = 0.18
    elif mode == "high_potential" and item.quality.overall >= 0.60:
        mode_bonus = 0.18
    age_bonus = min(0.08, max(0.0, math.log1p(max(0.0, capture.updated_at - capture.created_at)) / 100.0))
    return item.quality.overall + status_bonus + mode_bonus + age_bonus - application_penalty


def _related_score(target: CaptureRecord, candidate: CaptureRecord, target_tokens: list[str]) -> tuple[float, list[str]]:
    candidate_tokens = _tokens(_capture_text(candidate))
    if not candidate_tokens:
        return 0.0, []
    target_set = set(target_tokens)
    candidate_set = set(candidate_tokens)
    overlap = target_set & candidate_set
    union = target_set | candidate_set
    score = (len(overlap) / max(1, len(union))) * 1.2
    reasons: list[str] = []
    if overlap:
        reasons.append("shared terms: " + ", ".join(sorted(overlap)[:5]))
    if target.category == candidate.category:
        score += 0.18
        reasons.append(f"same category: {target.category}")
    if target.source_type == candidate.source_type:
        score += 0.04
    domain_overlap = overlap & _DOMAIN_TERMS.get(target.category, set())
    if domain_overlap:
        score += min(0.10, len(domain_overlap) * 0.04)
        reasons.append("same domain language")
    recency_days = abs(target.created_at - candidate.created_at) / 86400
    if recency_days <= 14:
        score += 0.04
        reasons.append("captured near the same period")
    return _clamp(score), reasons[:4]


def _quality_reasons(
    capture: CaptureRecord,
    domain_hits: int,
    action_hits: int,
    reusable_hits: int,
    metaphor_hits: int,
    related_count: int,
    applications: list[ApplicationRecord],
) -> list[str]:
    reasons: list[str] = []
    if capture.category in {"business", "investing"}:
        reasons.append(f"high-leverage category: {capture.category}")
    if domain_hits:
        reasons.append("specific domain language")
    if action_hits:
        reasons.append("contains action-oriented language")
    if reusable_hits:
        reasons.append("can become reusable language/process")
    if metaphor_hits:
        reasons.append("has metaphor or memorable phrasing")
    if related_count:
        reasons.append(f"connects to {related_count} related capture(s)")
    if applications:
        reasons.append("already has application proposals")
    return reasons[:5]


def _capture_text(capture: CaptureRecord) -> str:
    return " ".join(part for part in (capture.title, capture.cleaned_text, capture.original_text) if part)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _top_terms(tokens: Iterable[str], *, max_terms: int) -> list[str]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return [term for term, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_terms]]


def _length_bonus(token_count: int) -> float:
    if token_count < 5:
        return -0.08
    if token_count <= 24:
        return 0.10
    if token_count <= 60:
        return 0.06
    return -0.02


def _mode(mode: str | None) -> str:
    text = (mode or "needs_review").strip().lower().replace("-", "_")
    if text in {"all", "unapplied", "high_potential"}:
        return text
    return "needs_review"


def _label(score: float) -> str:
    if score >= 0.68:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float) -> float:
    return round(_clamp(value), 2)
