"""High-level Wisdom operations shared by commands and model tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wisdom.apply import create_application_proposals
from wisdom.capture import capture_text, effective_capture_mode, effective_enabled
from wisdom.config import load_wisdom_config
from wisdom.db import WisdomDB
from wisdom.interpret import interpret_capture
from wisdom.models import (
    ApplicationRecord,
    CaptureOutcome,
    CaptureRecord,
    InterpretationRecord,
    StatusSnapshot,
    VALID_APPLICATION_TYPES,
    VALID_CATEGORIES,
    VALID_SOURCE_TYPES,
    WisdomConfig,
)
from wisdom.review import (
    RelatedCapture,
    ReviewItem,
    build_review_items,
    related_captures,
    review_counts,
)


@dataclass(frozen=True)
class WisdomServiceContext:
    channel: str = "model_tool"
    source_kind: str = "model_tool"
    session_key: object | None = None
    message_ref: object | None = None


@dataclass(frozen=True)
class ReviewData:
    counts: dict[str, int]
    recent: list[CaptureRecord]
    unapplied: list[CaptureRecord]
    items: list[ReviewItem] = field(default_factory=list)
    mode: str = "needs_review"
    category: str | None = None
    period: str | None = None


def status_snapshot(*, config: WisdomConfig | None = None, db: WisdomDB | None = None) -> StatusSnapshot:
    config, db = _resolve(config, db)
    return db.status_snapshot(config)


def set_enabled(
    enabled: bool,
    *,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> StatusSnapshot:
    config, db = _resolve(config, db)
    db.set_setting("enabled", "true" if enabled else "false")
    db.set_setting("capture_mode", "explicit" if enabled else "off")
    return db.status_snapshot(config)


def can_natural_capture(*, config: WisdomConfig | None = None, db: WisdomDB | None = None) -> bool:
    config, db = _resolve(config, db)
    return effective_enabled(db, config) and effective_capture_mode(db, config) == "explicit"


def capture(
    text: str,
    *,
    source_kind: str | None = None,
    category: str | None = None,
    source_type: str | None = None,
    context_note: str | None = None,
    context: WisdomServiceContext | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
    require_enabled: bool = True,
) -> CaptureOutcome:
    if not str(text or "").strip():
        return CaptureOutcome("error", message="Capture text is required.")
    config, db = _resolve(config, db)
    context = context or WisdomServiceContext()
    return capture_text(
        text,
        channel=context.channel,
        source_kind=source_kind or context.source_kind,
        session_key=context.session_key,
        message_ref=context.message_ref,
        config=config,
        db=db,
        category=_valid(category, VALID_CATEGORIES),
        source_type=_valid(source_type, VALID_SOURCE_TYPES),
        context_note=context_note,
        require_enabled=require_enabled,
    )


def inbox(
    *,
    category: str | None = None,
    limit: int | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> list[CaptureRecord]:
    config, db = _resolve(config, db)
    records = db.list_captures(limit=_limit(limit, config.max_results, maximum=50), include_archived=False)
    category = _valid(category, VALID_CATEGORIES)
    return [record for record in records if record.category == category] if category else records


def search(
    query: str,
    *,
    category: str | None = None,
    limit: int | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> list[CaptureRecord]:
    config, db = _resolve(config, db)
    query = str(query or "").strip()
    if not query:
        return []
    result_limit = _limit(limit, config.max_results, maximum=50)
    category = _valid(category, VALID_CATEGORIES)
    records = db.search(query, limit=result_limit if not category else min(50, max(result_limit * 5, result_limit)))
    if category:
        records = [record for record in records if record.category == category]
    return records[:result_limit]


def original(
    capture_id: int,
    *,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> CaptureRecord | None:
    _config, db = _resolve(config, db)
    return db.get_capture(capture_id)


def interpret(
    capture_id: int,
    *,
    create: bool = True,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> InterpretationRecord | None:
    _config, db = _resolve(config, db)
    return interpret_capture(db, capture_id, create=create)


def apply(
    capture_id: int,
    *,
    application_type: str | None = None,
    context: str | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> list[ApplicationRecord]:
    config, db = _resolve(config, db)
    app_type = _valid(application_type, VALID_APPLICATION_TYPES)
    _ = context
    records = create_application_proposals(
        db,
        capture_id,
        mode=config.application_mode,
        timeout=config.apply_timeout_seconds,
    )
    if records:
        db.set_review_status(capture_id, "applied")
    return [record for record in records if record.application_type == app_type] if app_type else records


def accept(
    capture_id: int,
    *,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> CaptureRecord | None:
    _config, db = _resolve(config, db)
    if not db.set_review_status(capture_id, "accepted"):
        return None
    return db.get_capture(capture_id)


def dismiss(
    capture_id: int,
    *,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> CaptureRecord | None:
    _config, db = _resolve(config, db)
    if not db.set_review_status(capture_id, "dismissed"):
        return None
    return db.get_capture(capture_id)


def archive(
    capture_id: int,
    *,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> bool:
    _config, db = _resolve(config, db)
    return db.archive_capture(capture_id)


def review(
    *,
    category: str | None = None,
    mode: str | None = None,
    period: str | None = None,
    limit: int | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> ReviewData:
    config, db = _resolve(config, db)
    result_limit = _limit(limit, config.max_results, maximum=50)
    category = _valid(category, VALID_CATEGORIES)
    mode = _review_mode(mode)
    recent = db.list_captures(limit=result_limit if not category else min(50, max(result_limit * 5, result_limit)))
    unapplied = db.unapplied_captures(limit=result_limit if not category else min(50, max(result_limit * 5, result_limit)))
    items = build_review_items(db, category=category, mode=mode, limit=result_limit)
    if category:
        recent = [record for record in recent if record.category == category][:result_limit]
        unapplied = [record for record in unapplied if record.category == category][:result_limit]
        counts = {category: db.count_by_category().get(category, 0)}
    else:
        counts = db.count_by_category()
    counts = {**counts, **review_counts(db)}
    return ReviewData(
        counts=counts,
        recent=recent[:result_limit],
        unapplied=unapplied[:result_limit],
        items=items,
        mode=mode,
        category=category,
        period=period,
    )


def related(
    capture_id: int,
    *,
    limit: int | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> list[RelatedCapture]:
    config, db = _resolve(config, db)
    return related_captures(db, capture_id, limit=_limit(limit, config.max_results, maximum=20))


def status_payload(*, config: WisdomConfig | None = None, db: WisdomDB | None = None) -> dict[str, Any]:
    return {"ok": True, "status": status_to_dict(status_snapshot(config=config, db=db))}


def capture_payload(
    text: str,
    *,
    source_kind: str | None = None,
    category: str | None = None,
    source_type: str | None = None,
    context_note: str | None = None,
    context: WisdomServiceContext | None = None,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> dict[str, Any]:
    outcome = capture(
        text,
        source_kind=source_kind,
        category=category,
        source_type=source_type,
        context_note=context_note,
        context=context,
        config=config,
        db=db,
        require_enabled=True,
    )
    if outcome.status == "captured" and outcome.capture:
        return {
            "ok": True,
            "status": outcome.status,
            "capture": capture_to_dict(outcome.capture),
            "original_saved_exactly": True,
            "message": (
                f"Captured #{outcome.capture.id} as "
                f"{outcome.capture.category}/{outcome.capture.source_type}."
            ),
        }
    return {"ok": False, "status": outcome.status, "error": outcome.message or "Capture failed."}


def captures_payload(captures: list[CaptureRecord], *, title: str) -> dict[str, Any]:
    return {
        "ok": True,
        "title": title,
        "count": len(captures),
        "captures": [capture_to_dict(capture) for capture in captures],
    }


def original_payload(
    capture_id: int,
    *,
    config: WisdomConfig | None = None,
    db: WisdomDB | None = None,
) -> dict[str, Any]:
    capture_record = original(capture_id, config=config, db=db)
    if capture_record is None:
        return {"ok": False, "error": f"Capture #{capture_id} was not found.", "capture_id": capture_id}
    return {
        "ok": True,
        "capture_id": capture_record.id,
        "original_text": capture_record.original_text,
        "exact_original": True,
    }


def interpretation_payload(record: InterpretationRecord | None, *, capture_id: int) -> dict[str, Any]:
    if record is None:
        return {"ok": False, "error": f"No interpretation exists for capture #{capture_id}.", "capture_id": capture_id}
    return {"ok": True, "interpretation": interpretation_to_dict(record)}


def applications_payload(applications: list[ApplicationRecord], *, capture_id: int) -> dict[str, Any]:
    return {
        "ok": True,
        "capture_id": capture_id,
        "count": len(applications),
        "external_actions_created": False,
        "applications": [application_to_dict(application) for application in applications],
    }


def related_payload(related_items: list[RelatedCapture], *, capture_id: int) -> dict[str, Any]:
    return {
        "ok": True,
        "capture_id": capture_id,
        "count": len(related_items),
        "related": [related_to_dict(item) for item in related_items],
        "embeddings_used": False,
    }


def review_payload(data: ReviewData) -> dict[str, Any]:
    return {
        "ok": True,
        "period": data.period,
        "mode": data.mode,
        "category": data.category,
        "counts": data.counts,
        "items": [review_item_to_dict(item) for item in data.items],
        "recent": [capture_to_dict(capture) for capture in data.recent],
        "unapplied": [capture_to_dict(capture) for capture in data.unapplied],
        "scheduled": False,
    }


def archive_payload(capture_id: int, archived: bool) -> dict[str, Any]:
    if not archived:
        return {"ok": False, "error": f"Capture #{capture_id} was not found.", "capture_id": capture_id}
    return {"ok": True, "capture_id": capture_id, "status": "archived", "deleted": False}


def review_action_payload(action: str, capture: CaptureRecord | None, *, capture_id: int) -> dict[str, Any]:
    if capture is None:
        return {"ok": False, "error": f"Capture #{capture_id} was not found.", "capture_id": capture_id}
    return {
        "ok": True,
        "action": action,
        "capture": capture_to_dict(capture),
        "deleted": False,
    }


def status_to_dict(snapshot: StatusSnapshot) -> dict[str, Any]:
    return {
        "enabled": snapshot.enabled,
        "capture_mode": snapshot.capture_mode,
        "db_path": str(Path(snapshot.db_path).expanduser()),
        "fts_available": snapshot.fts_available,
        "counts": dict(snapshot.counts),
        "last_capture_at": snapshot.last_capture_at,
    }


def capture_to_dict(capture: CaptureRecord) -> dict[str, Any]:
    return {
        "id": capture.id,
        "created_at": capture.created_at,
        "updated_at": capture.updated_at,
        "title": capture.title,
        "category": capture.category,
        "source_type": capture.source_type,
        "status": capture.status,
        "review_status": capture.review_status,
        "reviewed_at": capture.reviewed_at,
        "accepted_at": capture.accepted_at,
        "dismissed_at": capture.dismissed_at,
        "applied_at": capture.applied_at,
        "confidence": capture.confidence,
        "original_excerpt": _excerpt(capture.original_text),
        "cleaned_excerpt": _excerpt(capture.cleaned_text or ""),
        "metadata": dict(capture.metadata),
    }


def review_item_to_dict(item: ReviewItem) -> dict[str, Any]:
    return {
        "capture": capture_to_dict(item.capture),
        "quality": {
            "importance": item.quality.importance,
            "actionability": item.quality.actionability,
            "novelty": item.quality.novelty,
            "reusability": item.quality.reusability,
            "overall": item.quality.overall,
            "label": item.quality.label,
            "reasons": item.quality.reasons,
        },
        "suggested_action": item.suggested_action,
        "application_count": item.application_count,
        "related": [related_to_dict(related_item) for related_item in item.related],
    }


def related_to_dict(item: RelatedCapture) -> dict[str, Any]:
    return {
        "capture": capture_to_dict(item.capture),
        "score": item.score,
        "reasons": item.reasons,
    }


def interpretation_to_dict(record: InterpretationRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "capture_id": record.capture_id,
        "created_at": record.created_at,
        "summary": record.summary,
        "insight": record.insight,
        "why_it_matters": record.why_it_matters,
        "possible_application": record.possible_application,
        "counterpoint": record.counterpoint,
        "confidence": record.confidence,
        "method": record.method,
        "model_used": record.model_used,
    }


def application_to_dict(record: ApplicationRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "capture_id": record.capture_id,
        "created_at": record.created_at,
        "application_type": record.application_type,
        "title": record.title,
        "body": record.body,
        "status": record.status,
        "metadata": dict(record.metadata),
    }


def _resolve(config: WisdomConfig | None, db: WisdomDB | None) -> tuple[WisdomConfig, WisdomDB]:
    resolved_config = config or load_wisdom_config()
    resolved_db = db or WisdomDB(resolved_config.db_path)
    resolved_db.init()
    return resolved_config, resolved_db


def _limit(value: int | None, default: int, *, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, min(maximum, parsed))


def _valid(value: str | None, allowed: set[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in allowed else None


def _review_mode(value: str | None) -> str:
    text = (value or "needs_review").strip().lower().replace("-", "_")
    if text in {"needs_review", "unapplied", "high_potential", "all"}:
        return text
    return "needs_review"


def _excerpt(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
