"""Telegram-friendly rendering for Wisdom Kernel results."""

from __future__ import annotations

from datetime import datetime

from wisdom.models import ApplicationRecord, CaptureRecord, InterpretationRecord, StatusSnapshot


def render_help() -> str:
    return (
        "Wisdom commands:\n"
        "/wisdom status\n"
        "/wisdom capture <text>\n"
        "/wisdom inbox\n"
        "/wisdom search <query>\n"
        "/wisdom original <id>\n"
        "/wisdom interpret <id>\n"
        "/wisdom apply <id>\n"
        "/wisdom archive <id>\n"
        "/wisdom review [category|unapplied|high-potential]\n"
        "/wisdom related <id>\n"
        "/wisdom accept <id>\n"
        "/wisdom dismiss <id>\n"
        "/wisdom on | /wisdom off"
    )


def render_capture(capture: CaptureRecord) -> str:
    return (
        f"Captured #{capture.id} - {capture.category.title()} - {capture.source_type.title()}\n"
        "Original saved exactly."
    )


def render_blocked_secret() -> str:
    return "Capture blocked because the text looks like it contains a secret."


def render_status(snapshot: StatusSnapshot) -> str:
    counts = snapshot.counts
    last = _date(snapshot.last_capture_at) if snapshot.last_capture_at else "never"
    return (
        "Wisdom status\n"
        f"Enabled: {'yes' if snapshot.enabled else 'no'}\n"
        f"Capture mode: {snapshot.capture_mode}\n"
        f"DB: {snapshot.db_path}\n"
        f"Captures: {counts.get('captures', 0)}\n"
        f"Interpretations: {counts.get('interpretations', 0)}\n"
        f"Applications: {counts.get('applications', 0)}\n"
        f"FTS: {'available' if snapshot.fts_available else 'LIKE fallback'}\n"
        f"Last capture: {last}"
    )


def render_captures(title: str, captures: list[CaptureRecord]) -> str:
    if not captures:
        return f"{title}\nNo captures found."
    lines = [title]
    for capture in captures:
        lines.append(
            f"#{capture.id} - {_date(capture.created_at)} - {capture.category} - "
            f"{capture.title}\n  {_excerpt(capture.original_text)}"
        )
    return "\n".join(lines)


def render_original(capture: CaptureRecord) -> str:
    return capture.original_text


def render_interpretation(record: InterpretationRecord | None) -> str:
    if record is None:
        return "No interpretation exists for that capture."
    parts = [
        f"Interpretation for #{record.capture_id}",
        f"Summary: {record.summary}",
    ]
    if record.insight:
        parts.append(f"Insight: {record.insight}")
    if record.why_it_matters:
        parts.append(f"Why it matters: {record.why_it_matters}")
    if record.possible_application:
        parts.append(f"Possible application: {record.possible_application}")
    if record.counterpoint:
        parts.append(f"Counterpoint: {record.counterpoint}")
    parts.append(f"Confidence: {record.confidence:.2f} ({record.method})")
    return "\n".join(parts)


def render_applications(capture_id: int, applications: list[ApplicationRecord]) -> str:
    if not applications:
        return f"No application proposals for #{capture_id}."
    lines = [f"Application proposals for #{capture_id}:"]
    for idx, app in enumerate(applications, start=1):
        lines.append(f"{idx}. {app.title}: {app.body}")
    return "\n".join(lines)


def render_review(counts: dict[str, int], recent: list[CaptureRecord], unapplied: list[CaptureRecord]) -> str:
    count_text = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"
    lines = ["Wisdom review", f"By category: {count_text}"]
    lines.append("Recent captures:")
    if recent:
        lines.extend(f"#{c.id} - {c.category} - {c.title}" for c in recent)
    else:
        lines.append("none")
    lines.append("Unapplied candidates:")
    if unapplied:
        lines.extend(f"#{c.id} - {c.category} - {c.title}" for c in unapplied)
    else:
        lines.append("none")
    return "\n".join(lines)


def render_review_payload(payload: dict) -> str:
    counts = payload.get("counts") or {}
    items = payload.get("items") or []
    mode = str(payload.get("mode") or "needs_review").replace("_", "-")
    category = payload.get("category")
    lines = [
        "Wisdom Review",
        f"Mode: {mode}" + (f" / {category}" if category else ""),
        (
            f"Needs review: {counts.get('needs_review', 0)} | "
            f"High-potential: {counts.get('high_potential', 0)} | "
            f"Unapplied: {counts.get('unapplied', 0)}"
        ),
    ]
    if not items:
        lines.append("No review candidates found.")
        return "\n".join(lines)
    for index, item in enumerate(items, start=1):
        capture = item.get("capture") or {}
        quality = item.get("quality") or {}
        related = item.get("related") or []
        related_ids = ", ".join(f"#{rel.get('capture', {}).get('id')}" for rel in related if rel.get("capture"))
        lines.extend(
            [
                f"{index}. #{capture.get('id')} - {str(capture.get('category', '')).title()} - {capture.get('review_status')}",
                f"   \"{capture.get('original_excerpt', '')}\"",
                (
                    f"   Quality: {quality.get('label')} "
                    f"(overall {quality.get('overall')}, actionability {quality.get('actionability')}, "
                    f"reusability {quality.get('reusability')})"
                ),
                f"   Why it may matter: {_join_reasons(quality.get('reasons') or [])}",
                f"   Suggested action: {item.get('suggested_action')}",
            ]
        )
        if related_ids:
            lines.append(f"   Related: {related_ids}")
    return "\n".join(lines)


def render_related_payload(payload: dict) -> str:
    capture_id = payload.get("capture_id")
    related = payload.get("related") or []
    if not related:
        return f"Related captures for #{capture_id}\nNo related captures found."
    lines = [f"Related captures for #{capture_id}"]
    for item in related:
        capture = item.get("capture") or {}
        reasons = item.get("reasons") or []
        lines.append(
            f"#{capture.get('id')} - {capture.get('category')} - {capture.get('title')}\n"
            f"  {_excerpt(capture.get('original_excerpt', ''))}\n"
            f"  Why related: {_join_reasons(reasons)}"
        )
    return "\n".join(lines)


def render_review_action(action: str, capture: CaptureRecord | None, capture_id: int) -> str:
    if capture is None:
        return render_not_found(capture_id)
    return f"{action.title()} #{capture.id}. Review status: {capture.review_status}."


def render_not_found(capture_id: int) -> str:
    return f"Capture #{capture_id} was not found."


def render_error() -> str:
    return "Wisdom command failed. Normal Hermes is still available."


def _date(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _excerpt(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _join_reasons(reasons: list[str]) -> str:
    return "; ".join(str(reason) for reason in reasons[:3]) if reasons else "specific enough to review"
