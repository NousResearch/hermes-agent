"""Append-only skill change ledger utilities.

This module stores local skill-change events as JSON Lines under
``get_hermes_home() / "skill_changes.jsonl"`` and optional diff artifacts under
``get_hermes_home() / "skill-history" / "events"``.  It intentionally has no
registry side effects so it can be imported safely by tools, API handlers, and
unit tests.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hermes_constants import get_hermes_home

REASON_KINDS = {"explicit", "system", "unattributed", "model_summary"}
REVIEW_STATUSES = {"unreviewed", "reviewed", "needs_followup"}
MAX_DIFF_TEXT_CHARS = 50_000
_DIFF_TRUNCATION_NOTICE = "\n\n[diff truncated to 50,000 characters for dashboard display]"
EVENT_FIELDS = (
    "event_id",
    "timestamp",
    "skill",
    "category",
    "action",
    "actor",
    "source",
    "session_id",
    "reason",
    "reason_kind",
    "before_hash",
    "after_hash",
    "changed_files",
    "diff_path",
    "metadata",
    "review_status",
    "reviewed_at",
    "review_note",
)


def _default_ledger_path() -> Path:
    return get_hermes_home() / "skill_changes.jsonl"


def _default_artifacts_root() -> Path:
    return get_hermes_home() / "skill-history" / "events"


def _resolve_path(path: str | os.PathLike[str] | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _safe_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-_")
    return slug or "unknown"


def _new_event_id(skill: str, action: str) -> str:
    stamp = _now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{_safe_component(skill)}_{_safe_component(action)}_{uuid.uuid4().hex[:8]}"


def _validate_reason_kind(reason: str | None, reason_kind: str | None) -> str:
    if reason_kind is None:
        return "explicit" if reason else "unattributed"
    if reason_kind not in REASON_KINDS:
        raise ValueError(
            f"Invalid reason_kind {reason_kind!r}; expected one of {sorted(REASON_KINDS)}"
        )
    if not reason and reason_kind != "unattributed":
        # Caller explicitly supplied provenance even without a text reason; keep it.
        return reason_kind
    return reason_kind


def _validate_review_status(status: str) -> str:
    if status not in REVIEW_STATUSES:
        raise ValueError(
            f"Invalid review status {status!r}; expected one of {sorted(REVIEW_STATUSES)}"
        )
    return status


def hash_skill_dir(skill_dir: Path) -> str | None:
    """Return a deterministic sha256 digest for regular files in a skill dir.

    The digest is based on sorted POSIX relative file paths and raw file bytes.
    ``None`` is returned when ``skill_dir`` does not exist or is not a directory.
    """

    root = Path(skill_dir)
    if not root.is_dir():
        return None

    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )

    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(b"path\0")
        digest.update(rel)
        digest.update(b"\0size\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0content\0")
        digest.update(data)
        digest.update(b"\0")

    return f"sha256:{digest.hexdigest()}"


def compute_text_diff(before: dict[str, str], after: dict[str, str]) -> str:
    """Return a unified diff for text snapshots keyed by relative file path."""

    chunks: list[str] = []
    for file_name in sorted(set(before) | set(after)):
        before_lines = before.get(file_name, "").splitlines()
        after_lines = after.get(file_name, "").splitlines()
        diff_lines = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{file_name}",
                tofile=f"b/{file_name}",
                lineterm="",
            )
        )
        if diff_lines:
            chunks.append("\n".join(diff_lines) + "\n")
    return "".join(chunks)


def _append_event(
    event: Mapping[str, Any],
    *,
    ledger_path: str | os.PathLike[str] | None = None,
) -> None:
    path = _resolve_path(ledger_path, _default_ledger_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def _read_records(
    *,
    ledger_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    path = _resolve_path(ledger_path, _default_ledger_path())
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except FileNotFoundError:
        return []
    return records


def _latest_events(
    *,
    ledger_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    creation_order: list[str] = []
    for record in _read_records(ledger_path=ledger_path):
        event_id = record.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            continue
        if event_id not in by_id:
            creation_order.append(event_id)
        by_id[event_id] = record
    return [dict(by_id[event_id]) for event_id in creation_order]


def _find_latest_event(
    event_id: str,
    *,
    ledger_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    for event in _latest_events(ledger_path=ledger_path):
        if event.get("event_id") == event_id:
            return event
    return None


def _derive_changed_files(
    changed_files: Sequence[str] | None,
    before_text: Mapping[str, str] | None,
    after_text: Mapping[str, str] | None,
) -> list[str]:
    if changed_files is not None:
        return list(changed_files)
    if before_text is not None or after_text is not None:
        return sorted(set(before_text or {}) | set(after_text or {}))
    return []


def _build_diff_text(
    *,
    diff_text: str | None,
    before_text: Mapping[str, str] | None,
    after_text: Mapping[str, str] | None,
) -> str | None:
    if diff_text is not None:
        return diff_text
    if before_text is None and after_text is None:
        return None
    return compute_text_diff(dict(before_text or {}), dict(after_text or {}))


def record_skill_change(
    skill: str,
    action: str,
    actor: str = "hermes-agent",
    source: str = "unknown",
    *,
    category: str | None = None,
    session_id: str | None = None,
    reason: str | None = None,
    reason_kind: str | None = None,
    before_hash: str | None = None,
    after_hash: str | None = None,
    changed_files: Sequence[str] | None = None,
    before_text: Mapping[str, str] | None = None,
    after_text: Mapping[str, str] | None = None,
    diff_text: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    ledger_path: str | os.PathLike[str] | None = None,
    artifacts_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Append a skill change event and optional diff artifact.

    Missing reasons are accepted and recorded as ``reason_kind='unattributed'``.
    If ``diff_text`` is supplied, or if ``before_text``/``after_text`` produce a
    non-empty unified diff, the diff is written under the event artifact root and
    ``diff_path`` is set on the event.
    """

    event_id = _new_event_id(skill, action)
    event_diff = _build_diff_text(
        diff_text=diff_text,
        before_text=before_text,
        after_text=after_text,
    )

    diff_path: str | None = None
    if event_diff:
        root = _resolve_path(artifacts_root, _default_artifacts_root())
        artifact_path = root / event_id / "diff.patch"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(event_diff, encoding="utf-8")
        diff_path = str(artifact_path)

    event = {
        "event_id": event_id,
        "timestamp": _now_iso(),
        "skill": skill,
        "category": category,
        "action": action,
        "actor": actor,
        "source": source,
        "session_id": session_id,
        "reason": reason,
        "reason_kind": _validate_reason_kind(reason, reason_kind),
        "before_hash": before_hash,
        "after_hash": after_hash,
        "changed_files": _derive_changed_files(changed_files, before_text, after_text),
        "diff_path": diff_path,
        "metadata": dict(metadata or {}),
        "review_status": "unreviewed",
        "reviewed_at": None,
        "review_note": None,
    }

    # Preserve the schema field set/order and avoid accidental extra fields.
    event = {field: event[field] for field in EVENT_FIELDS}
    _append_event(event, ledger_path=ledger_path)
    return event


def list_skill_changes(
    skill: str | None = None,
    limit: int = 50,
    unreviewed: bool | None = None,
    *,
    ledger_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Return recent skill change events, newest first.

    Review updates are append-only records with the same ``event_id``; listing
    collapses them to the latest state while preserving original event recency.
    """

    events = _latest_events(ledger_path=ledger_path)
    filtered: list[dict[str, Any]] = []
    for event in events:
        if skill is not None and event.get("skill") != skill:
            continue
        if unreviewed is True and event.get("review_status") != "unreviewed":
            continue
        if unreviewed is False and event.get("review_status") == "unreviewed":
            continue
        filtered.append(event)

    recent = list(reversed(filtered))
    if limit is None:  # type: ignore[unreachable]
        return recent
    return recent[: max(0, limit)]


def _safe_diff_artifact_path(diff_path: str) -> Path | None:
    """Resolve a diff path only if it stays under the expected artifact root."""
    try:
        path = Path(os.path.expanduser(diff_path)).resolve()
        root = _default_artifacts_root().resolve()
        if not path.is_relative_to(root):
            return None
        return path
    except OSError:
        return None


def _bounded_diff_text(text: str) -> str:
    """Bound diff text returned through APIs while preserving the raw artifact."""
    if len(text) <= MAX_DIFF_TEXT_CHARS:
        return text
    return text[:MAX_DIFF_TEXT_CHARS] + _DIFF_TRUNCATION_NOTICE


def get_skill_change(
    event_id: str,
    *,
    ledger_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Return the latest state for one event, including diff text when present."""

    event = _find_latest_event(event_id, ledger_path=ledger_path)
    if event is None:
        return None

    detail = dict(event)
    diff_path = detail.get("diff_path")
    if isinstance(diff_path, str) and diff_path:
        path = _safe_diff_artifact_path(diff_path)
        try:
            if path is not None and path.exists():
                detail["diff_text"] = _bounded_diff_text(path.read_text(encoding="utf-8"))
        except OSError:
            pass
    return detail


def mark_skill_change_reviewed(
    event_id: str,
    status: str = "reviewed",
    note: str | None = None,
    *,
    ledger_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Append a review-state update for an existing event.

    The original ledger line is never rewritten; the returned event is the latest
    state after appending the review update.
    """

    _validate_review_status(status)
    event = _find_latest_event(event_id, ledger_path=ledger_path)
    if event is None:
        return None

    updated = dict(event)
    updated["review_status"] = status
    updated["reviewed_at"] = _now_iso()
    updated["review_note"] = note
    updated = {field: updated.get(field) for field in EVENT_FIELDS}
    _append_event(updated, ledger_path=ledger_path)
    return updated


__all__ = [
    "hash_skill_dir",
    "compute_text_diff",
    "record_skill_change",
    "list_skill_changes",
    "get_skill_change",
    "mark_skill_change_reviewed",
]
