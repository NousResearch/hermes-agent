"""Read-only view of one bot's unsettled ``message_agent`` deliveries to local teammates.

The receipts already exist: every local delivery the sender's ``message_agent`` admits lands in
the TARGET profile's ``runtime/bot_live_delivery/<id>.json``, stamped ``author.id ==
"bot:<sender>"`` (``tools/bot_mode_dm.py``). This module only reads them, so the sender's Bot
Chat can show what each teammate is doing with the request: still queued, working on it, parked
on a question only the user can answer, or failed. Nothing here writes, claims, cancels or resends
a delivery — an outcome this reader cannot see is reported as unknown, never retried.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

log = logging.getLogger(__name__)

# How long a failed delivery stays listed after its last receipt write. Long enough to come back
# to after lunch; Dismiss hides it sooner.
FAILED_VISIBLE_SECONDS = 24 * 3600
PREVIEW_CHARS = 140

_PENDING_RECEIPTS = frozenset({"queued", "claimed", "canonical"})
_FAILED_RECEIPTS = frozenset({"failed", "cancelled", "ambiguous"})
# ``session_admissions.status`` of the admission the receipt follows → strip state.
_ADMISSION_STATES = {"queued": "queued", "started": "working", "terminal": "claimed"}

# ``(admission_status | None, waiting_on | None)`` for one pending receipt; waiting_on is the kind
# of the target's open prompt (``"clarify"`` / ``"approval"``).
LiveState = Callable[[str, Path, dict], tuple[Optional[str], Optional[str]]]


def delivery_state(receipt_status: str, admission_status: str | None,
                   waiting_on: str | None) -> str | None:
    """The strip state of one receipt, or None when it is settled and drops off the strip.

    ``claimed`` covers the short windows where the target holds the delivery but its turn is not
    running on it: an admission that just finished while its receipt is still being written, or a
    failed attempt the owner is about to retry once."""
    if receipt_status in _FAILED_RECEIPTS:
        return "failed"
    if receipt_status not in _PENDING_RECEIPTS:
        return None
    if admission_status == "unknown":
        return "failed"
    if waiting_on and admission_status == "started":
        # The open prompt belongs to the turn running THIS delivery; one queued behind it in the
        # same chat is still just queued.
        return "waiting_on_you"
    if admission_status in _ADMISSION_STATES:
        return _ADMISSION_STATES[admission_status]
    return "claimed" if receipt_status == "claimed" else "queued"


def _failure_reason(record: dict, admission_status: str | None) -> str:
    """Typed reason for a failed row (``tools.bot_failure_reasons`` vocabulary where the owner
    wrote one). Unknown outcomes say so: the user must not resend those."""
    if record.get("status") == "ambiguous" or admission_status == "unknown":
        return "unknown_outcome"
    if record.get("status") == "cancelled":
        return str(record.get("reason") or "cancelled")
    return str(record.get("reason") or "unknown")


def _preview(message: Any) -> str:
    """The body the sender wrote, without the ``Message from 🤖 … (@x): `` signature."""
    text = message if isinstance(message, str) else ""
    head, sep, body = text.partition("): ")
    if sep and head.startswith("Message from "):
        text = body
    text = " ".join(text.split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


def _read_receipt(path: Path) -> dict | None:
    """Tolerant bulk read: an unreadable or foreign file is skipped, never fatal to the list."""
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        log.debug("bot_delivery_status: skipping unreadable receipt %s", path, exc_info=True)
        return None
    return record if isinstance(record, dict) else None


def stored_admission_status(home: Path, admission_id: str) -> str | None:
    """``session_admissions.status`` read from the target's own store, read-only. None when the
    profile has no store or the row is gone."""
    from hermes_state import SessionDB
    from hermes_state_runtime import get_session_admission

    if not admission_id or not (home / "state.db").is_file():
        return None
    db = SessionDB(db_path=home / "state.db", read_only=True)
    try:
        row = get_session_admission(db, admission_id=admission_id)
    finally:
        db.close()
    return row["status"] if row else None


def stored_live_state(_name: str, home: Path, record: dict) -> tuple[str | None, str | None]:
    """Default join when the target's authority is not in this process: the durable admission
    state only (an open prompt lives in the owner's memory, so it cannot be seen from here)."""
    return stored_admission_status(home, record.get("retry_admission_id") or record.get("admission_id") or ""), None


def _receipts(home: Path) -> Iterable[tuple[Path, dict]]:
    root = home / "runtime" / "bot_live_delivery"
    if not root.is_dir():
        return ()
    return ((path, record) for path in root.glob("*.json") if (record := _read_receipt(path)) is not None)


def list_sender_deliveries(
    root: Path, sender: str, *, live_state: LiveState = stored_live_state,
    now: float | None = None, profiles: Iterable[tuple[str, Path]] | None = None,
) -> list[dict]:
    """Every unsettled local delivery ``sender`` made, newest first.

    ``root`` is the install root (``bot_mode_probe._hermes_root``); ``profiles`` overrides the
    roster (``bot_mode_probe._roster``) for callers that already hold it. Settled replies drop
    off; failed ones stay for ``FAILED_VISIBLE_SECONDS``."""
    from tools.bot_mode_probe import _handle, _roster

    now = time.time() if now is None else now
    author_id = f"bot:{sender}"
    rows: list[dict] = []
    for name, home in (_roster(root) if profiles is None else profiles):
        if name == sender:
            continue
        for path, record in _receipts(Path(home)):
            author = record.get("author")
            if not isinstance(author, dict) or author.get("id") != author_id:
                continue
            status = str(record.get("status") or "")
            if status not in _PENDING_RECEIPTS | _FAILED_RECEIPTS:
                continue
            try:
                since = path.stat().st_mtime
            except OSError:
                continue
            if status in _FAILED_RECEIPTS:
                if now - since > FAILED_VISIBLE_SECONDS:
                    continue
                admission, waiting_on = None, None
            else:
                admission, waiting_on = live_state(name, Path(home), record)
            state = delivery_state(status, admission, waiting_on)
            if state is None:
                continue
            row = {"delivery_id": path.stem, "target": name, "target_handle": _handle(name),
                   "state": state, "since": since, "preview": _preview(record.get("message"))}
            if state == "waiting_on_you":
                row["waiting_on"] = waiting_on
            if state == "failed":
                row["reason"] = _failure_reason(record, admission)
                if record.get("error"):
                    row["error"] = str(record["error"])[:500]
            rows.append(row)
    rows.sort(key=lambda r: r["since"], reverse=True)
    return rows
