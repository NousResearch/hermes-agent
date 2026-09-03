"""Durable handoff of Bot Chat turns to an existing live owner.

A sender writes one request under the target profile's runtime directory.
The live TUI/Desktop owner atomically claims it and writes one terminal
receipt. The rename boundary distinguishes a request that was never
delivered from one whose transport outcome is no longer knowable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

DELIVERY_DIR_NAME = "bot_live_delivery"

# Surfaces that run tui_gateway's idle poller and can claim pending DMs.
_LIVE_OWNER_SURFACES = frozenset({"desktop", "tui"})


def _owner_can_consume(entry: dict[str, Any]) -> bool:
    meta = entry.get("metadata") or {}
    if meta.get("bot_live_delivery_consumer") is True:
        return True
    return str(entry.get("surface") or "").strip().lower() in _LIVE_OWNER_SURFACES


def find_canonical_live_owner(profile_home: Path | str) -> str | None:
    """Return the canonical Bot Chat id when a Desktop/TUI owner holds it."""
    from hermes_cli.active_sessions import active_session_registry_snapshot
    from hermes_state import SessionDB

    home = Path(profile_home)
    db = SessionDB(db_path=home / "state.db")
    try:
        row = db.get_session_by_title("Bot Chat")
    finally:
        db.close()
    session_id = str((row or {}).get("id") or "")
    if not session_id:
        return None
    try:
        owners = active_session_registry_snapshot(registry_home=home)
    except Exception:
        return None
    return session_id if any(
        str(entry.get("session_id") or "") == session_id and _owner_can_consume(entry)
        for entry in owners
    ) else None


def _session_dir(profile_home: Path | str, session_id: str) -> Path:
    key = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:32]
    return Path(profile_home) / "runtime" / DELIVERY_DIR_NAME / key


def _paths(profile_home: Path | str, session_id: str) -> tuple[Path, Path, Path]:
    base = _session_dir(profile_home, session_id)
    pending = base / "pending"
    claimed = base / "claimed"
    replies = base / "replies"
    expired = base / "expired"
    for directory in (pending, claimed, replies, expired):
        directory.mkdir(parents=True, exist_ok=True)
    return pending, claimed, replies


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def deliver_to_live_owner(
    profile_home: Path | str,
    session_id: str,
    message: str,
    *,
    owner_wait_seconds: float,
    receipt_wait_seconds: float,
    poll_seconds: float = 0.05,
) -> dict[str, Any]:
    """Submit one message and wait for the live owner's durable terminal receipt."""
    pending_dir, claimed_dir, replies_dir = _paths(profile_home, session_id)
    delivery_id = uuid.uuid4().hex
    pending = pending_dir / f"{delivery_id}.json"
    claimed = claimed_dir / pending.name
    reply_path = replies_dir / pending.name
    expired_path = pending_dir.parent / "expired" / pending.name
    created = time.time()
    owner_deadline = created + max(0.0, float(owner_wait_seconds))
    _atomic_json(
        pending,
        {
            "id": delivery_id,
            "session_id": str(session_id),
            "message": str(message),
            "created_at": created,
            "owner_deadline": owner_deadline,
        },
    )

    sleep_for = max(0.001, float(poll_seconds))
    while time.time() < owner_deadline:
        receipt = _read_json(reply_path)
        if receipt is not None:
            return _finish_result(delivery_id, receipt, claimed, reply_path)
        if claimed.exists():
            break
        time.sleep(sleep_for)
    else:
        # Winning this rename proves the owner never claimed the request. If it
        # loses, the owner crossed the claim boundary and only a receipt can say
        # whether the message entered the session.
        cancelled = pending.with_suffix(".cancelled")
        try:
            os.replace(pending, cancelled)
        except FileNotFoundError:
            if expired_path.exists():
                return {
                    "status": "not_delivered",
                    "reason": "target_busy",
                    "delivery_id": delivery_id,
                }
        else:
            cancelled.unlink(missing_ok=True)
            return {
                "status": "not_delivered",
                "reason": "target_busy",
                "delivery_id": delivery_id,
            }

    receipt_deadline = time.time() + max(0.0, float(receipt_wait_seconds))
    while time.time() < receipt_deadline:
        receipt = _read_json(reply_path)
        if receipt is not None:
            return _finish_result(delivery_id, receipt, claimed, reply_path)
        time.sleep(sleep_for)
    return {
        "status": "ambiguous",
        "reason": "delivery_timeout",
        "delivery_id": delivery_id,
    }


def _finish_result(
    delivery_id: str,
    receipt: dict[str, Any],
    claimed_path: Path,
    reply_path: Path,
) -> dict[str, Any]:
    claimed_path.unlink(missing_ok=True)
    reply_path.unlink(missing_ok=True)
    status = str(receipt.get("status") or "")
    if status == "settled":
        return {
            "status": "delivered",
            "delivery_id": delivery_id,
            "reply": str(receipt.get("reply") or ""),
        }
    return {
        "status": "not_delivered" if status in {"failed", "cancelled"} else "ambiguous",
        "reason": str(receipt.get("reason") or "unknown"),
        "delivery_id": delivery_id,
        "error": str(receipt.get("error") or ""),
    }


def claim_pending_delivery(
    profile_home: Path | str, session_id: str
) -> dict[str, Any] | None:
    """Atomically claim the oldest unexpired request for one live session."""
    pending_dir, claimed_dir, _replies_dir = _paths(profile_home, session_id)
    expired_dir = pending_dir.parent / "expired"
    now = time.time()
    candidates: list[tuple[float, str, Path, dict[str, Any]]] = []
    for pending in pending_dir.glob("*.json"):
        payload = _read_json(pending)
        if payload is None or payload.get("session_id") != str(session_id):
            pending.unlink(missing_ok=True)
            continue
        try:
            owner_deadline = float(payload.get("owner_deadline"))
            created_at = float(payload.get("created_at"))
        except (TypeError, ValueError):
            pending.unlink(missing_ok=True)
            continue
        delivery_id = str(payload.get("id") or "")
        if delivery_id != pending.stem:
            pending.unlink(missing_ok=True)
            continue
        if owner_deadline <= now:
            try:
                os.replace(pending, expired_dir / pending.name)
            except FileNotFoundError:
                pass
            continue
        candidates.append((created_at, delivery_id, pending, payload))

    for _created_at, _delivery_id, pending, payload in sorted(candidates):
        claimed = claimed_dir / pending.name
        try:
            os.replace(pending, claimed)
        except FileNotFoundError:
            continue
        return payload
    return None


def complete_delivery(
    profile_home: Path | str,
    delivery_id: str,
    *,
    status: str,
    reply: str = "",
    error: str = "",
    reason: str = "",
) -> None:
    """Persist the live owner's terminal result for the waiting sender."""
    safe_id = str(delivery_id or "")
    if len(safe_id) != 32 or any(ch not in "0123456789abcdef" for ch in safe_id):
        raise ValueError("invalid delivery id")
    # Delivery ids are globally random, so locate the receipt directory without
    # trusting caller-provided session/path data.
    root = Path(profile_home) / "runtime" / DELIVERY_DIR_NAME
    matches = list(root.glob(f"*/claimed/{safe_id}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"claimed delivery not found: {safe_id}")
    claimed = matches[0]
    replies = claimed.parent.parent / "replies"
    _atomic_json(
        replies / claimed.name,
        {
            "id": safe_id,
            "status": str(status),
            "reply": str(reply),
            "error": str(error),
            "reason": str(reason),
            "completed_at": time.time(),
        },
    )
