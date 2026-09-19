"""Durable, at-most-once handoff to an existing Bot Chat owner.

Adapted from FalconOrtiz's live-owner mailbox (#101564). A single private
record advances queued -> claimed -> terminal under a process-shared lock.
Claims never expire: a crashed consumer leaves an inspectable unknown outcome,
not permission to execute the same input again. Receipts are permanent.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager

from utils import atomic_json_write, atomic_write_text, fsync_directory
from pathlib import Path
from typing import Any

from hermes_cli.active_sessions import _FileLock

log = logging.getLogger(__name__)

DELIVERY_DIR_NAME = "bot_live_delivery"
_SEQUENCE_FILE = ".sequence"
_OWNER_KEYS = ("profile_home", "session_id", "lease_id", "live_session_id")
_TERMINAL = frozenset({"settled", "failed", "cancelled", "ambiguous"})


def find_canonical_owner(profile_home: Path | str) -> dict[str, Any] | None:
    """Return the exact Bot Chat tip's lease, including unsupported CLI owners."""
    from hermes_cli.active_sessions import active_session_registry_snapshot
    from hermes_state import SessionDB

    home = Path(profile_home).resolve()
    if not (home / "state.db").is_file():
        return None
    db = SessionDB(db_path=home / "state.db", read_only=True)
    try:
        row = db.get_session_by_title("Bot Chat")
        session_id = db.get_compression_tip(row["id"]) if row else None
    finally:
        db.close()
    if not session_id:
        return None
    for entry in active_session_registry_snapshot(registry_home=home):
        if entry["session_id"] == session_id:
            return {**entry, "profile_home": str(home)}
    return None


def find_canonical_live_owner(profile_home: Path | str) -> dict[str, Any] | None:
    """Only advertised consumers may receive owner-pinned mailbox deliveries."""
    entry = find_canonical_owner(profile_home)
    meta = (entry or {}).get("metadata") or {}
    if entry and meta.get("bot_live_delivery_consumer") is True and meta.get("live_session_id"):
        return {key: entry[key] for key in ("profile_home", "session_id", "lease_id")} | {
            "live_session_id": meta["live_session_id"]}
    return None


def _owner(home: Path | str, owner: dict[str, Any]) -> dict[str, str]:
    pinned = {key: owner.get(key) for key in _OWNER_KEYS}
    if not all(isinstance(value, str) and value for value in pinned.values()):
        raise ValueError("owner requires profile_home, session_id, lease_id and live_session_id")
    if pinned["profile_home"] != str(Path(home).resolve()):
        raise ValueError("owner belongs to a different profile home")
    return pinned


def _delivery_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32,64}", value) is None:
        raise ValueError("delivery id must be 32 to 64 lowercase hex characters")
    return value


def _root(home: Path | str) -> Path:
    return Path(home).resolve() / "runtime" / DELIVERY_DIR_NAME


def has_mailbox(profile_home: Path | str) -> bool:
    """Whether any delivery was ever admitted for this profile (the mailbox directory is created on
    first admission only). A cheap pre-check for pollers: no mailbox means nothing to claim, so the
    owner lookup — a state.db open plus the exclusive active-session registry lock — can be skipped."""
    return _root(profile_home).is_dir()


@contextmanager
def _locked(home: Path | str):
    root = _root(home)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    fsync_directory(root.parent)
    fsync_directory(root.parent.parent)
    lock = root / ".lock"
    fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    with _FileLock(lock):
        yield root


def _read(path: Path) -> dict[str, Any] | None:
    """Exact-id read: absent → None; unreadable or not a JSON object → raises (callers fail closed)."""
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"ticket {path.name} is not a JSON object ({type(record).__name__})")
    return record


# Tickets already reported unreadable by this process. The live poller rescans the
# dir twice a second, so a persistent bad ticket is WARNING once and DEBUG after.
_warned_unreadable: set[Path] = set()


def _scan_read(path: Path) -> dict[str, Any] | None:
    """Bulk-scan variant: one unreadable ticket must not wedge the whole dir.

    Directory scans (sequence high-water mark, queued-claim sweep) may only
    treat a file as absent when it is provably absent; an unreadable ticket
    degrades to "that one delivery is uninspectable" with a warning.
    Exact-id reads (admission idempotency, completion, result lookup) keep
    using _read so a permission error still fails closed instead of
    licensing an overwrite of a possibly-live receipt.
    """
    try:
        record = _read(path)
    except (OSError, ValueError) as exc:  # ValueError: corrupt JSON and invalid UTF-8 alike
        level = logging.DEBUG if path in _warned_unreadable else logging.WARNING
        _warned_unreadable.add(path)
        log.log(level, "bot_live_delivery: skipping unreadable ticket %s (%s)", path.name, exc)
        return None
    _warned_unreadable.discard(path)
    return record


def _next_sequence(root: Path) -> int:
    """Allocate the next admission sequence under the dir lock.

    The high-water mark lives in a counter file beside the tickets, so a ticket
    the scan cannot read does not drop its sequence and hand a later admission
    a duplicate or lower one. Readable tickets still bootstrap dirs written
    before the counter existed. Wall time can roll back; sequences never do.
    """
    counter = root / _SEQUENCE_FILE
    try:
        persisted = int(counter.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        persisted = 0
    scanned = max((record.get("sequence", record["created_at"])
                   for candidate in root.glob("*.json")
                   if (record := _scan_read(candidate)) is not None), default=0)
    sequence = max(persisted, scanned) + 1
    atomic_write_text(counter, str(sequence), mode=0o600, fsync_dir=True)
    return sequence


def _write(path: Path, record: dict[str, Any]) -> None:
    atomic_json_write(path, record, indent=None, sort_keys=True, fsync_dir=True, mode=0o600)


def deliver_to_live_owner(
    profile_home: Path | str, owner: dict[str, Any], message: str,
    *, delivery_id: str | None = None, author: dict[str, Any] | None = None,
    notification_category: str = "result",
) -> dict[str, Any]:
    """Return durable admission immediately, without waiting for the owner.

    Retry with the same id AND pinned owner/message to inspect the existing
    state. Reusing an id with a different payload is an error, never an overwrite.
    """
    pinned = _owner(profile_home, owner)
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    key = _delivery_id(delivery_id if delivery_id is not None else uuid.uuid4().hex)
    with _locked(profile_home) as root:
        path = root / f"{key}.json"
        existing = _read(path)
        if existing is not None:
            if (existing["owner"] != pinned or existing["message"] != message or existing.get("author") != author
                    or existing.get("notification_category", "result") != notification_category):
                raise ValueError("delivery id already belongs to a different payload")
            return existing
        record = dict(delivery_id=key, id=key, owner=pinned, **pinned,
                      message=message, status="queued", created_at=time.time_ns(),
                      sequence=_next_sequence(root), **({"author": dict(author)} if author else {}))
        if notification_category == "diagnostic":
            record["notification_category"] = notification_category
        _write(path, record)
        return record


def _matches(home: Path | str, record: dict, owner: dict) -> bool:
    pinned = record["owner"]
    if any(pinned[key] != owner[key] for key in ("profile_home", "lease_id", "live_session_id")):
        return False
    if pinned["session_id"] == owner["session_id"]:
        return True
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(home) / "state.db", read_only=True)
    try:
        return db.get_compression_tip(pinned["session_id"]) == owner["session_id"]
    finally:
        db.close()


def claim_pending_delivery(
    profile_home: Path | str, owner: dict[str, Any],
) -> dict[str, Any] | None:
    """Claim oldest matching input exactly once; caller supplies its current lease.

    A lease transfer across compression is accepted only along the original
    stored session's compression chain. A new lease/live session cannot steal it.
    Caller must hold its normal turn-admission guard before invoking this.
    """
    current = _owner(profile_home, owner)
    if not _root(profile_home).is_dir():
        return None
    with _locked(profile_home) as root:
        pending = []
        for path in root.glob("*.json"):
            record = _scan_read(path)
            if record is not None and record["status"] == "queued" and _matches(profile_home, record, current):
                pending.append(record)
        if not pending:
            return None
        record = min(pending, key=lambda item: (
            item.get("sequence", item["created_at"]), item["delivery_id"]))
        record.update(status="claimed", claimed_at=time.time_ns())
        _write(root / f"{record['delivery_id']}.json", record)
        return record


def complete_delivery(
    profile_home: Path | str, delivery_id: str, *, status: str,
    reply: str = "", error: str = "", reason: str = "",
) -> dict[str, Any]:
    """Persist an immutable terminal receipt; duplicate identical completion is safe."""
    key = _delivery_id(delivery_id)
    if status not in _TERMINAL:
        raise ValueError("invalid terminal delivery status")
    outcome = dict(status=status, reply=reply, error=error, reason=reason)
    with _locked(profile_home) as root:
        path = root / f"{key}.json"
        record = _read(path)
        if record is None:
            raise FileNotFoundError(f"delivery not found: {key}")
        if record["status"] in _TERMINAL:
            if any(record.get(k) != v for k, v in outcome.items()):
                raise ValueError("delivery already has a different terminal receipt")
            return record
        if record["status"] != "claimed":
            raise ValueError("delivery must be claimed before completion")
        record.update(outcome, completed_at=time.time_ns())
        _write(path, record)
        return record


def read_delivery_result(profile_home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """Read admission/claim/terminal state without waiting or deleting its receipt."""
    return _read(_root(profile_home) / f"{_delivery_id(delivery_id)}.json")
