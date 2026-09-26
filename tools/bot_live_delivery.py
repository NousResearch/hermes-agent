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
from typing import Any, Callable

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
    created = not root.is_dir()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    if created:
        # Only a fresh mailbox dir needs its parents durably linked; the live
        # poller re-enters this lock twice a second per profile, and two
        # directory fsyncs per idle poll was measurable disk churn for nothing.
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


def _ticket_shape_error(path: Path, record: dict[str, Any]) -> str | None:
    """Why a parsed ticket is unusable by the scans, or None when it is well-formed.

    A ticket that parses as JSON but lost a field (truncated rewrite, foreign
    writer, hand edit) used to raise KeyError/TypeError out of the sequence
    scan and the claim sweep — wedging admission and delivery for the whole
    profile exactly like corrupt JSON did before ``_scan_read`` existed.
    """
    owner = record.get("owner")
    created_at, sequence = record.get("created_at"), record.get("sequence", record.get("created_at"))
    if record.get("delivery_id") != path.stem or record.get("id") != path.stem:
        return "id does not match filename"
    status = record.get("status")
    if not isinstance(status, str) or status not in ({"queued", "claimed"} | _TERMINAL):
        return f"unknown status {status!r}"
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (created_at, sequence)):
        return "created_at/sequence are not integers"
    if not isinstance(owner, dict) or not all(isinstance(owner.get(k), str) and owner[k] for k in _OWNER_KEYS):
        return "owner pin is incomplete"
    return None


def _scan_read(path: Path) -> dict[str, Any] | None:
    """Bulk-scan variant: one unreadable or malformed ticket must not wedge the whole dir.

    Directory scans (sequence high-water mark, queued-claim sweep) may only
    treat a file as absent when it is provably absent; an unreadable ticket
    degrades to "that one delivery is uninspectable" with a warning.
    Exact-id reads (admission idempotency, completion, result lookup) keep
    using _read so a permission error still fails closed instead of
    licensing an overwrite of a possibly-live receipt.
    """
    try:
        record = _read(path)
        problem = None if record is None else _ticket_shape_error(path, record)
    except (OSError, ValueError) as exc:  # ValueError: corrupt JSON and invalid UTF-8 alike
        record, problem = None, str(exc)
    if problem is not None:
        level = logging.DEBUG if path in _warned_unreadable else logging.WARNING
        _warned_unreadable.add(path)
        log.log(level, "bot_live_delivery: skipping unreadable ticket %s (%s)", path.name, problem)
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


OWNER_GONE_REASON = "owner_gone"


def live_lease_ids(profile_home: Path | str) -> set[str]:
    """Lease ids the profile's active-session registry still holds (dead owners pruned).

    Raises when the registry is unreadable: callers must not treat "can't tell" as "gone".
    """
    from hermes_cli.active_sessions import active_session_registry_snapshot

    return {str(entry.get("lease_id") or "") for entry in
            active_session_registry_snapshot(registry_home=Path(profile_home).resolve())} - {""}


def _same_chat_lineage(home: Path | str, pinned_session: str, owner_session: str) -> bool:
    if pinned_session == owner_session:
        return True
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(home) / "state.db", read_only=True)
    try:
        return db.get_compression_tip(pinned_session) == owner_session
    finally:
        db.close()


def queued_pins(profile_home: Path | str, *, only_id: str | None = None) -> dict[str, str]:
    """``{delivery_id: pinned lease_id}`` for queued envelopes — read BEFORE the liveness snapshot so
    ``reconcile_orphaned_deliveries`` can fence each mutation on the pin being unchanged."""
    if not _root(profile_home).is_dir():
        return {}
    with _locked(profile_home) as root:
        paths = [root / f"{_delivery_id(only_id)}.json"] if only_id else list(root.glob("*.json"))
        return {record["delivery_id"]: record["owner"]["lease_id"] for path in paths
                if (record := _scan_read(path)) is not None and record["status"] == "queued"}


def reconcile_orphaned_deliveries(
    profile_home: Path | str, owner: dict[str, Any] | None, *,
    live_leases: set[str], expected_pins: dict[str, str], adopt_max_age_seconds: float,
    now_ns: int | None = None, only_id: str | None = None,
) -> dict[str, list[str]]:
    """Settle QUEUED envelopes whose pinned owner lease no longer exists.

    A live owner (Desktop Bot Chat) that closes or restarts leaves every envelope it had not yet
    claimed pinned to a dead lease: nobody can ever claim it, so the sender only ever sees
    'pending or unknown' and the recipient's transcript never gets the row. Queued means the
    turn never started, so each orphan is safe to move exactly once, under the mailbox lock:

    * ``owner`` given (a live consumer on the same Bot Chat compression lineage) and the envelope
      is at most ``adopt_max_age_seconds`` old → re-pinned to ``owner``; it runs in FIFO order.
    * otherwise → terminal ``cancelled`` receipt with reason ``owner_gone`` (never executed), which
      the sender's runner reads as "take the CLI transport" or reports as not delivered.

    ``expected_pins`` (from ``queued_pins``, read BEFORE ``live_leases``/``owner``) fences every mutation:
    an envelope is moved only if it is still pinned to the lease observed then, so an owner that registered
    and adopted it after the liveness snapshot is never overridden (the caller retries with fresh evidence).
    Claimed envelopes are never touched (their turn may have run). ``only_id`` limits the pass to one
    envelope (a sender's runner settling its own delivery). Returns ``{"adopted": [...], "retired": [...]}``.
    """
    result: dict[str, list[str]] = {"adopted": [], "retired": []}
    if not _root(profile_home).is_dir():
        return result
    current = _owner(profile_home, owner) if owner is not None else None
    now = time.time_ns() if now_ns is None else now_ns
    max_age_ns = int(max(0.0, float(adopt_max_age_seconds)) * 1e9)
    with _locked(profile_home) as root:
        paths = [root / f"{_delivery_id(only_id)}.json"] if only_id else list(root.glob("*.json"))
        for path in paths:
            record = _scan_read(path)
            if record is None or record["status"] != "queued":
                continue
            pinned = record["owner"]
            if expected_pins.get(record["delivery_id"]) != pinned["lease_id"]:
                continue  # re-pinned (or admitted) after the caller's liveness snapshot
            if pinned["lease_id"] in live_leases or (current is not None and pinned["lease_id"] == current["lease_id"]):
                continue
            if (current is not None and now - record["created_at"] <= max_age_ns
                    and pinned["lease_id"] != current["lease_id"]
                    and _same_chat_lineage(profile_home, pinned["session_id"], current["session_id"])):
                record.setdefault("repinned_from", []).append(pinned)
                record.update(owner=dict(current), **current, repinned_at=now)
                _write(path, record)
                result["adopted"].append(record["delivery_id"])
                continue
            record.update(status="cancelled", reply="", reason=OWNER_GONE_REASON, completed_at=now,
                          error=("The live Bot Chat owner this message was queued for closed before it ran; "
                                 "it was NOT delivered through the live session."))
            _write(path, record)
            result["retired"].append(record["delivery_id"])
    return result


_ADOPT_EVERY_SECONDS = 60.0


def adopt_orphaned_deliveries(profile_home: Path | str, owner: dict[str, Any], *, state: dict) -> None:
    """Live consumer hook: at most once per ``_ADOPT_EVERY_SECONDS`` (tracked in ``state``), re-pin to
    ``owner`` the queued envelopes a closed previous owner of this Bot Chat left behind, or retire them.
    Never raises: a failed pass must not block the caller's own claim; the next pass retries."""
    now = time.monotonic()
    if now - state.get("_bot_orphan_reconciled_at", -_ADOPT_EVERY_SECONDS) < _ADOPT_EVERY_SECONDS:
        return
    state["_bot_orphan_reconciled_at"] = now
    try:
        from tools.bot_relay import dm_queue_wait_seconds

        pins = queued_pins(profile_home)
        if not pins:
            return
        moved = reconcile_orphaned_deliveries(profile_home, owner, live_leases=live_lease_ids(profile_home),
                                              expected_pins=pins,
                                              adopt_max_age_seconds=dm_queue_wait_seconds(profile_home))
    except Exception:
        log.debug("bot_live_delivery: orphan reconcile failed", exc_info=True)
        return
    if moved["adopted"] or moved["retired"]:
        log.info("bot_live_delivery: adopted %d and retired %d delivery(ies) orphaned by a closed owner",
                 len(moved["adopted"]), len(moved["retired"]))


def read_delivery_result(profile_home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """Read admission/claim/terminal state without waiting or deleting its receipt."""
    return _read(_root(profile_home) / f"{_delivery_id(delivery_id)}.json")


_PENDING = ("queued", "claimed")
_POLL_SECONDS = 0.5


def await_delivery(
    profile_home: Path | str, delivery_id: str, timeout: float | None,
    *, should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """Poll a receipt until the owner settles it, ``timeout`` lapses, or ``should_stop`` says so.

    Every transport that hands a turn to a live Bot Chat owner (local ``message_agent``, the
    Desktop relay, ``hermes peer dm`` and ``hermes peer run``) waits on the same receipt; keeping
    the loop here is what stops the lanes drifting (one lane returned a receipt sentence instead
    of the reply, two never waited at all). Returns the last record read — still pending when the
    budget lapsed, None when the receipt was never readable.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        record = read_delivery_result(profile_home, delivery_id)
        if record is None or record["status"] not in _PENDING:
            return record
        if should_stop is not None and should_stop():
            return record
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return record
        time.sleep(_POLL_SECONDS if remaining is None else min(_POLL_SECONDS, remaining))


async def await_delivery_async(
    profile_home: Path | str, delivery_id: str, timeout: float | None,
    *, should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    """``await_delivery`` for an event loop: never blocks a worker thread for the whole budget."""
    import asyncio

    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        record = await asyncio.to_thread(read_delivery_result, profile_home, delivery_id)
        if record is None or record["status"] not in _PENDING:
            return record
        if should_stop is not None and should_stop():
            return record
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return record
        await asyncio.sleep(_POLL_SECONDS if remaining is None else min(_POLL_SECONDS, remaining))
