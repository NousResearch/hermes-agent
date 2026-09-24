"""Canonical Bot Chat transport and durable legacy delivery receipts.

Only gateway/session_bot.py admits execution. Old claimed/terminal records remain
inspectable; the legacy UI claim consumer is retired. No authority means refusal.
Receipt storage derives from FalconOrtiz's live-owner mailbox (#101564).
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


def _bot_chat_tip(home: Path) -> str | None:
    """Current Bot Chat tip id, or None when the profile has no store or no Bot Chat yet."""
    from hermes_state import SessionDB

    if not (home / 'state.db').is_file():
        return None
    db = SessionDB(db_path=home / 'state.db', read_only=True)
    try:
        row = db.get_session_by_title('Bot Chat')
        return db.get_compression_tip(row['id']) if row else None
    finally:
        db.close()


def find_canonical_live_owner(profile_home: Path | str) -> dict[str, Any] | None:
    """Discover the profile authority and its exact Bot Chat without acquiring a lease.

    ``session_id`` is the current Bot Chat tip, or ``''`` when the profile has none yet: the
    authority's deliver door creates it on first delivery (main's ``--create-if-missing``),
    so a missing chat is not a refusal. None only when the profile has no store at all."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    home = Path(profile_home).resolve()
    discovery = discover_gateway_endpoint(home, timeout=5)
    if discovery.state != 'ready' or discovery.endpoint is None:
        raise ValueError('profile authority is not ready')
    if not (home / 'state.db').is_file():
        return None
    tip = _bot_chat_tip(home)
    return dict(profile_home=str(home), session_id=tip or '', canonical=True,
                lease_id=discovery.endpoint.instance_id, live_session_id=tip or '')


def authority_delivery(home, params):
    """Call only this home's already-running authority; never start a fallback."""
    import asyncio
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_client import GatewayClient, _session_ticket
    from websockets.asyncio.client import connect

    home = Path(home).resolve()

    async def request():
        discovery = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=5)
        if discovery.state != 'ready' or discovery.endpoint is None:
            raise ValueError('profile authority is not ready')
        endpoint = discovery.endpoint
        ticket = await asyncio.to_thread(_session_ticket, home, endpoint)
        url = endpoint.api_origin.replace('http:', 'ws:').replace('https:', 'wss:') + '/api/ws'
        async with connect(url, subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket],
                           open_timeout=10) as ws:
            if ws.subprotocol != 'hermes-gateway-v1':
                raise ValueError('authority protocol mismatch')
            async with GatewayClient(ws) as client:
                return await client.rpc('bot_relay.deliver', **params)

    return asyncio.run(request())


def _owner(home: Path | str, owner: dict[str, Any]) -> dict[str, str]:
    pinned = {key: owner.get(key) for key in _OWNER_KEYS}
    # An empty session_id is the discovered "no Bot Chat yet" state; the door creates it.
    if not all(isinstance(value, str) and (value or key in ("session_id", "live_session_id"))
               for key, value in pinned.items()):
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
        record = json.loads(path.read_text(encoding="utf-8"))
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
        persisted = int(counter.read_text(encoding="utf-8"))
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
    home = Path(profile_home).resolve()
    return authority_delivery(home, dict(id=_delivery_id(delivery_id if delivery_id is not None else uuid.uuid4().hex),
        profile=home.name if home.parent.name == "profiles" else "default",
        message=message, **({"session_id": pinned["session_id"]} if pinned["session_id"] else {}),
        **({"author": dict(author)} if author else {})))


def claim_pending_delivery(profile_home, owner):
    """Retired UI poller: only session_bot may migrate and admit queued records.

    Keep this refusal until the legacy notification poller's call site is removed.
    A previously claimed record can still publish its terminal receipt below.
    """
    return None


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
    record = _read(_root(profile_home) / f"{_delivery_id(delivery_id)}.json")
    if record is not None and record.get('admission_id'):
        home = Path(profile_home).resolve()
        # The authority compares the stored author to the retry payload; omitting it is a conflict.
        return authority_delivery(home, dict(id=delivery_id,
            profile=home.name if home.parent.name == 'profiles' else 'default', message=record['message'],
            **({'author': dict(record['author'])} if record.get('author') else {})))
    return record


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
