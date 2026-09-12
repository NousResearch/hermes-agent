"""Canonical Bot Chat transport and durable legacy delivery receipts.

Only gateway/session_bot.py admits execution. Old claimed/terminal records remain
inspectable; the legacy UI claim consumer is retired. No authority means refusal.
Receipt storage derives from FalconOrtiz's live-owner mailbox (#101564).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_cli.active_sessions import _FileLock

DELIVERY_DIR_NAME = "bot_live_delivery"
_OWNER_KEYS = ("profile_home", "session_id", "lease_id", "live_session_id")
_TERMINAL = frozenset({"settled", "failed", "cancelled", "ambiguous"})


def find_canonical_live_owner(profile_home: Path | str) -> dict[str, Any] | None:
    """Discover the profile authority and exact Bot Chat without acquiring a lease."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_state import SessionDB

    home = Path(profile_home).resolve()
    discovery = discover_gateway_endpoint(home, timeout=5)
    if discovery.state != 'ready' or discovery.endpoint is None:
        raise ValueError('profile authority is not ready')
    if not (home / 'state.db').is_file():
        return None
    db = SessionDB(db_path=home / 'state.db', read_only=True)
    try:
        row = db.get_session_by_title('Bot Chat')
        tip = db.get_compression_tip(row['id']) if row else None
    finally:
        db.close()
    if not tip:
        return None
    return dict(profile_home=str(home), session_id=tip, canonical=True,
                lease_id=discovery.endpoint.instance_id, live_session_id=tip)


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


def _fsync_dir(path: Path) -> None:
    # Windows cannot open directories with os.open; file fsync still applies.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _locked(home: Path | str):
    root = _root(home)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    _fsync_dir(root.parent)
    _fsync_dir(root.parent.parent)
    lock = root / ".lock"
    fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    with _FileLock(lock):
        yield root


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write(path: Path, record: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".delivery-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def deliver_to_live_owner(
    profile_home: Path | str, owner: dict[str, Any], message: str,
    *, delivery_id: str | None = None, author: dict[str, Any] | None = None,
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
        message=message, session_id=pinned["session_id"],
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
