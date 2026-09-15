"""Durable, at-most-once handoff to an existing Bot Chat owner.

Adapted from FalconOrtiz's live-owner mailbox (#101564). A single private
record advances queued -> claimed -> terminal under a process-shared lock.
Claims never expire: a crashed consumer leaves an inspectable unknown outcome,
not permission to execute the same input again. Receipts are permanent.

A pinned lease is a destination only while its consumer proves it is still
consuming: the owner's poll loop renews the lease stamp
(``hermes_cli.active_sessions.touch_active_session_lease``) and an entry whose
stamp is older than ``LIVE_CONSUMER_TTL_SECONDS`` is not a live owner, however
alive its process is. Mail pinned to a lease that is gone is reclaimed by the
profile's current live owner along the same compression lineage — but only
after the session store proves the body is not already there, so a reclaim can
never run the same input twice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_cli.active_sessions import _FileLock

logger = logging.getLogger(__name__)

DELIVERY_DIR_NAME = "bot_live_delivery"
QUARANTINE_DIR_NAME = "quarantine"
_OWNER_KEYS = ("profile_home", "session_id", "lease_id", "live_session_id")
_TERMINAL = frozenset({"settled", "failed", "cancelled", "ambiguous"})
# A live consumer renews its lease stamp on every poll tick (writes are throttled inside the
# registry), so this only has to outlast one long turn in an open pane: the poll loop keeps
# stamping between turns. Anything older is a consumer that stopped polling. Measured: a leaked
# dashboard Bot Chat lease whose process stayed alive ~20h held 68 queued peer messages.
LIVE_CONSUMER_TTL_SECONDS = 900.0
# A queued record this profile's live consumer can never reach along its compression lineage is
# terminal after this long instead of sitting silent forever. Reachable records keep their place:
# an owner may appear later.
UNREACHABLE_AFTER_SECONDS = 24 * 3600.0
# Below this length a body substring is too weak a signal to call a message already delivered.
_SHORTEST_SUBSTRING_BODY = 64


def _now(now: float | None = None) -> float:
    return time.time() if now is None else float(now)


def _stamp(entry: dict[str, Any], key: str) -> float:
    try:
        return float(entry.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_live_consumer(entry: dict[str, Any], *, now: float, ttl_seconds: float) -> bool:
    """True when the entry advertises the mailbox capability and is still being renewed."""
    meta = entry.get("metadata") or {}
    if meta.get("bot_live_delivery_consumer") is not True or not meta.get("live_session_id"):
        return False
    stamp = max(_stamp(entry, "updated_at"), _stamp(entry, "started_at"))
    return stamp > 0.0 and now - stamp <= ttl_seconds


def _registry_entry(profile_home: Path | str, owner: dict[str, Any]) -> dict[str, Any] | None:
    """Registry entry for this exact lease/live pair, or None when it is not provably live.

    Strict liveness prunes entries whose process is dead, and an unprovable registry (unreadable,
    or an owner whose liveness cannot be established) yields None rather than a guess: "could not
    prove an owner" must never become "this owner is live", or a corpse stays a destination.
    """
    from hermes_cli.active_sessions import ActiveSessionRegistryError, active_session_registry_snapshot

    try:
        entries = active_session_registry_snapshot(
            registry_home=Path(profile_home).resolve(), strict=True)
    except ActiveSessionRegistryError:
        logger.warning("Active-session registry is unprovable; owner is not live", exc_info=True)
        return None
    for entry in entries:
        if entry.get("lease_id") != owner["lease_id"]:
            continue
        if ((entry.get("metadata") or {}).get("live_session_id")) != owner["live_session_id"]:
            continue
        return entry
    return None


def _owner_is_live_consumer(
    profile_home: Path | str, owner: dict[str, Any], *,
    now: float | None = None, ttl_seconds: float = LIVE_CONSUMER_TTL_SECONDS,
) -> bool:
    """True when this owner is a registered consumer whose poll loop is still renewing."""
    entry = _registry_entry(profile_home, owner)
    if entry is None:
        return False
    return _is_live_consumer(entry, now=_now(now), ttl_seconds=ttl_seconds)


def find_canonical_live_owner(
    profile_home: Path | str, *, now: float | None = None,
    ttl_seconds: float = LIVE_CONSUMER_TTL_SECONDS,
) -> dict[str, Any] | None:
    """Resolve exact Bot Chat's compression tip without creating/migrating its DB.

    Capability advertisement is mandatory AND must be current, and the registry must prove the
    owner's process is alive (strict liveness prunes dead leases). Old Desktop/TUI processes must
    not receive work they cannot consume, and neither may a lease whose consumer stopped polling —
    its process can outlive its poll loop by hours (measured on this fleet: a Bot Chat lease whose
    last stamp was 14.9h old, still the only capability-advertising entry). An unreadable registry
    yields None, never a stale owner: the caller then takes the plain transport path, so an
    unprovable registry degrades to pre-mailbox behaviour instead of stranding mail in a mailbox.
    """
    from hermes_cli.active_sessions import ActiveSessionRegistryError, active_session_registry_snapshot
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
    try:
        entries = active_session_registry_snapshot(registry_home=home, strict=True)
    except ActiveSessionRegistryError:
        logger.warning("Active-session registry is unprovable; no live owner resolved", exc_info=True)
        return None
    stamp = _now(now)
    for entry in entries:
        meta = entry.get("metadata") or {}
        if (entry["session_id"] == session_id
                and _is_live_consumer(entry, now=stamp, ttl_seconds=ttl_seconds)):
            return dict(profile_home=str(home), session_id=session_id,
                        lease_id=entry["lease_id"], live_session_id=meta["live_session_id"])
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
    """Return the record, or None when it is absent, unreadable, or not a record.

    A corrupt ``*.json`` is moved to ``<spool>/quarantine/`` instead of raising: one bad byte
    must not wedge every later pass of the lane, and re-reading it on every sweep would keep
    the lane noisy forever. The record itself is preserved, never deleted.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:  # JSONDecodeError and UnicodeDecodeError are ValueError
        _quarantine(path, exc)
        return None
    if not isinstance(record, dict):
        _quarantine(path, ValueError("delivery record is not a JSON object"))
        return None
    return record


def _quarantine(path: Path, exc: BaseException) -> None:
    target = path.parent / QUARANTINE_DIR_NAME
    try:
        target.mkdir(mode=0o700, exist_ok=True)
        destination = target / path.name
        if destination.exists():
            destination = target / f"{path.stem}.{uuid.uuid4().hex[:8]}{path.suffix}"
        os.replace(path, destination)
        _fsync_dir(target)
        logger.warning("Quarantined unreadable delivery record %s: %s", destination.name, exc)
    except OSError:  # pragma: no cover - defensive: quarantine must never raise into a sweep
        logger.error("Could not quarantine unreadable delivery record %s", path, exc_info=True)


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


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finish_in_place(
    root: Path, record: dict[str, Any], *, status: str, reason: str, detail: str = "",
) -> dict[str, Any]:
    """Write a terminal receipt under the caller's lock.

    ``complete_delivery`` takes the same lock, so it cannot be used from inside a claim pass.
    """
    record.update(status=status, reply="", error="", reason=reason, status_detail=detail,
                  completed_at=time.time_ns())
    _write(root / f"{record['delivery_id']}.json", record)
    logger.info("Delivery %s finished as %s (%s)", record["delivery_id"], status, reason)
    return record


class _LineageIndex:
    """Compression-tip lookups for one claim pass (memoised; the store is opened at most once)."""

    def __init__(self, home: Path | str) -> None:
        self._path = Path(home).resolve() / "state.db"
        self._db = None
        self._tips: dict[str, str | None] = {}

    def _session_db(self):
        if self._db is None:
            from hermes_state import SessionDB

            try:
                self._db = SessionDB(db_path=self._path, read_only=True)
            except Exception:
                logger.warning("Could not open session store for lineage lookup", exc_info=True)
                self._db = False
        return self._db or None

    def tip(self, session_id: str) -> str | None:
        if session_id not in self._tips:
            db = self._session_db()
            if db is None:
                self._tips[session_id] = None
            else:
                try:
                    self._tips[session_id] = db.get_compression_tip(session_id)
                except Exception:
                    logger.warning("Could not resolve compression tip of %s", session_id, exc_info=True)
                    self._tips[session_id] = None
        return self._tips[session_id]

    def matches(self, record: dict[str, Any], owner: dict[str, str]) -> bool:
        pinned = str(record["owner"]["session_id"])
        return pinned == owner["session_id"] or self.tip(pinned) == owner["session_id"]


class _StoreBodies:
    """role=user bodies (with their sha256) of the sessions a reclaim could collide with.

    Read once per claim pass. The instrument is exact sha256 plus a substring fallback, matching
    the audit that produced this card's numbers (59/59 settled records matched, 0/90 queued):
    sha256 equality of two bodies IS exact string equality, so the hash is the fast path for the
    same predicate — it adds no tolerance, and a hit means the bytes are already in the store.
    """

    def __init__(self, home: Path | str, session_ids: tuple[str, ...]) -> None:
        self._digests: set[str] = set()
        self._bodies: list[str] = []
        state = Path(home).resolve() / "state.db"
        if not state.is_file():
            return
        from hermes_state import SessionDB

        try:
            db = SessionDB(db_path=state, read_only=True)
        except Exception:
            logger.warning("Could not open session store for delivery reclaim", exc_info=True)
            return
        try:
            for session_id in dict.fromkeys(sid for sid in session_ids if sid):
                try:
                    rows = db.get_messages(session_id, include_inactive=True, include_compacted=True)
                except Exception:
                    logger.warning("Could not read session %s for delivery reclaim", session_id, exc_info=True)
                    continue
                for row in rows:
                    if str(row.get("role") or "") != "user":
                        continue
                    body = row.get("content")
                    if isinstance(body, str) and body.strip():
                        text = body.strip()
                        self._digests.add(_digest(text))
                        self._bodies.append(text)
        finally:
            db.close()

    def contains(self, message: Any) -> bool:
        """True when this body is already in the store, so running it again would duplicate it."""
        body = message.strip() if isinstance(message, str) else ""
        if not body:
            return False
        if _digest(body) in self._digests:
            return True
        return any(len(body) >= _SHORTEST_SUBSTRING_BODY and body in text for text in self._bodies)


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
    key = _delivery_id(delivery_id if delivery_id is not None else uuid.uuid4().hex)
    with _locked(profile_home) as root:
        path = root / f"{key}.json"
        existing = _read(path)
        if existing is not None:
            if existing["owner"] != pinned or existing["message"] != message or existing.get("author") != author:
                raise ValueError("delivery id already belongs to a different payload")
            return existing
        # Wall time can roll back. Permanent receipts retain the admission
        # high-water mark, allocated while holding the cross-process lock.
        sequence = max((record.get("sequence", record["created_at"])
                        for candidate in root.glob("*.json")
                        if (record := _read(candidate)) is not None), default=0) + 1
        record = dict(delivery_id=key, id=key, owner=pinned, **pinned,
                      message=message, status="queued", created_at=time.time_ns(),
                      sequence=sequence, **({"author": dict(author)} if author else {}))
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
    profile_home: Path | str, owner: dict[str, Any], *, now: float | None = None,
) -> dict[str, Any] | None:
    """Claim oldest matching input exactly once; caller supplies its current lease.

    A lease transfer across compression is accepted only along the original
    stored session's compression chain. A new lease/live session cannot steal it.
    Caller must hold its normal turn-admission guard before invoking this.

    Two lanes keep the spool bounded and honest:

    * Reclaim: a record whose pinned owner is no longer a live consumer (lease released, or
      its poll loop stopped renewing) is claimable by this profile's current live consumer when
      the record sits on that consumer's compression lineage. Before such a record is handed
      over, the session store is checked for its body: mail that already ran is settled in place
      with an ``already_present`` receipt and never runs again.
    * Expiry: a queued record this consumer can never reach (a different lineage, its pinned
      owner gone) becomes ``failed``/``no_live_consumer`` once it is older than
      ``UNREACHABLE_AFTER_SECONDS`` — an unreachable ask is reported, not left silent. Its body
      stays in the receipt.

    Only a provably-live consumer may reclaim or expire: an unregistered caller cannot take
    mail off another lease's hands.
    """
    current = _owner(profile_home, owner)
    if not _root(profile_home).is_dir():
        return None
    stamp = _now(now)
    with _locked(profile_home) as root:
        claimant_is_live = _owner_is_live_consumer(profile_home, current, now=stamp)
        lineage = _LineageIndex(profile_home)
        bodies: _StoreBodies | None = None
        pending: list[dict[str, Any]] = []
        for path in root.glob("*.json"):
            record = _read(path)
            if record is None or record.get("status") != "queued":
                continue
            if _matches(profile_home, record, current):
                pending.append(record)
                continue
            if not claimant_is_live:
                continue
            pinned = record["owner"]
            if not lineage.matches(record, current):
                created_at = float(record.get("created_at") or 0.0)
                reachable_by_its_owner = (pinned["lease_id"] == current["lease_id"])
                if (not reachable_by_its_owner
                        and stamp - created_at / 1e9 > UNREACHABLE_AFTER_SECONDS):
                    _finish_in_place(root, record, status="failed", reason="no_live_consumer",
                                     detail="no_live_consumer")
                continue
            if _owner_is_live_consumer(profile_home, pinned, now=stamp):
                continue  # its owner is still consuming: never steal a live owner's mail
            if bodies is None:
                bodies = _StoreBodies(profile_home, (pinned["session_id"], current["session_id"]))
            if bodies.contains(record.get("message")):
                _finish_in_place(root, record, status="settled", reason="already_present",
                                 detail="already_present")
                continue
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
