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

Two lanes share the spool. Mail with a pinned ``owner`` is the live-owner lane
above. Mail with ``lane == LEASE_QUEUE_LANE`` is a conversation lane: a turn that
could not out-wait the conversation's session turn lease hands its inbound
delivery here, pinned to the conversation instead of to a lease (the holder it
failed to out-wait is exactly the holder it must not name as its own), and the
next turn admitted on that conversation drains it as its first user message. A
queue record is never claimable by the live-owner lane and vice versa.
"""
from __future__ import annotations

import hashlib
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
# Lane tag for mail pinned to a conversation rather than to a live-owner lease (a turn that lost
# the session turn lease wait). Absent tag + a pinned ``owner`` dict means the live-owner lane.
LEASE_QUEUE_LANE = "lease_queue"


def _is_live_owner_record(record: dict[str, Any]) -> bool:
    """True when this record belongs to the live-owner lane (pinned to a lease/live pair)."""
    return isinstance(record.get("owner"), dict)


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
        log.warning("Active-session registry is unprovable; owner is not live", exc_info=True)
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
    """Only advertised consumers may receive owner-pinned mailbox deliveries.

    Resolve exact Bot Chat's compression tip without creating/migrating its DB.
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
        log.warning("Active-session registry is unprovable; no live owner resolved", exc_info=True)
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
    """Exact-id read: absent → None; unreadable or not a JSON object → raises (callers fail closed).

    Directory sweeps call :func:`_scan_read`, which quarantines such a ticket instead of raising.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"ticket {path.name} is not a JSON object ({type(record).__name__})")
    return record


def _quarantine(path: Path, exc: BaseException) -> None:
    target = path.parent / QUARANTINE_DIR_NAME
    try:
        target.mkdir(mode=0o700, exist_ok=True)
        destination = target / path.name
        if destination.exists():
            destination = target / f"{path.stem}.{uuid.uuid4().hex[:8]}{path.suffix}"
        os.replace(path, destination)
        fsync_directory(target)
        log.warning("Quarantined unreadable delivery record %s: %s", destination.name, exc)
    except OSError:  # pragma: no cover - defensive: quarantine must never raise into a sweep
        log.error("Could not quarantine unreadable delivery record %s", path, exc_info=True)


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

    A ticket whose *content* is dead — unparseable JSON, invalid UTF-8, JSON
    that is not an object — is moved to ``<spool>/quarantine/`` as it is
    skipped: re-reading one bad byte on every pass would keep the lane noisy
    forever, and the record is preserved, never deleted. An I/O error is left
    in place (the record may be perfectly good once the environment recovers).
    """
    try:
        record = _read(path)
        problem = None if record is None else _ticket_shape_error(path, record)
    except ValueError as exc:  # corrupt JSON, invalid UTF-8, or not a JSON object
        _quarantine(path, exc)
        return None
    except OSError as exc:  # permission/IO error: uninspectable, not provably dead
        level = logging.DEBUG if path in _warned_unreadable else logging.WARNING
        _warned_unreadable.add(path)
        log.log(level, "bot_live_delivery: skipping unreadable ticket %s (%s)", path.name, exc)
        return None
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
    scanned = max((record.get("sequence") or record.get("created_at") or 0
                   for candidate in root.glob("*.json")
                   if (record := _scan_read(candidate)) is not None), default=0)
    sequence = max(persisted, scanned) + 1
    atomic_write_text(counter, str(sequence), mode=0o600, fsync_dir=True)
    return sequence


def _write(path: Path, record: dict[str, Any]) -> None:
    atomic_json_write(path, record, indent=None, sort_keys=True, fsync_dir=True, mode=0o600)


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
    log.info("Delivery %s finished as %s (%s)", record["delivery_id"], status, reason)
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
                log.warning("Could not open session store for lineage lookup", exc_info=True)
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
                    log.warning("Could not resolve compression tip of %s", session_id, exc_info=True)
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
            log.warning("Could not open session store for delivery reclaim", exc_info=True)
            return
        try:
            for session_id in dict.fromkeys(sid for sid in session_ids if sid):
                try:
                    rows = db.get_messages(session_id, include_inactive=True, include_compacted=True)
                except Exception:
                    log.warning("Could not read session %s for delivery reclaim", session_id, exc_info=True)
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
        """True when this body is already in the store, so running it again would duplicate it.

        The arm that matched is logged with the lengths it matched at — the exact ``sha256`` arm, or
        the ``substring`` arm over stored user bodies. The substring arm is the one that can match a
        body the store never ran verbatim, and a false match settles a record as ``already_present``
        (a silent drop), so this line is the audit trail that measures how often it does; the bodies
        themselves are never logged.
        """
        body = message.strip() if isinstance(message, str) else ""
        if not body:
            return False
        if _digest(body) in self._digests:
            log.info("Delivery reclaim matched by sha256 (%d-char body)", len(body))
            return True
        if len(body) < _SHORTEST_SUBSTRING_BODY:
            return False
        for text in self._bodies:
            if body in text:
                log.info("Delivery reclaim matched by substring (%d-char body inside a %d-char "
                         "stored body)", len(body), len(text))
                return True
        return False


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


def _looks_like_delivery(record: dict[str, Any]) -> bool:
    """Whether a scanned JSON object carries the fields the reclaim/expiry lanes read.

    The spool is a plain directory, so a foreign or half-written ``*.json`` can
    appear in it; such a file must be ignored, never crash the sweep the lane
    depends on.
    """
    owner = record.get("owner")
    return (isinstance(record.get("delivery_id"), str) and isinstance(owner, dict)
            and all(isinstance(owner.get(key), str) and owner[key] for key in _OWNER_KEYS))


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


def _created_at_seconds(record: dict[str, Any]) -> float | None:
    """The record's age basis in epoch seconds, or ``None`` when it carries no usable stamp.

    ``None`` means *not measurable*, never *infinitely old*: expiry is terminal (the record never
    reaches a turn afterwards), and ``_looks_like_delivery`` does not require ``created_at``, so a
    half-written or foreign ticket must not be failed on its first sweep merely because its age
    cannot be read. A stamp that is absent, zero or not a number is retained for inspection or for a
    later sweep that can read it — and a non-numeric stamp must not raise out of the sweep, which
    would wedge every other record behind it.
    """
    raw = record.get("created_at")
    try:
        stamp = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return None
    return stamp / 1e9 if stamp > 0 else None


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
      stays in the receipt. A record whose ``created_at`` is absent, zero or unreadable has no age
      to compare and is kept queued (``_created_at_seconds``) rather than failed on its first sweep.

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
            record = _scan_read(path)
            if record is None:  # quarantined or uninspectable: _scan_read reported it
                continue
            looks_like_delivery = _looks_like_delivery(record)
            if not _is_live_owner_record(record):
                # Conversation-lane mail is claimed by the turn that takes that conversation's
                # session turn lease, never by a live-owner consumer.
                continue
            if (looks_like_delivery and record["status"] == "queued"
                    and _matches(profile_home, record, current)):
                pending.append(record)
                continue
            if not claimant_is_live or not looks_like_delivery:
                continue
            pinned = record["owner"]
            if not lineage.matches(record, current):
                created_at = _created_at_seconds(record)
                reachable_by_its_owner = (pinned["lease_id"] == current["lease_id"])
                if (not reachable_by_its_owner and created_at is not None
                        and stamp - created_at > UNREACHABLE_AFTER_SECONDS):
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


def _complete_in_place(
    root: Path, key: str, *, status: str, reply: str, error: str, reason: str,
) -> dict[str, Any]:
    """Terminal-receipt discipline for both lanes: immutable, idempotent, claimed-first."""
    if status not in _TERMINAL:
        raise ValueError("invalid terminal delivery status")
    outcome = dict(status=status, reply=reply, error=error, reason=reason)
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


def complete_delivery(
    profile_home: Path | str, delivery_id: str, *, status: str,
    reply: str = "", error: str = "", reason: str = "",
) -> dict[str, Any]:
    """Persist an immutable terminal receipt; duplicate identical completion is safe."""
    with _locked(profile_home) as root:
        return _complete_in_place(
            root, _delivery_id(delivery_id), status=status, reply=reply, error=error, reason=reason,
        )


def complete_lease_delivery(
    profile_home: Path | str, delivery_id: str, *, status: str,
    reply: str = "", error: str = "", reason: str = "",
) -> dict[str, Any]:
    """Persist a conversation-lane terminal receipt (same discipline as ``complete_delivery``)."""
    with _locked(profile_home) as root:
        return _complete_in_place(
            root, _lease_queue_key(delivery_id), status=status, reply=reply, error=error,
            reason=reason,
        )


#: ENUM B (§4.8): a sender never sees the transport's storage words. The storage layer keeps
#: ``queued -> claimed -> settled`` (the ``claimed/``/``settled/`` dirs and the record's
#: ``status`` field are unchanged); only this boundary translation renames them.
_PUBLIC_STATUS = {"claimed": "running", "settled": "delivered"}


def public_delivery_status(status: str) -> str:
    """Translate a stored delivery status to the sender-visible ENUM B word (§4.8)."""
    word = str(status or "")
    return _PUBLIC_STATUS.get(word, word)


def public_delivery_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """A sender-visible copy of ``record`` — status translated, storage never mutated."""
    if record is None:
        return None
    public = dict(record)
    public["status"] = public_delivery_status(record.get("status"))
    return public


def read_public_delivery_result(profile_home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """Sender-facing read: :func:`read_delivery_result` with ENUM B status words (§4.8)."""
    return public_delivery_record(read_delivery_result(profile_home, delivery_id))


def read_delivery_result(profile_home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """Read admission/claim/terminal state without waiting or deleting its receipt."""
    return _read(_root(profile_home) / f"{_delivery_id(delivery_id)}.json")


def _lease_queue_key(delivery_id: str) -> str:
    """Filesystem-safe spool key for a conversation-lane delivery id.

    The live-owner lane hands over canonical hex ids; a turn that loses the lease wait hands over
    whatever its caller uses (peer receipts, request ids), so a foreign id is digested, never
    trusted as a path component.
    """
    if re.fullmatch(r"[0-9a-f]{32,64}", delivery_id):
        return delivery_id
    return _digest(delivery_id)


def enqueue_lease_delivery(
    profile_home: Path | str, *, conversation_id: str, message: str, holder: str = "",
    author: str | dict[str, Any] | None = None, delivery_id: str | None = None,
) -> dict[str, Any]:
    """Spool an inbound turn that lost the session turn lease wait onto its conversation.

    ``holder`` is recorded for inspection only: the point of this lane is that the delivery is
    pinned to the CONVERSATION, not to a lease, so the next turn admitted on that conversation runs
    the message instead of the transcript losing it. Re-sending the same ``delivery_id`` with the
    same payload returns the record unchanged and a terminal record is never reopened, so a peer
    that retries after delivery cannot make the conversation run the same message twice.
    """
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if author is not None and not isinstance(author, (str, dict)):
        raise ValueError("author must be a string, a dict, or None")
    record_id = str(delivery_id).strip() if delivery_id else uuid.uuid4().hex
    if not record_id:
        raise ValueError("delivery_id must be a non-empty string when given")
    pinned_author = author.strip() if isinstance(author, str) else author
    if not pinned_author:
        pinned_author = None
    with _locked(profile_home) as root:
        path = root / f"{_lease_queue_key(record_id)}.json"
        existing = _read(path)
        if existing is not None:
            if (
                existing.get("lane") != LEASE_QUEUE_LANE
                or existing.get("conversation_id") != conversation_id
                or existing.get("message") != message
                or existing.get("author") != pinned_author
            ):
                raise ValueError("delivery id already belongs to a different payload")
            return existing
        record: dict[str, Any] = {
            "delivery_id": record_id,
            "id": record_id,
            "lane": LEASE_QUEUE_LANE,
            "conversation_id": conversation_id,
            "holder": str(holder or ""),
            "message": message,
            "status": "queued",
            "created_at": time.time_ns(),
            "sequence": _next_sequence(root),
        }
        if pinned_author is not None:
            record["author"] = pinned_author
        _write(path, record)
        return record


def read_lease_delivery(profile_home: Path | str, delivery_id: str) -> dict[str, Any] | None:
    """Read a conversation-lane record's state without consuming it (receipt inspection)."""
    return _read(_root(profile_home) / f"{_lease_queue_key(delivery_id)}.json")


def _lease_body_bytes(record: dict[str, Any]) -> int:
    """The byte size a caller spends its drain budget on: the stripped body in UTF-8."""
    return len(str(record.get("message") or "").strip().encode("utf-8"))


def claim_lease_delivery(
    profile_home: Path | str, *, conversation_id: str, max_bytes: int | None = None,
) -> dict[str, Any] | None:
    """Claim this conversation's oldest queued lease delivery, exactly once.

    The claimant (a turn that just took the conversation's session turn lease) owns the body from
    here: the record is marked claimed and stays inspectable, so a crash mid-turn leaves a receipt
    of an unknown outcome rather than a message that runs twice. ``complete_delivery`` closes it
    once the body has been handed to the turn.

    ``max_bytes`` is the room left in the batch the caller is assembling: when the oldest record's
    body does not fit, ``None`` is returned and that record stays QUEUED and unclaimed, so a bounded
    drain never claims a record it cannot run. The measure is the stripped body's UTF-8 length, the
    same one the caller spends its budget with. ``None`` means no budget - take the head.
    """
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string")
    if not _root(profile_home).is_dir():
        # Nothing spooled for this profile: never create a store just to look in it.
        return None
    with _locked(profile_home) as root:
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for path in root.glob("*.json"):
            record = _read(path)
            if record is None or record.get("status") != "queued":
                continue
            if record.get("lane") != LEASE_QUEUE_LANE:
                continue
            if record.get("conversation_id") != conversation_id:
                continue
            candidates.append((
                record.get("sequence") or record.get("created_at") or 0.0,
                str(record.get("delivery_id") or path.name),
                record,
            ))
        if not candidates:
            return None
        _, _, record = min(candidates, key=lambda item: (item[0], item[1]))
        if max_bytes is not None and _lease_body_bytes(record) > max_bytes:
            return None
        record.update(status="claimed", claimed_at=time.time_ns())
        _write(root / f"{_lease_queue_key(str(record['delivery_id']))}.json", record)
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
