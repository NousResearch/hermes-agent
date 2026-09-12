"""Durable receiver-side delivery queue for peer sends (spec §2).

A busy target's peer DM is *accepted* and queued here (``result: "receipt"``);
the **receiver** -- never the sender -- waits for a free turn slot and retries
with backoff. Nothing in this module blocks a sender for more than
``receipt_after_seconds``.

Storage mirrors the atomic-write / rename-claim idiom of
``tools.bot_live_delivery``: a per-home ``_FileLock`` plus
``tempfile`` + ``fsync`` + ``os.replace`` writes, so a crash can never strand a
half-written record.

Layout (``<home>/runtime/bot_delivery``)::

    queue/<delivery_id>.json      status queued | running
    claimed/<delivery_id>.json    atomically renamed here before the turn
    settled/<delivery_id>.json    immutable audit trail

Record lifecycle is ENUM B (7 values); the sender-visible result is ENUM A
(5 values). ``attempts`` counts turn executions only -- a contended lease
increments ``reoffer_count`` instead, so contention never consumes an attempt.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from hermes_cli.active_sessions import _FileLock

logger = logging.getLogger(__name__)

DELIVERY_DIR_NAME = "bot_delivery"
QUEUE_DIR = "queue"
CLAIMED_DIR = "claimed"
SETTLED_DIR = "settled"

# --- ENUM B: durable record lifecycle (7 values, no "target_busy").
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DELIVERED = "delivered"
STATUS_FAILED = "failed"
STATUS_EXPIRED = "expired"
STATUS_CANCELLED = "cancelled"
STATUS_AMBIGUOUS = "ambiguous"

STATUSES = (
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_DELIVERED,
    STATUS_FAILED,
    STATUS_EXPIRED,
    STATUS_CANCELLED,
    STATUS_AMBIGUOUS,
)

TERMINAL_STATUSES = frozenset(
    {
        STATUS_DELIVERED,
        STATUS_FAILED,
        STATUS_EXPIRED,
        STATUS_CANCELLED,
        STATUS_AMBIGUOUS,
    }
)

# --- ENUM A: sender-visible result (5 values).
RESULT_DELIVERED = "delivered"
RESULT_RECEIPT = "receipt"
RESULT_FAILED = "failed"
RESULT_REFUSED = "refused"
RESULT_UNKNOWN = "unknown"

RESULTS = (
    RESULT_DELIVERED,
    RESULT_RECEIPT,
    RESULT_FAILED,
    RESULT_REFUSED,
    RESULT_UNKNOWN,
)

OBJECT_NAME = "hermes.peer.send_result"

#: Envelope field order (§1.2/§1.3) -- exactly 23 fields, ``object`` first.
ENVELOPE_FIELDS = (
    "object",
    "result",
    "status",
    "status_detail",
    "send_id",
    "delivery_id",
    "idempotency_key",
    "replayed",
    "peer",
    "profile",
    "session_id",
    "queue_position",
    "attempts",
    "queued_seconds",
    "waited_seconds",
    "busy",
    "retryable",
    "retry_after_seconds",
    "reply",
    "error",
    "reason",
    "detail",
    "at",
)

#: Durable queue-record field order (§2.3) -- exactly 20 fields.
RECORD_FIELDS = (
    "delivery_id",
    "idempotency_key",
    "fingerprint",
    "sender_profile",
    "target_profile",
    "target_session_id",
    "message",
    "status",
    "status_detail",
    "attempts",
    "reoffer_count",
    "queue_position",
    "created_at",
    "updated_at",
    "claimed_at",
    "attempts_log",
    "reply",
    "error",
    "reason",
    "sequence",
)

# --- exact user-facing strings (§3.4); asserted verbatim by the tests.
DETAIL_RECEIPT = "accepted and queued — do NOT resend; receipt is retained"
DETAIL_HELD_LIVE = (
    "Delivery remains pending or its outcome is unknown. "
    "Do not resend; receipt is retained."
)
STATUS_DETAIL_TARGET_BUSY = "target_busy: turn slot held by another turn; delivery queued"
STATUS_DETAIL_LIVE_OWNER = "live_owner_present"
STATUS_DETAIL_LEASE_CONTENDED = "lease_contended"
STATUS_DETAIL_QUEUE_FULL = "queue_full"
#: A queued delivery whose target slot is FREE -- a different cause from
#: TARGET_BUSY, and the one a receipt must not misreport (VERIFICATION D2).
STATUS_DETAIL_BACKLOG = (
    "target_free: turn slot free, delivery queued behind earlier deliveries"
)

#: Which cap :func:`check_capacity`/:func:`admit` refused on (VERIFICATION D6).
LIMIT_PER_PROFILE = "per_profile"
LIMIT_PER_SENDER = "per_sender"

REASON_QUEUED_EXPIRED = "queued_expired"

#: `retryable(envelope)` is derived from the *envelope reason* only. It is
#: unrelated to ``retry_action()`` (agent auto-retry) -- never derive one from
#: the other.
RETRYABLE_REASONS = frozenset({"queue_full", "runtime_offline"})

QUEUE_FULL_RETRY_AFTER_SECONDS = 60.0

_DELIVERY_ID_RE = re.compile(r"[0-9a-f]{32,64}")

#: Spec §2.4 defaults. Read live from ``bot_mode`` config via :func:`_cfg`.
_DEFAULTS: dict[str, float] = {
    "receipt_after_seconds": 15,
    "delivery_queue_ttl_seconds": 1800,
    "delivery_queue_max_per_profile": 32,
    "delivery_queue_max_per_sender": 8,
    "delivery_retry_base_seconds": 2,
    "delivery_sweep_seconds": 30,
    "delivery_max_turn_attempts": 3,
    "delivery_probe_base_seconds": 0.5,
    "delivery_probe_max_seconds": 2.0,
    "delivery_probe_jitter": 0.2,
    "lease_probe_seconds": 2,
    "dedup_window_seconds": 900,
}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def _cfg(key: str, *, loader: str = "load_config_readonly") -> Any:
    """Read one ``bot_mode`` setting, falling back to the spec default."""
    value = None
    try:  # pragma: no cover - exercised through the accessors
        import hermes_cli.config as _config

        getter = getattr(_config, loader, None) or getattr(_config, "load_config")
        cfg = getter() or {}
        value = (cfg.get("bot_mode") or {}).get(key)
    except Exception:
        value = None
    if value is None:
        return _DEFAULTS[key]
    return value


def receipt_after_seconds() -> float:
    return float(_cfg("receipt_after_seconds"))


def queue_ttl_seconds() -> float:
    return float(_cfg("delivery_queue_ttl_seconds"))


def max_per_profile() -> int:
    return int(_cfg("delivery_queue_max_per_profile"))


def max_per_sender() -> int:
    return int(_cfg("delivery_queue_max_per_sender"))


def max_turn_attempts() -> int:
    return int(_cfg("delivery_max_turn_attempts"))


def sweep_seconds() -> float:
    return float(_cfg("delivery_sweep_seconds"))


def retry_base_seconds() -> float:
    return float(_cfg("delivery_retry_base_seconds"))


def lease_probe_seconds() -> float:
    return float(_cfg("lease_probe_seconds"))


# --------------------------------------------------------------------------
# backoff curves (§2.6)
# --------------------------------------------------------------------------
def probe_delay(n: int, *, rng: Any = random) -> float:
    """Lease-probe loop delay: ``uniform(0.8, 1.2) * min(2.0, 0.5 * 2**n)``."""
    base = float(_cfg("delivery_probe_base_seconds"))
    cap = float(_cfg("delivery_probe_max_seconds"))
    jitter = float(_cfg("delivery_probe_jitter"))
    return rng.uniform(1.0 - jitter, 1.0 + jitter) * min(cap, base * (2 ** n))


def drainer_delay(n: int, *, rng: Any = random) -> float:
    """Queue-drain loop delay: ``uniform(0.5, 1.0) * min(30, 2 * 2**n)``."""
    base = float(_cfg("delivery_retry_base_seconds"))
    cap = float(_cfg("delivery_sweep_seconds"))
    return rng.uniform(0.5, 1.0) * min(cap, base * (2 ** n))


# --------------------------------------------------------------------------
# observed turn-slot state (spec 3.1; VERIFICATION D2/D7)
# --------------------------------------------------------------------------
def _log(event: str, record: dict[str, Any] | None = None, **fields: Any) -> None:
    """One log line per delivery decision.

    The drain is a background path: without these, production has no way to
    tell an accepted-but-waiting delivery from a dropped one (VERIFICATION D5).
    """
    parts = []
    if record is not None:
        parts.append(f"delivery_id={record.get('delivery_id')}")
        for key_ in ("status", "target_profile", "sender_profile"):
            if record.get(key_):
                parts.append(f"{key_}={record.get(key_)}")
    parts.extend(f"{key_}={value}" for key_, value in sorted(fields.items()))
    logger.info("bot_delivery %s %s", event, " ".join(parts))


def busy_from_detail(record: dict[str, Any]) -> bool:
    """Whether the record's own ``status_detail`` says the slot is held.

    Deriving ``busy`` from the same field the envelope prints is what stops a
    replayed envelope from contradicting its own cause (VERIFICATION D7).

    A live-owner hold is a HELD slot too (P12): the target's Bot Chat is owned by
    another surface, so the receipt for it must report ``busy: true`` — exactly as
    the target_busy receipt does (VERIFICATION D2).
    """
    return record.get("status_detail") in (STATUS_DETAIL_TARGET_BUSY, STATUS_DETAIL_LIVE_OWNER)


def slot_held(home: str | os.PathLike[str], profile: str) -> bool:
    """Whether ``profile``'s turn slot is currently held by another turn.

    The lock is the durable relay lock, so this sees a holder that is not a
    queue consumer at all -- a desktop or card session mid-turn holds the same
    slot. Imported lazily because :mod:`tools.bot_relay` imports this module.
    Never raises: an unreadable lock reads as free (a probe failure must not
    expire a delivery, and it cannot authorise a claim either).
    """
    try:
        from tools.bot_mode_probe import _hermes_root
        from tools.bot_relay import TurnBusyError, acquire_delivery_turn_lock

        try:
            with acquire_delivery_turn_lock(_hermes_root(Path(home)), profile):
                return False
        except TurnBusyError:
            return True
    except Exception:  # pragma: no cover - defensive
        return False


def observed_detail(home: str | os.PathLike[str], profile: str) -> str:
    """The honest ``status_detail`` for a queued delivery, observed now."""
    if slot_held(home, profile):
        return STATUS_DETAIL_TARGET_BUSY
    return STATUS_DETAIL_BACKLOG


# --------------------------------------------------------------------------
# user-facing strings
# --------------------------------------------------------------------------
def expired_detail(ttl_seconds: float | None = None) -> str:
    ttl = queue_ttl_seconds() if ttl_seconds is None else ttl_seconds
    return f"queued {int(ttl)}s without a free turn slot; not delivered"


def queue_full_detail(
    per_profile: int | None = None,
    retry_after: float | None = None,
    *,
    limit_kind: str = LIMIT_PER_PROFILE,
    limit: int | None = None,
) -> str:
    """Name the cap that actually refused the send (VERIFICATION D6).

    A send refused by the per-sender cap was previously reported as
    ``queue full (32 per profile)`` -- an operator would raise the wrong limit.
    """
    if limit is not None:
        cap = int(limit)
    elif per_profile is None:
        cap = max_per_profile()
    else:
        cap = int(per_profile)
    after = QUEUE_FULL_RETRY_AFTER_SECONDS if retry_after is None else retry_after
    scope = "per profile" if limit_kind == LIMIT_PER_PROFILE else "pending from this sender"
    return (
        f"queue full ({cap} {scope}); no reservation was taken — "
        f"safe to retry with the same idempotency_key after {int(after)}s"
    )


def unknown_detail(idempotency_key: str | None = None) -> str:
    key = idempotency_key or "<idempotency_key>"
    return f"outcome unknown; a resend must reuse idempotency_key {key}"


def build_notification_text(record: dict[str, Any]) -> str:
    """Held-delivery notice for the target's live session (§3.5)."""
    if record.get("status_detail") == STATUS_DETAIL_LIVE_OWNER:
        return DETAIL_HELD_LIVE
    status = record.get("status")
    if status == STATUS_EXPIRED:
        return expired_detail()
    if status == STATUS_AMBIGUOUS:
        return unknown_detail(record.get("idempotency_key"))
    if record.get("reason") == "queue_full":
        return queue_full_detail()
    return DETAIL_RECEIPT


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
class QueueFullError(RuntimeError):
    """Raised by :func:`admit` when the per-profile or per-sender cap is met.

    ``QUEUE_FULL`` is the only new failure reason; there is no ``target_busy``
    reason anywhere in the vocabulary.
    """

    reason = "queue_full"

    def __init__(
        self,
        profile: str,
        *,
        limit_kind: str = LIMIT_PER_PROFILE,
        limit: int | None = None,
        message: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.profile = profile
        self.limit_kind = limit_kind
        self.limit = limit
        self.retry_after_seconds = (
            QUEUE_FULL_RETRY_AFTER_SECONDS
            if retry_after_seconds is None
            else float(retry_after_seconds)
        )
        super().__init__(
            message or queue_full_detail(limit_kind=limit_kind, limit=limit)
        )


# --------------------------------------------------------------------------
# validation + paths
# --------------------------------------------------------------------------
def validate_idempotency_key(key: Any) -> str:
    """Normalize and validate an idempotency key (mirrors the CLI validator)."""
    text = ("" if key is None else str(key)).strip()
    if not text or len(text) > 255 or re.search(r"[\r\n\x00]", text):
        raise ValueError("idempotency key must be 1-255 characters without control newlines")
    return text


def validate_delivery_id(delivery_id: Any) -> str:
    text = "" if delivery_id is None else str(delivery_id)
    if not _DELIVERY_ID_RE.fullmatch(text):
        raise ValueError("delivery_id must be 32-64 lowercase hex characters")
    return text


def delivery_root(home: str | os.PathLike[str]) -> Path:
    return Path(home).resolve() / "runtime" / DELIVERY_DIR_NAME


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - platform specific
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform specific
        pass
    finally:
        os.close(fd)


@contextmanager
def _locked(home: str | os.PathLike[str]) -> Iterator[Path]:
    root = delivery_root(home)
    for sub in (QUEUE_DIR, CLAIMED_DIR, SETTLED_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(root, 0o700)
    except OSError:  # pragma: no cover
        pass
    lock_path = root / ".lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    with _FileLock(lock_path):
        yield root


def _write(path: Path, record: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _list(root: Path, sub: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / sub).glob("*.json")):
        record = _read(path)
        if isinstance(record, dict):
            records.append(record)
    return records


def _find(root: Path, delivery_id: str) -> dict[str, Any] | None:
    for sub in (QUEUE_DIR, CLAIMED_DIR, SETTLED_DIR):
        record = _read(root / sub / f"{delivery_id}.json")
        if record is not None:
            return record
    return None


def _high_water(root: Path) -> int:
    best = 0
    for sub in (QUEUE_DIR, CLAIMED_DIR, SETTLED_DIR):
        for record in _list(root, sub):
            marker = record.get("sequence", record.get("created_at", 0))
            try:
                marker = int(marker)
            except (TypeError, ValueError):
                marker = 0
            best = max(best, marker)
    return best


def _pending(root: Path, *, target_profile: str | None = None,
             sender_profile: str | None = None) -> int:
    count = 0
    for sub in (QUEUE_DIR, CLAIMED_DIR):
        for record in _list(root, sub):
            if record.get("status") in TERMINAL_STATUSES:
                continue
            if target_profile is not None and record.get("target_profile") != target_profile:
                continue
            if sender_profile is not None and record.get("sender_profile") != sender_profile:
                continue
            count += 1
    return count


# --------------------------------------------------------------------------
# timestamps / derived fields
# --------------------------------------------------------------------------
def queued_seconds(record: dict[str, Any], *, now_ns: int | None = None) -> float:
    """Seconds the record spent *waiting* (sum of its queued intervals)."""
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    cursor = int(record.get("created_at") or 0)
    total = 0
    for entry in sorted(
        record.get("attempts_log") or [], key=lambda e: (e.get("attempt") or 0)
    ):
        started = entry.get("started_at")
        if started and started > cursor:
            total += started - cursor
        ended = entry.get("ended_at")
        if ended is None:
            # a turn is running right now: no further queued time accrues
            return total / 1e9
        cursor = max(cursor, ended)
    if record.get("status") not in TERMINAL_STATUSES:
        total += max(0, now_ns - cursor)
    return total / 1e9


def _queue_position(root: Path, record: dict[str, Any]) -> int | None:
    if record.get("status") != STATUS_QUEUED:
        return None
    sequence = record.get("sequence", 0)
    ahead = 0
    for other in _list(root, QUEUE_DIR):
        if other.get("target_profile") != record.get("target_profile"):
            continue
        if other.get("status") != STATUS_QUEUED:
            continue
        if other.get("delivery_id") == record.get("delivery_id"):
            continue
        if other.get("sequence", 0) < sequence:
            ahead += 1
    return ahead + 1


def _result_for_status(status: str) -> str:
    if status == STATUS_DELIVERED:
        return RESULT_DELIVERED
    if status in (STATUS_FAILED, STATUS_EXPIRED, STATUS_CANCELLED):
        return RESULT_FAILED
    if status == STATUS_AMBIGUOUS:
        return RESULT_UNKNOWN
    return RESULT_RECEIPT  # queued | running


def _open_attempt(record: dict[str, Any]) -> dict[str, Any] | None:
    for entry in reversed(record.get("attempts_log") or []):
        if entry.get("ended_at") is None:
            return entry
    return None


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def admit(
    home: str | os.PathLike[str],
    *,
    sender_profile: str,
    target_profile: str,
    target_session_id: str | None,
    idempotency_key: str,
    fingerprint: str,
    delivery_id: str,
    message: str,
    sequence_hint: int | None = None,
    status_detail: str | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Admit one delivery into the queue, or return the existing record.

    Idempotent on ``delivery_id``: a repeat admit returns the stored record and
    writes nothing (one record, one sequence). Raises :class:`QueueFullError`
    when either capacity cap would be exceeded.
    """
    key = validate_idempotency_key(idempotency_key)
    key_id = validate_delivery_id(delivery_id)
    if not isinstance(message, str):
        raise ValueError("message must be a string")
    now_ns = time.time_ns() if now_ns is None else int(now_ns)

    with _locked(home) as root:
        existing = _find(root, key_id)
        if existing is not None:
            return dict(existing)

        if _pending(root, target_profile=target_profile) >= max_per_profile():
            raise QueueFullError(
                target_profile,
                limit_kind=LIMIT_PER_PROFILE,
                limit=max_per_profile(),
            )
        if (
            _pending(root, target_profile=target_profile, sender_profile=sender_profile)
            >= max_per_sender()
        ):
            raise QueueFullError(
                target_profile,
                limit_kind=LIMIT_PER_SENDER,
                limit=max_per_sender(),
            )

        sequence = max(int(sequence_hint or 0), _high_water(root) + 1)
        record: dict[str, Any] = {
            "delivery_id": key_id,
            "idempotency_key": key,
            "fingerprint": fingerprint,
            "sender_profile": sender_profile,
            "target_profile": target_profile,
            "target_session_id": target_session_id,
            "message": message,
            "status": STATUS_QUEUED,
            "status_detail": status_detail or STATUS_DETAIL_TARGET_BUSY,
            "attempts": 0,
            "reoffer_count": 0,
            "queue_position": 0,
            "created_at": now_ns,
            "updated_at": now_ns,
            "claimed_at": None,
            "attempts_log": [],
            "reply": None,
            "error": None,
            "reason": None,
            "sequence": sequence,
        }
        record["queue_position"] = _queue_position(root, record) or 1
        _write(root / QUEUE_DIR / f"{key_id}.json", record)
        _log("admitted", record, queue_position=record["queue_position"])
        return dict(record)


def claim_next(
    home: str | os.PathLike[str],
    *,
    target_profile: str,
    lease_ok: bool = False,
    now_ns: int | None = None,
) -> dict[str, Any] | None:
    """Claim the oldest queued record for ``target_profile`` (FIFO).

    With ``lease_ok=False`` (the lease probe did not grant a turn) the oldest
    record is left ``queued`` and only ``reoffer_count`` is bumped -- attempts
    are never consumed by contention. Returns ``None`` in that case and when
    the queue is empty for the profile.
    """
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    with _locked(home) as root:
        queued = [
            record
            for record in _list(root, QUEUE_DIR)
            if record.get("target_profile") == target_profile
            and record.get("status") == STATUS_QUEUED
        ]
        if not queued:
            return None
        record = min(
            queued,
            key=lambda r: (r.get("sequence", r.get("created_at", 0)), r["delivery_id"]),
        )

        if not lease_ok:
            updated = dict(record)
            updated["reoffer_count"] = int(updated.get("reoffer_count", 0)) + 1
            updated["updated_at"] = now_ns
            updated["status_detail"] = STATUS_DETAIL_LEASE_CONTENDED
            _write(root / QUEUE_DIR / f"{record['delivery_id']}.json", updated)
            _log("reoffer", updated, reoffer_count=updated["reoffer_count"])
            return None

        updated = dict(record)
        updated["status"] = STATUS_RUNNING
        updated["attempts"] = int(updated.get("attempts", 0)) + 1
        updated["claimed_at"] = now_ns
        updated["updated_at"] = now_ns
        updated["queue_position"] = None
        log = list(updated.get("attempts_log") or [])
        log.append(
            {
                "attempt": updated["attempts"],
                "started_at": now_ns,
                "ended_at": None,
                "status": STATUS_RUNNING,
                "reason": None,
            }
        )
        updated["attempts_log"] = log
        os.replace(
            root / QUEUE_DIR / f"{record['delivery_id']}.json",
            root / CLAIMED_DIR / f"{record['delivery_id']}.json",
        )
        _write(root / CLAIMED_DIR / f"{record['delivery_id']}.json", updated)
        _fsync_dir(root / QUEUE_DIR)
        _log("claimed", updated, attempt=updated["attempts"])
        return dict(updated)


def requeue(
    home: str | os.PathLike[str], delivery_id: str, *, now_ns: int | None = None
) -> dict[str, Any]:
    """Return a claimed record to ``queued``.

    ``attempts`` is unchanged; ``reoffer_count`` increments. Already-queued
    records are returned untouched so the call is idempotent.
    """
    key_id = validate_delivery_id(delivery_id)
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    with _locked(home) as root:
        queued_path = root / QUEUE_DIR / f"{key_id}.json"
        existing = _read(queued_path)
        if existing is not None:
            return dict(existing)

        claimed_path = root / CLAIMED_DIR / f"{key_id}.json"
        record = _read(claimed_path)
        if record is None:
            raise FileNotFoundError(f"no claimed record for {key_id}")

        updated = dict(record)
        updated["status"] = STATUS_QUEUED
        updated["reoffer_count"] = int(updated.get("reoffer_count", 0)) + 1
        updated["claimed_at"] = None
        updated["updated_at"] = now_ns
        updated["status_detail"] = STATUS_DETAIL_LEASE_CONTENDED
        entry = _open_attempt(updated)
        if entry is not None:
            entry["ended_at"] = now_ns
            entry["status"] = STATUS_QUEUED
            entry["reason"] = "requeued"
        updated["sequence"] = int(updated.get("sequence", 0))
        os.replace(claimed_path, queued_path)
        updated["queue_position"] = _queue_position(root, updated) or 1
        _write(queued_path, updated)
        _fsync_dir(root / CLAIMED_DIR)
        _log("requeued", updated, reoffer_count=updated["reoffer_count"])
        return dict(updated)


def requeue_unstarted(
    home: str | os.PathLike[str], delivery_id: str, *, now_ns: int | None = None
) -> dict[str, Any]:
    """Return a claimed record to ``queued`` without charging the attempt.

    The in-call lease-contention path (spec §2.7 step 2, §4.9): the turn was
    never paid for, so the claim's ``attempts += 1`` and its still-open
    ``attempts_log`` entry are rolled back -- contention must never consume an
    attempt. ``reoffer_count`` increments; already-queued records are untouched.
    """
    key_id = validate_delivery_id(delivery_id)
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    with _locked(home) as root:
        queued_path = root / QUEUE_DIR / f"{key_id}.json"
        existing = _read(queued_path)
        if existing is not None:
            return dict(existing)

        claimed_path = root / CLAIMED_DIR / f"{key_id}.json"
        record = _read(claimed_path)
        if record is None:
            raise FileNotFoundError(f"no claimed record for {key_id}")

        updated = dict(record)
        updated["attempts"] = max(0, int(updated.get("attempts", 0)) - 1)
        log = list(updated.get("attempts_log") or [])
        if log and log[-1].get("ended_at") is None:
            log.pop()
        updated["attempts_log"] = log
        updated["status"] = STATUS_QUEUED
        updated["reoffer_count"] = int(updated.get("reoffer_count", 0)) + 1
        updated["claimed_at"] = None
        updated["status_detail"] = STATUS_DETAIL_LEASE_CONTENDED
        updated["updated_at"] = now_ns
        updated["sequence"] = int(updated.get("sequence", 0))
        os.replace(claimed_path, queued_path)
        updated["queue_position"] = _queue_position(root, updated) or 1
        _write(queued_path, updated)
        _fsync_dir(root / CLAIMED_DIR)
        _log("requeued", updated, reoffer_count=updated["reoffer_count"])
        return dict(updated)


def settle(
    home: str | os.PathLike[str],
    delivery_id: str,
    *,
    status: str,
    reply: str | None = "",
    error: str | None = "",
    reason: str | None = "",
    status_detail: str | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Move a claimed record to ``settled/`` with a terminal status."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"status must be terminal, got {status!r}")
    key_id = validate_delivery_id(delivery_id)
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    with _locked(home) as root:
        settled_path = root / SETTLED_DIR / f"{key_id}.json"
        existing = _read(settled_path)
        if existing is not None:
            if existing.get("status") != status:
                raise ValueError(
                    f"{key_id} already settled as {existing.get('status')!r}"
                )
            return dict(existing)

        claimed_path = root / CLAIMED_DIR / f"{key_id}.json"
        record = _read(claimed_path)
        if record is None:
            raise FileNotFoundError(f"no claimed record for {key_id}")

        updated = dict(record)
        updated["status"] = status
        updated["updated_at"] = now_ns
        updated["status_detail"] = status_detail
        updated["queue_position"] = None
        if reply not in (None, ""):
            updated["reply"] = reply
        if error not in (None, ""):
            updated["error"] = error
        if reason not in (None, ""):
            updated["reason"] = reason
        entry = _open_attempt(updated)
        if entry is not None:
            entry["ended_at"] = now_ns
            entry["status"] = status
            entry["reason"] = reason or None
        os.replace(claimed_path, settled_path)
        _write(settled_path, updated)
        _fsync_dir(root / CLAIMED_DIR)
        _log("settled", updated, attempt=updated.get("attempts"))
        return dict(updated)


def mark_expired(
    home: str | os.PathLike[str], delivery_id: str, *, now_ns: int | None = None
) -> dict[str, Any]:
    """Settle an over-age ``queued`` record as ``expired``."""
    key_id = validate_delivery_id(delivery_id)
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    with _locked(home) as root:
        queued_path = root / QUEUE_DIR / f"{key_id}.json"
        record = _read(queued_path)
        if record is None:
            settled = _read(root / SETTLED_DIR / f"{key_id}.json")
            if settled is not None:
                return dict(settled)
            raise FileNotFoundError(f"no queued record for {key_id}")
        updated = dict(record)
        updated["status"] = STATUS_EXPIRED
        updated["reason"] = REASON_QUEUED_EXPIRED
        updated["status_detail"] = expired_detail()
        updated["updated_at"] = now_ns
        updated["queue_position"] = None
        os.replace(queued_path, root / SETTLED_DIR / f"{key_id}.json")
        _write(root / SETTLED_DIR / f"{key_id}.json", updated)
        _fsync_dir(root / QUEUE_DIR)
        return dict(updated)


def mark_contended(
    home: str | os.PathLike[str],
    delivery_id: str,
    *,
    status_detail: str | None = None,
    now_ns: int | None = None,
) -> dict[str, Any] | None:
    """Stamp an observed contended turn slot onto a still-queued delivery.

    Admission enqueues before it probes the turn lock, so the receipt's
    ``status_detail`` is only known once the synchronous window closes.
    Contention is never an attempt: ``attempts`` and ``reoffer_count`` are left
    alone. Returns the updated record, or ``None`` when it is no longer queued.
    """
    key_id = validate_delivery_id(delivery_id)
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    with _locked(home) as root:
        path = root / QUEUE_DIR / f"{key_id}.json"
        record = _read(path)
        if record is None:
            return None
        if record.get("status") != STATUS_QUEUED:
            return dict(record)
        record["status_detail"] = status_detail or STATUS_DETAIL_TARGET_BUSY
        record["updated_at"] = now_ns
        _write(path, record)
        return dict(record)


def reoffer_ambiguous(
    home: str | os.PathLike[str], delivery_id: str, *, now_ns: int | None = None
) -> dict[str, Any] | None:
    """Re-offer a settled ``ambiguous`` record, once, under the same delivery_id.

    The crash-mid-turn rule permits exactly ONE recovery replay, only under the
    same idempotency key, and never a second one. Returns the re-queued record, or
    ``None`` when the delivery is not re-offerable (already re-offered once, not
    ``ambiguous``, or unknown). ``attempts`` is incremented by the claim that
    follows, never by the re-offer itself.
    """
    key_id = validate_delivery_id(delivery_id)
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    with _locked(home) as root:
        settled_path = root / SETTLED_DIR / f"{key_id}.json"
        record = _read(settled_path)
        if record is None or record.get("status") != STATUS_AMBIGUOUS:
            return None
        if int(record.get("reoffer_count", 0)) >= 1:
            return None
        updated = dict(record)
        updated["status"] = STATUS_QUEUED
        updated["reoffer_count"] = int(record.get("reoffer_count", 0)) + 1
        updated["status_detail"] = STATUS_DETAIL_TARGET_BUSY
        updated["updated_at"] = now_ns
        entry = _open_attempt(updated)
        if entry is not None:
            entry["ended_at"] = now_ns
            entry["status"] = STATUS_QUEUED
            entry["reason"] = "recovery_replay"
        updated["queue_position"] = 0
        queued_path = root / QUEUE_DIR / f"{key_id}.json"
        _write(queued_path, updated)
        settled_path.unlink(missing_ok=True)
        _fsync_dir(root / SETTLED_DIR)
        updated["queue_position"] = _queue_position(root, updated) or 1
        _write(queued_path, updated)
        return dict(updated)


def read_record(
    home: str | os.PathLike[str], delivery_id: str
) -> dict[str, Any] | None:
    """Read a record from whichever lifecycle directory holds it."""
    key_id = validate_delivery_id(delivery_id)
    with _locked(home) as root:
        return _find(root, key_id)


def cleanup_bot_delivery_queue(max_age_hours: float | None = None) -> int:
    """Hourly housekeeping hook (spec 3.1 trigger 4): sweep THIS home's queue.

    ``max_age_hours`` exists only for signature parity with the other
    ``cleanup_*`` chores; the queue's own TTL governs expiry.
    """
    del max_age_hours
    try:
        from tools.bot_mode_probe import _default_home

        return sweep_delivery_queue(Path(_default_home()))
    except Exception:  # pragma: no cover - housekeeping must not raise
        logger.debug("bot_delivery sweep failed", exc_info=True)
        return 0


def sweep_delivery_queue(
    home: str | os.PathLike[str],
    *,
    now_ns: int | None = None,
    slot_held_fn: Any = None,
) -> int:
    """Recover orphaned claims, then expire over-age records.

    Runs hourly and on every 30s drainer tick (VERIFICATION D1 trigger 4).
    Returns the number of records it acted on.

    An over-age record is expired only while its target's turn slot is FREE. A
    target mid-turn (a desktop session can hold the slot for a whole lease wait)
    must not have its inbound delivery expire underneath it: the slot holder is
    exactly the turn that will drain it next.
    """
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    ttl = queue_ttl_seconds()
    held = slot_held if slot_held_fn is None else slot_held_fn
    actions = 0
    with _locked(home) as root:
        # 1. a claim renamed the file but crashed before rewriting it
        for record in _list(root, CLAIMED_DIR):
            if record.get("status") != STATUS_QUEUED:
                continue
            delivery_id = record["delivery_id"]
            os.replace(
                root / CLAIMED_DIR / f"{delivery_id}.json",
                root / QUEUE_DIR / f"{delivery_id}.json",
            )
            _write(root / QUEUE_DIR / f"{delivery_id}.json", record)
            actions += 1

        # 2. expire records that waited past the TTL
        if ttl > 0:
            for record in _list(root, QUEUE_DIR):
                if record.get("status") != STATUS_QUEUED:
                    continue
                if queued_seconds(record, now_ns=now_ns) <= ttl:
                    continue
                if held(home, record.get("target_profile") or ""):
                    _log("age_kept_slot_held", record)
                    continue
                updated = dict(record)
                updated["status"] = STATUS_EXPIRED
                updated["reason"] = REASON_QUEUED_EXPIRED
                updated["status_detail"] = expired_detail(ttl)
                updated["updated_at"] = now_ns
                updated["queue_position"] = None
                path = root / QUEUE_DIR / f"{updated['delivery_id']}.json"
                os.replace(path, root / SETTLED_DIR / f"{updated['delivery_id']}.json")
                _write(root / SETTLED_DIR / f"{updated['delivery_id']}.json", updated)
                _log("expired", updated, reason=REASON_QUEUED_EXPIRED)
                actions += 1
    _log("swept", None, actions=actions, ttl_seconds=ttl)
    return actions


def queued_target_profiles(home: str | os.PathLike[str]) -> list[str]:
    """Distinct target profiles that have a queued delivery in this home's queue.

    Trigger 3 has to ask "what can be drained?" without a profile in hand
    (VERIFICATION D1).
    """
    with _locked(home) as root:
        profiles: list[str] = []
        for record in _list(root, QUEUE_DIR):
            profile = str(record.get("target_profile") or "")
            if profile and profile not in profiles:
                profiles.append(profile)
    return profiles


def check_capacity(
    home: str | os.PathLike[str],
    *,
    target_profile: str,
    sender_profile: str,
) -> None:
    """Raise :class:`QueueFullError` if one more delivery would exceed a cap.

    Admission checks capacity *before* it reserves an idempotency row, and an
    overflow may take no reservation at all -- so the check has to be reachable
    without :func:`admit`'s write.
    """
    with _locked(home) as root:
        if _pending(root, target_profile=target_profile) >= max_per_profile():
            raise QueueFullError(
                target_profile,
                limit_kind=LIMIT_PER_PROFILE,
                limit=max_per_profile(),
            )
        if (
            _pending(root, target_profile=target_profile, sender_profile=sender_profile)
            >= max_per_sender()
        ):
            raise QueueFullError(
                target_profile,
                limit_kind=LIMIT_PER_SENDER,
                limit=max_per_sender(),
            )


def queue_depth(home: str | os.PathLike[str], target_profile: str | None = None) -> int:
    """Number of not-yet-terminal records (optionally for one profile)."""
    with _locked(home) as root:
        return _pending(root, target_profile=target_profile)


# --------------------------------------------------------------------------
# envelope construction (§1.2/§1.3)
# --------------------------------------------------------------------------
def retryable(envelope: dict[str, Any]) -> bool:
    """Whether the *sender* may retry this failed send as-is.

    Derived from the envelope reason only; unrelated to ``retry_action()``
    (the receiver's own agent auto-retry), which must never be derived from it.
    """
    if envelope.get("result") != RESULT_FAILED:
        return False
    return envelope.get("reason") in RETRYABLE_REASONS


def build_envelope(
    record: dict[str, Any],
    *,
    send_id: str | None = None,
    busy: bool | None = None,
    waited_seconds: float = 0.0,
    replayed: bool = False,
) -> dict[str, Any]:
    """Build the 23-field sender-visible envelope for a durable record.

    ``busy`` defaults to what the record's own ``status_detail`` asserts, so a
    replayed envelope cannot contradict the cause it prints (VERIFICATION D7)
    and a receipt for a free-slot backlog never claims the slot was held (D2).
    """
    status = record.get("status") or STATUS_QUEUED
    if busy is None:
        busy = busy_from_detail(record)
    result = _result_for_status(status)
    reason = record.get("reason")
    is_retryable = result == RESULT_FAILED and reason in RETRYABLE_REASONS
    retry_after: float | None = None
    if is_retryable and reason == "queue_full":
        retry_after = QUEUE_FULL_RETRY_AFTER_SECONDS
    elif is_retryable:
        retry_after = record.get("retry_after_seconds")

    if record.get("status_detail") == STATUS_DETAIL_LIVE_OWNER:
        detail: str | None = DETAIL_HELD_LIVE
    elif status == STATUS_EXPIRED:
        detail = expired_detail()
    elif status == STATUS_AMBIGUOUS:
        detail = unknown_detail(record.get("idempotency_key"))
    elif reason == "queue_full":
        # Report the cap that ACTUALLY refused the send (VERIFICATION D6): the
        # per-sender cap (8) used to be reported as the per-profile cap (32), so an
        # operator would raise the wrong limit. Absent the fired limit, the
        # per-profile reading is the historical one.
        detail = queue_full_detail(
            limit_kind=str(record.get("limit_kind") or LIMIT_PER_PROFILE),
            limit=record.get("limit"))
    elif result == RESULT_RECEIPT:
        detail = DETAIL_RECEIPT
    else:
        detail = None

    envelope = {
        "object": OBJECT_NAME,
        "result": result,
        "status": status,
        "status_detail": record.get("status_detail"),
        "send_id": send_id or uuid.uuid4().hex,
        "delivery_id": record.get("delivery_id"),
        "idempotency_key": record.get("idempotency_key"),
        "replayed": bool(replayed),
        "peer": record.get("peer"),
        "profile": record.get("target_profile"),
        "session_id": record.get("target_session_id"),
        "queue_position": (
            record.get("queue_position") if status == STATUS_QUEUED else None
        ),
        "attempts": int(record.get("attempts", 0)),
        "queued_seconds": round(queued_seconds(record), 3),
        "waited_seconds": float(waited_seconds),
        "busy": bool(busy),
        "retryable": is_retryable,
        "retry_after_seconds": retry_after,
        "reply": record.get("reply") if result == RESULT_DELIVERED else None,
        "error": (
            record.get("error")
            if result in (RESULT_FAILED, RESULT_REFUSED)
            else None
        ),
        "reason": (
            reason if result in (RESULT_FAILED, RESULT_REFUSED) else None
        ),
        "detail": detail,
        "at": int(time.time()),
    }
    return {field: envelope[field] for field in ENVELOPE_FIELDS}


def build_receipt(
    record: dict[str, Any],
    *,
    busy: bool | None = None,
    replayed: bool = False,
    waited_seconds: float | None = None,
) -> dict[str, Any]:
    """The acceptance receipt for a queued delivery (``result: receipt``).

    ``busy`` is the OBSERVED slot state (VERIFICATION D2): a receipt handed out
    while the target's turn slot is free must not report it as held.
    """
    waited = receipt_after_seconds() if waited_seconds is None else waited_seconds
    envelope = build_envelope(
        record,
        send_id=uuid.uuid4().hex,
        busy=busy,
        waited_seconds=waited,
        replayed=replayed,
    )
    envelope["result"] = RESULT_RECEIPT
    envelope["detail"] = build_notification_text(record)
    if record.get("status") != STATUS_RUNNING:
        envelope["attempts"] = 0
    return envelope


def build_delivered_envelope(
    record: dict[str, Any], *, replayed: bool = False, waited_seconds: float = 0.0
) -> dict[str, Any]:
    return build_envelope(
        record, busy=False, waited_seconds=waited_seconds, replayed=replayed
    )
