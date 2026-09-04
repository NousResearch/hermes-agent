"""Group relay — gateway ↔ Desktop file plumbing for relaying into Desktop rooms.

Desktop-coordinated Group Chats (``apps/desktop/src/plugins/hermes-bots``)
keep their orchestration state in the Desktop's plugin storage and start
member rounds only from the renderer (``sendToGroupChat``). Nothing outside
the app can start a round, so ``hermes group send`` reaches those rooms the
same way cross-connection DMs reach other machines (``tools/bot_relay.py``):
plain files under the gateway's HERMES root that the Desktop drains.

Layout — ``<root>/group_relay/``:

- ``outbox/<id>.json``  — envelopes queued by the CLI. The Desktop claims
  them (atomic rename into ``claimed/``) and calls ``sendToGroupChat`` on
  the user's behalf.
- ``replies/<id>.jsonl`` — append-only progress lines written by the Desktop
  as the round runs, so the CLI's ``--wait`` can stream partial output:
  ``accepted`` (thread minted) → ``reply`` (one per committed member
  message) → ``done`` (``settled|capped|cancelled|timeout``) or ``error``.

This is deliberately a sibling of ``bot_relay`` rather than an extension of
it: the DM drain is cross-connection-only and its envelopes carry a
different contract. Deleting this module and its callers removes the
feature cleanly. Helpers never raise except for the documented validation
errors; ids are 32 hex chars so paths cannot be steered.
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
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

RELAY_DIR_NAME = "group_relay"
OUTBOX_DIR = "outbox"
CLAIMED_DIR = "claimed"
REPLIES_DIR = "replies"

# Reply files hold room transcript text; sweep on the same cadence as the DM
# relay (six hours) via the gateway housekeeping loop.
STALE_AFTER_SECONDS = 6 * 3600
# Unclaimed envelopes older than this are expired at drain time with an
# ``error`` line so a waiting CLI resolves instead of hanging until timeout.
DEFAULT_ENVELOPE_TTL_SECONDS = 15 * 60

MAX_TEXT_CHARS = 64 * 1024
MAX_LABEL_CHARS = 200

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_LINE_KINDS = frozenset({"receipt", "accepted", "reply", "done", "error"})
_IDEMPOTENCY_FIELDS = ("room_id", "room_name", "thread", "text")
DONE_STATUSES = frozenset({"settled", "capped", "cancelled", "timeout"})


class GroupRelayError(ValueError):
    """Validation failure on an envelope or reply line."""


class GroupRelayConflictError(GroupRelayError):
    """An ``event_key`` was reused with different content."""


def envelope_id_for_key(event_key: str) -> str:
    """Deterministic 32-hex envelope id for a caller retry key."""
    return hashlib.sha256(f"group-relay:{event_key}".encode("utf-8")).hexdigest()[:32]


def _receipt_from_replies(base: Path, envelope_id: str) -> dict[str, Any] | None:
    """The immutable idempotency receipt recorded as the reply file's first line."""
    path = base / REPLIES_DIR / f"{envelope_id}.jsonl"
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            first = handle.readline()
        parsed = json.loads(first.decode("utf-8")) if first.strip() else None
    except (OSError, ValueError):
        return None
    if isinstance(parsed, dict) and parsed.get("kind") == "receipt":
        return parsed
    return None


def _find_envelope(base: Path, envelope_id: str) -> dict[str, Any] | None:
    for sub in (OUTBOX_DIR, CLAIMED_DIR):
        path = base / sub / f"{envelope_id}.json"
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(loaded, dict):
                return loaded
    return None


def relay_root(root: Path | str) -> Path:
    return Path(root) / RELAY_DIR_NAME


def _ensure_dirs(root: Path | str) -> Path:
    base = relay_root(root)
    for sub in (OUTBOX_DIR, CLAIMED_DIR, REPLIES_DIR):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def _validate_id(value: Any) -> str:
    safe = str(value or "").strip()
    if not _ID_RE.match(safe):
        raise GroupRelayError(f"invalid group relay id: {value!r}")
    return safe


def _atomic_write_json(directory: Path, name: str, payload: dict, *, prefix: str) -> Path:
    path = directory / name
    fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=prefix, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    os.replace(tmp, path)
    return path


# ── envelopes (CLI → Desktop) ────────────────────────────────────────────────


def enqueue(
    root: Path | str,
    *,
    room_id: str,
    room_name: str,
    text: str,
    from_profile: str,
    label: str,
    thread: str | None = None,
    event_key: str | None = None,
) -> dict[str, Any]:
    """Queue one relay request for the Desktop. Returns the envelope.

    With ``event_key`` the envelope id is deterministic: repeating the same
    key with identical room/thread/text returns the already-queued (or
    already-claimed) envelope instead of queueing a duplicate; the same key
    with different content raises :class:`GroupRelayConflictError`. Without
    a key every call is a fresh envelope.
    """
    room_id = str(room_id or "").strip()
    room_name = str(room_name or "").strip()
    body = str(text or "").strip()
    who = str(label or "").strip()
    if not room_id and not room_name:
        raise GroupRelayError("room_id or room_name is required")
    if not body:
        raise GroupRelayError("text is required")
    if len(body) > MAX_TEXT_CHARS:
        raise GroupRelayError(f"text too long ({len(body)} > {MAX_TEXT_CHARS} chars)")
    if len(who) > MAX_LABEL_CHARS:
        raise GroupRelayError(f"label too long ({len(who)} > {MAX_LABEL_CHARS} chars)")
    base = _ensure_dirs(root)
    key = str(event_key or "").strip() or None
    normalized_thread = (str(thread).strip() or None) if thread is not None else None
    envelope = {
        "id": envelope_id_for_key(key) if key else uuid.uuid4().hex,
        "created_at": int(time.time()),
        "from_profile": str(from_profile or "default"),
        "label": who,
        "room_id": room_id,
        "room_name": room_name,
        "thread": normalized_thread,
        "text": body,
        **({"event_key": key} if key else {}),
    }
    if key:
        conflict = GroupRelayConflictError(
            f"event key {key!r} was already used with different content; pick a new key"
        )
        existing = _find_envelope(base, envelope["id"])
        if existing is not None:
            if any(existing.get(field) != envelope[field] for field in _IDEMPOTENCY_FIELDS):
                raise conflict
            return existing
        receipt = _receipt_from_replies(base, envelope["id"])
        if receipt is not None:
            # The receipt at the head of the reply file is the durable record
            # of what this key sent; it outlives outbox/ and claimed/.
            if any(receipt.get(field) != envelope[field] for field in _IDEMPOTENCY_FIELDS):
                raise conflict
            if _desktop_took_delivery(base, envelope["id"]):
                return envelope
            # Receipt exists but no envelope and the Desktop never saw it: a
            # crash landed between the receipt and the outbox write (below).
            # Recreate the outbox entry — idempotently, same id — so the
            # message is not lost.
        else:
            # First enqueue for this key: pin the receipt BEFORE the envelope
            # is visible so conflict detection survives every later sweep.
            append_reply_line(
                root,
                envelope["id"],
                {"kind": "receipt", **{field: envelope[field] for field in _IDEMPOTENCY_FIELDS}},
            )
    _atomic_write_json(base / OUTBOX_DIR, f"{envelope['id']}.json", envelope, prefix=".env-")
    return envelope


def _desktop_took_delivery(base: Path, envelope_id: str) -> bool:
    """True once the reply file shows the Desktop acted on the envelope
    (``accepted``/``done``/``error``) — the only states where suppressing a
    requeue cannot lose the message."""
    path = base / REPLIES_DIR / f"{envelope_id}.jsonl"
    try:
        with open(path, "rb") as handle:
            for raw in handle:
                try:
                    line = json.loads(raw.decode("utf-8"))
                except ValueError:
                    continue
                if isinstance(line, dict) and line.get("kind") in ("accepted", "reply", "done", "error"):
                    return True
    except OSError:
        return False
    return False


def pending_count(root: Path | str, *, older_than_seconds: float = 0.0) -> int:
    """Unclaimed envelopes (optionally only those older than N seconds)."""
    outbox = relay_root(root) / OUTBOX_DIR
    if not outbox.is_dir():
        return 0
    cutoff = time.time() - max(0.0, older_than_seconds)
    count = 0
    for path in outbox.glob("*.json"):
        try:
            if older_than_seconds <= 0 or path.stat().st_mtime < cutoff:
                count += 1
        except OSError:
            continue
    return count


def claim_pending(root: Path | str, *, ttl_seconds: int | None = None) -> list[dict[str, Any]]:
    """Drain the outbox (atomic rename → claimed/). Expired envelopes get an
    ``error`` reply line and are dropped rather than delivered."""
    base = _ensure_dirs(root)
    _sweep_stale(base)
    ttl = DEFAULT_ENVELOPE_TTL_SECONDS if ttl_seconds is None else int(ttl_seconds)
    now = time.time()
    out: list[dict[str, Any]] = []
    for path in sorted((base / OUTBOX_DIR).glob("*.json")):
        envelope: dict[str, Any] | None = None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            envelope = None
        if ttl > 0 and isinstance(envelope, dict):
            created = float(envelope.get("created_at") or path.stat().st_mtime)
            if now - created > ttl:
                try:
                    append_reply_line(
                        root,
                        envelope.get("id"),
                        {
                            "kind": "error",
                            "reason": "queued_expired",
                            "error": (
                                f"relay to group {envelope.get('room_name') or envelope.get('room_id')} "
                                f"expired after {ttl}s waiting for the Desktop to drain it — "
                                "it was NOT delivered. Is the Hermes app open?"
                            ),
                        },
                    )
                except GroupRelayError:
                    pass
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
        claimed = base / CLAIMED_DIR / path.name
        try:
            os.replace(path, claimed)
            if envelope is None:
                envelope = json.loads(claimed.read_text(encoding="utf-8"))
            if isinstance(envelope, dict):
                out.append(envelope)
        except (OSError, ValueError):
            continue
    return out


# ── reply lines (Desktop → CLI) ──────────────────────────────────────────────


def _reply_path(root: Path | str, envelope_id: Any) -> Path:
    return _ensure_dirs(root) / REPLIES_DIR / f"{_validate_id(envelope_id)}.jsonl"


def append_reply_line(root: Path | str, envelope_id: Any, line: dict[str, Any]) -> Path:
    """Append one progress line. ``line['kind']`` ∈ accepted|reply|done|error."""
    if not isinstance(line, dict):
        raise GroupRelayError("reply line must be an object")
    kind = str(line.get("kind") or "")
    if kind not in _LINE_KINDS:
        raise GroupRelayError(f"invalid reply line kind: {kind!r}")
    if kind == "done" and str(line.get("status") or "") not in DONE_STATUSES:
        raise GroupRelayError(f"invalid done status: {line.get('status')!r}")
    if kind == "reply" and not str(line.get("text") or "").strip():
        raise GroupRelayError("reply line requires text")
    path = _reply_path(root, envelope_id)
    record = {"id": _validate_id(envelope_id), "at": int(time.time()), **line}
    encoded = json.dumps(record, ensure_ascii=False) + "\n"
    # O_APPEND writes below PIPE_BUF are atomic on POSIX; lines are small.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, encoded.encode("utf-8"))
    finally:
        os.close(fd)
    return path


def read_reply_lines(root: Path | str, envelope_id: Any, *, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Return complete lines after byte ``offset`` and the new offset.

    A partially written trailing line (no newline yet) is left for the next
    read so a concurrent append never yields a truncated record.
    """
    path = relay_root(root) / REPLIES_DIR / f"{_validate_id(envelope_id)}.jsonl"
    if not path.exists():
        return [], offset
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, int(offset)))
            chunk = handle.read()
    except OSError:
        return [], offset
    lines: list[dict[str, Any]] = []
    consumed = 0
    while True:
        newline = chunk.find(b"\n", consumed)
        if newline < 0:
            break  # trailing partial line: leave for the next read
        raw = chunk[consumed:newline]
        consumed = newline + 1
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(parsed, dict):
            lines.append(parsed)
    return lines, int(offset) + consumed


# ── housekeeping ─────────────────────────────────────────────────────────────


def _sweep_stale(base: Path, *, now: float | None = None) -> int:
    cutoff = (time.time() if now is None else now) - STALE_AFTER_SECONDS
    removed = 0
    for sub, pattern in ((CLAIMED_DIR, "*.json"), (OUTBOX_DIR, "*.json"), (REPLIES_DIR, "*.jsonl")):
        try:
            for path in (base / sub).glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
        except OSError:
            continue
    return removed


def cleanup_group_relay_artifacts(max_age_hours: float | None = None) -> int:
    """Housekeeping hook (same contract as ``cleanup_bot_relay_artifacts``)."""
    del max_age_hours
    try:
        home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
        root = home.parent.parent if home.parent.name == "profiles" else home
        base = relay_root(root)
        if not base.is_dir():
            return 0
        return _sweep_stale(base)
    except Exception:
        logger.debug("group_relay artifact sweep failed", exc_info=True)
        return 0


def gateway_root() -> Path:
    """Machine root (not the profile home) — the store is profile-independent."""
    home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
    return home.parent.parent if home.parent.name == "profiles" else home
