#!/usr/bin/env python3
"""Write-approval gate + pending store for memory and skill writes.

Background
----------
The agent writes to two persistent stores that survive across sessions:

  * **memory** — MEMORY.md / USER.md, small (~200 char) declarative entries
  * **skills** — SKILL.md + supporting files, potentially huge (10-100 KB)

Both stores are written from two origins:

  * **foreground** — a normal agent turn (user is present / chatting)
  * **background_review** — the self-improvement review fork that runs after a
    turn and autonomously decides what to save (the source of the
    "wrong assumptions" users complained about)

This module lets the user gate those writes per-subsystem with a boolean
``write_approval``:

  * ``false`` (default) — write freely (the pre-gate behaviour)
  * ``true``            — require approval: do not commit the write; either
    prompt inline (memory, interactive CLI only) or **stage** it to a pending
    store and surface it for the user to approve or reject out-of-band

The size asymmetry between memory and skills is real and unavoidable: a memory
entry can be reviewed inline in a chat bubble; a 100 KB SKILL.md cannot. So
the gate stages BOTH to disk, but review affordances differ by subsystem
(see ``hermes_cli`` slash handlers): memory shows full content, skills show
metadata + a one-line gist + a ``diff`` escape hatch (CLI/dashboard/file).

Staging is mandatory for background-origin writes (a daemon thread cannot
block on an interactive prompt) and for gateway sessions (no inline prompt
channel — review happens via ``/memory pending``). Foreground CLI memory
writes prompt inline via the dangerous-command approval callback; skill
writes always stage (too big to eyeball mid-loop).

Pending records live under ``<HERMES_HOME>/pending/{memory,skills}/<id>.json``
so they survive process restarts and can be reviewed from CLI, gateway, or the
web dashboard.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Subsystem identifiers
MEMORY = "memory"
SKILLS = "skills"
_SUBSYSTEMS = (MEMORY, SKILLS)

# Config key (per subsystem). A single boolean: the approval gate is OFF by
# default (writes flow freely, the pre-gate behaviour), and ON means stage /
# prompt every write for the user's approval. There is intentionally no third
# "block all writes" state — to disable a subsystem entirely use its own
# enable flag (e.g. ``memory.memory_enabled: false``).
CONFIG_KEY = "write_approval"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def write_approval_enabled(subsystem: str) -> bool:
    """Return whether the approval gate is enabled for ``subsystem``.

    Reads ``<subsystem>.write_approval`` from config.yaml. Defaults to
    ``False`` (gate off — writes flow freely) for any unset / invalid value so
    existing installs keep their current behaviour until the user opts in.
    """
    if subsystem not in _SUBSYSTEMS:
        return False
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        raw = cfg_get(cfg, subsystem, CONFIG_KEY, default=False)
    except Exception:
        return False
    return _normalize_enabled(raw)


def _normalize_enabled(value: Any) -> bool:
    """Coerce a config value to a bool. Default (unknown) is False (gate off).

    Accepts real bools and the usual truthy/falsey strings. YAML 1.1 parses
    bare ``on``/``off``/``yes``/``no`` as bools already, so the string branch
    is mostly for hand-edited configs.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "1", "approve", "enabled"}
    return False


# ---------------------------------------------------------------------------
# Pending store (file-backed)
# ---------------------------------------------------------------------------

class PendingStoreError(RuntimeError):
    """Raised when a pending record cannot be staged safely."""


_SCHEMA_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PENDING_ID_RE = re.compile(r"^[0-9a-f]{8}$")
_MAX_PENDING_RECORD_BYTES = 8 * 1024 * 1024
_MAX_PENDING_DIRECTORY_ENTRIES = 4096
_ALLOWED_ORIGINS = frozenset({"foreground", "background_review"})
_REQUIRED_SESSION_CONTEXT_FIELDS = (
    "profile",
    "session_id",
    "surface",
    "tool_call_id",
)


def _canonical_payload_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_record_bytes(record: Dict[str, Any]) -> bytes:
    integrity_fields = {
        key: value for key, value in record.items() if key != "record_hash"
    }
    return json.dumps(
        integrity_fields,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _owner_only_permissions_supported() -> bool:
    """Return whether this runtime can enforce the pending-store owner boundary."""
    return (
        os.name != "nt"
        and isinstance(getattr(os, "O_NOFOLLOW", None), int)
        and isinstance(getattr(os, "O_DIRECTORY", None), int)
        and os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.scandir in os.supports_fd
    )


def collect_session_context(
    *, session_id: Optional[str], tool_call_id: Optional[str]
) -> Dict[str, str]:
    """Build audit context from task-local identity plus dispatch IDs."""
    try:
        from gateway.session_context import get_session_var

        profile = get_session_var("HERMES_SESSION_PROFILE", "")
        surface = (
            get_session_var("HERMES_SESSION_PLATFORM", "")
            or get_session_var("HERMES_SESSION_SOURCE", "")
        )
        context_session_id = get_session_var("HERMES_SESSION_ID", "")
    except ImportError:
        profile = ""
        surface = ""
        context_session_id = ""
    if not profile:
        try:
            from hermes_cli.profiles import get_active_profile_name

            profile = get_active_profile_name()
        except Exception:
            profile = "default"
    return {
        "profile": profile or "default",
        "session_id": (session_id or context_session_id or "").strip(),
        "surface": (surface or "").strip(),
        "tool_call_id": (tool_call_id or "").strip(),
    }


def validate_pending_record(record: Dict[str, Any]) -> tuple[bool, str]:
    """Validate the integrity and bound provenance of a v2 pending record."""
    if not isinstance(record, dict):
        return False, "pending record is not an object"
    if record.get("schema_version") != _SCHEMA_VERSION:
        return False, "pending record is not schema v2"
    subsystem = record.get("subsystem")
    if subsystem not in _SUBSYSTEMS:
        return False, "pending record has an invalid subsystem"
    pending_id = record.get("id")
    if not isinstance(pending_id, str) or not _PENDING_ID_RE.fullmatch(pending_id):
        return False, "pending record has an invalid id"
    summary = record.get("summary")
    if not isinstance(summary, str):
        return False, "pending record summary is not text"
    origin = record.get("origin")
    if origin not in _ALLOWED_ORIGINS:
        return False, "pending record has an invalid origin"
    created_at = record.get("created_at")
    if (
        isinstance(created_at, bool)
        or not isinstance(created_at, (int, float))
        or not math.isfinite(created_at)
        or created_at <= 0
    ):
        return False, "pending record has an invalid created_at"
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return False, "pending record payload is not an object"
    action = record.get("action")
    if not isinstance(action, str) or not action or action != payload.get("action"):
        return False, "pending record action does not match its payload"
    try:
        expected_payload_hash = hashlib.sha256(
            _canonical_payload_bytes(payload)
        ).hexdigest()
    except (TypeError, ValueError):
        return False, "pending record payload is not canonical JSON"
    if record.get("payload_hash") != expected_payload_hash:
        return False, "pending record payload hash does not match"
    session_context = record.get("session_context")
    if not isinstance(session_context, dict):
        return False, "pending record has no session context"
    for field in _REQUIRED_SESSION_CONTEXT_FIELDS:
        value = session_context.get(field)
        if not isinstance(value, str) or not value.strip():
            return False, f"pending record session context is missing {field}"
    if subsystem == SKILLS:
        target_hash = record.get("target_tree_pre_image_hash")
        if not isinstance(target_hash, str) or not _SHA256_RE.fullmatch(target_hash):
            return False, "pending skill record has no valid target pre-image hash"
    try:
        expected_record_hash = hashlib.sha256(
            _canonical_record_bytes(record)
        ).hexdigest()
    except (TypeError, ValueError):
        return False, "pending record metadata is not canonical JSON"
    if record.get("record_hash") != expected_record_hash:
        return False, "pending record integrity hash does not match"
    return True, ""


def _validate_owner_only_directory_fd(fd: int, label: str) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        raise PendingStoreError(f"{label} is not a directory")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise PendingStoreError(f"{label} is not owned by the current user")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise PendingStoreError(f"{label} must have mode 0700")


@contextmanager
def _open_pending_directory_fd(subsystem: str, *, create: bool = False):
    """Open the pending store through held no-follow directory descriptors."""
    if subsystem not in _SUBSYSTEMS:
        raise PendingStoreError("invalid pending subsystem")
    if not _owner_only_permissions_supported():
        raise PendingStoreError(
            "owner-only pending-store permissions are not supported on this platform"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    home_fd = pending_fd = subsystem_fd = None
    try:
        home_fd = os.open(get_hermes_home(), flags)
        home_info = os.fstat(home_fd)
        if not stat.S_ISDIR(home_info.st_mode):
            raise PendingStoreError("Hermes home is not a real directory")
        if hasattr(os, "geteuid") and home_info.st_uid != os.geteuid():
            raise PendingStoreError("Hermes home is not owned by the current user")
        if create:
            try:
                os.mkdir("pending", 0o700, dir_fd=home_fd)
            except FileExistsError:
                pass
        pending_fd = os.open("pending", flags, dir_fd=home_fd)
        _validate_owner_only_directory_fd(pending_fd, "pending store directory")
        if create:
            try:
                os.mkdir(subsystem, 0o700, dir_fd=pending_fd)
            except FileExistsError:
                pass
        subsystem_fd = os.open(subsystem, flags, dir_fd=pending_fd)
        _validate_owner_only_directory_fd(
            subsystem_fd, f"pending {subsystem} directory"
        )
        yield subsystem_fd
    except PendingStoreError:
        raise
    except OSError as exc:
        raise PendingStoreError(
            f"pending store path is not a real directory or cannot be opened safely: {exc}"
        ) from exc
    finally:
        for fd in (subsystem_fd, pending_fd, home_fd):
            if fd is not None:
                os.close(fd)


def _write_owner_only_record(
    directory_fd: int, filename: str, record: Dict[str, Any]
) -> None:
    """Atomically publish one 0600 JSON record without overwriting an id."""
    data = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    tmp = f".{filename}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = None
    published = False
    try:
        fd = os.open(tmp, flags, 0o600, dir_fd=directory_fd)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            tmp,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(tmp, dir_fd=directory_fd)
        os.fsync(directory_fd)
        info = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1:
            raise PendingStoreError("pending record was not published safely as 0600")
    except Exception:
        if fd is not None:
            os.close(fd)
        if published:
            try:
                os.unlink(filename, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        try:
            os.unlink(tmp, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise


def stage_write(
    subsystem: str,
    payload: Dict[str, Any],
    *,
    summary: str,
    origin: str,
    session_context: Optional[Dict[str, str]] = None,
    target_tree_pre_image_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a pending write and return a short record describing it.

    Args:
        subsystem: ``memory`` or ``skills``.
        payload: the exact kwargs needed to replay the write when approved
            (e.g. ``{"action": "add", "target": "user", "content": "..."}``
            for memory, or the full ``skill_manage`` kwargs for skills).
        summary: a one-line human-readable description shown in pending lists.
            For skills this is the LLM/heuristic gist; for memory it can be the
            entry text itself.
        origin: ``foreground`` or ``background_review`` — recorded for audit.

    Returns a validated dict with ``id`` and metadata. Any validation, ownership,
    path-safety, size, or disk failure raises ``PendingStoreError`` so the caller
    can fail closed without reporting a staged write that was not persisted.
    """
    pid = uuid.uuid4().hex[:8]
    try:
        payload_bytes = _canonical_payload_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise PendingStoreError(
            "pending record payload is not canonical JSON"
        ) from exc
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    record = {
        "schema_version": _SCHEMA_VERSION,
        "id": pid,
        "subsystem": subsystem,
        "action": payload.get("action", ""),
        "summary": (summary or "").strip(),
        "origin": origin or "foreground",
        "created_at": time.time(),
        "payload": payload,
        "payload_hash": payload_hash,
        "session_context": dict(session_context or {}),
    }
    if target_tree_pre_image_hash is not None:
        record["target_tree_pre_image_hash"] = target_tree_pre_image_hash
    record["record_hash"] = hashlib.sha256(
        _canonical_record_bytes(record)
    ).hexdigest()
    valid, reason = validate_pending_record(record)
    if not valid:
        raise PendingStoreError(reason)
    record_size = len(json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"))
    if record_size > _MAX_PENDING_RECORD_BYTES:
        raise PendingStoreError("pending record exceeds the staging size limit")

    try:
        with _open_pending_directory_fd(subsystem, create=True) as directory_fd:
            _write_owner_only_record(directory_fd, f"{pid}.json", record)
    except PendingStoreError:
        raise
    except Exception as exc:
        logger.error(
            "Failed to stage pending %s write: %s", subsystem, exc, exc_info=True
        )
        raise PendingStoreError(f"failed to stage pending {subsystem} write") from exc
    return record


def _read_pending_record_fd(
    directory_fd: int, subsystem: str, pending_id: str
) -> Optional[tuple[Dict[str, Any], tuple[int, int]]]:
    if not isinstance(pending_id, str) or not _PENDING_ID_RE.fullmatch(pending_id):
        return None
    filename = f"{pending_id}.json"
    try:
        info = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            return None
        if info.st_nlink != 1 or info.st_size > _MAX_PENDING_RECORD_BYTES:
            return None
        if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
            return None
        if stat.S_IMODE(info.st_mode) != 0o600:
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(filename, flags, dir_fd=directory_fd)
        try:
            before = os.fstat(fd)
            if (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
                return None
            chunks = []
            total = 0
            while True:
                chunk = os.read(
                    fd,
                    min(1024 * 1024, _MAX_PENDING_RECORD_BYTES + 1 - total),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_PENDING_RECORD_BYTES:
                    return None
            after = os.fstat(fd)
        finally:
            os.close(fd)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_nlink")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            return None
        record = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    valid, _ = validate_pending_record(record)
    if not valid:
        return None
    if record.get("id") != pending_id or record.get("subsystem") != subsystem:
        return None
    return record, (after.st_dev, after.st_ino)


def _read_pending_record(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    try:
        with _open_pending_directory_fd(subsystem) as directory_fd:
            result = _read_pending_record_fd(directory_fd, subsystem, pending_id)
    except PendingStoreError:
        return None
    return result[0] if result is not None else None


def list_pending(subsystem: str) -> List[Dict[str, Any]]:
    """Return safe, valid v2 records for ``subsystem``, oldest first."""
    records: List[Dict[str, Any]] = []
    try:
        with _open_pending_directory_fd(subsystem) as directory_fd:
            names = []
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > _MAX_PENDING_DIRECTORY_ENTRIES:
                        logger.warning(
                            "Pending %s directory exceeds the entry budget", subsystem
                        )
                        return []
            names.sort()
            for name in names:
                if not name.endswith(".json"):
                    continue
                result = _read_pending_record_fd(
                    directory_fd, subsystem, name[:-5]
                )
                if result is None:
                    logger.warning(
                        "Skipping unsafe or invalid pending record: %s/%s",
                        subsystem,
                        name,
                    )
                    continue
                records.append(result[0])
    except (OSError, PendingStoreError):
        return []
    records.sort(key=lambda record: record.get("created_at", 0))
    return records


def get_pending(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    """Return one safe, valid v2 pending record by id, or None."""
    return _read_pending_record(subsystem, pending_id)


def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a validated pending record. Unsafe/legacy records stay untouched."""
    try:
        with _open_pending_directory_fd(subsystem) as directory_fd:
            result = _read_pending_record_fd(directory_fd, subsystem, pending_id)
            if result is None:
                return False
            _, expected_identity = result
            filename = f"{pending_id}.json"
            current = os.stat(
                filename, dir_fd=directory_fd, follow_symlinks=False
            )
            if (current.st_dev, current.st_ino) != expected_identity:
                return False
            if current.st_nlink != 1 or not stat.S_ISREG(current.st_mode):
                return False
            os.unlink(filename, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return True
    except (OSError, PendingStoreError) as exc:  # pragma: no cover - disk failure path
        logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, exc)
        return False


def _pending_claim_name(pending_id: str) -> str:
    return f".{pending_id}.applying"


def claim_pending(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    """Move a valid record to a non-replayable claim before applying it."""
    try:
        with _open_pending_directory_fd(subsystem) as directory_fd:
            result = _read_pending_record_fd(directory_fd, subsystem, pending_id)
            if result is None:
                return None
            record, expected_identity = result
            filename = f"{pending_id}.json"
            current = os.stat(
                filename, dir_fd=directory_fd, follow_symlinks=False
            )
            if (current.st_dev, current.st_ino) != expected_identity:
                return None
            claim_name = _pending_claim_name(pending_id)
            os.link(
                filename,
                claim_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.unlink(filename, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return record
    except (OSError, PendingStoreError) as exc:
        logger.error("Failed to claim pending %s/%s: %s", subsystem, pending_id, exc)
        return None


def restore_pending_claim(subsystem: str, pending_id: str) -> bool:
    """Restore a failed apply claim to the reviewable queue without overwrite."""
    try:
        with _open_pending_directory_fd(subsystem) as directory_fd:
            claim_name = _pending_claim_name(pending_id)
            info = os.stat(
                claim_name, dir_fd=directory_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            ):
                return False
            os.link(
                claim_name,
                f"{pending_id}.json",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.unlink(claim_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return True
    except (OSError, PendingStoreError) as exc:
        logger.error("Failed to restore pending claim %s/%s: %s", subsystem, pending_id, exc)
        return False


def finalize_pending_claim(subsystem: str, pending_id: str) -> bool:
    """Delete a successfully applied record from its non-replayable quarantine."""
    try:
        with _open_pending_directory_fd(subsystem) as directory_fd:
            claim_name = _pending_claim_name(pending_id)
            info = os.stat(
                claim_name, dir_fd=directory_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
                or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            ):
                return False
            os.unlink(claim_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return True
    except (OSError, PendingStoreError) as exc:
        logger.error("Failed to finalize pending claim %s/%s: %s", subsystem, pending_id, exc)
        return False


def pending_count(subsystem: str) -> int:
    """Count safe, valid pending records for notification badges."""
    return len(list_pending(subsystem))


# ---------------------------------------------------------------------------
# Write origin
# ---------------------------------------------------------------------------

def current_origin() -> str:
    """Return the active write origin: ``foreground`` or ``background_review``.

    Reuses the skill-provenance ContextVar, which the background review fork
    already sets (see ``agent.background_review`` /
    ``AIAgent._spawn_background_review``). Foreground agent turns leave it at
    the default ``foreground``.
    """
    try:
        from tools.skill_provenance import get_current_write_origin
        origin = get_current_write_origin()
        if origin == "assistant_tool":
            return "foreground"
        return origin
    except Exception:
        return "foreground"


def is_background() -> bool:
    return current_origin() == "background_review"


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------

class GateDecision:
    """Result of evaluating the write gate for a single write attempt.

    Exactly one of the boolean flags is True:
      * ``allow``  — proceed with the real write (gate off, or an inline
        approval was granted).
      * ``blocked`` — refuse the write (the user denied an inline approval
        prompt). ``message`` explains why; surface it to the agent.
      * ``stage``  — do not write; the caller should stage the payload via
        ``stage_write`` (gate on, and no inline prompt is available — gateway,
        background review, script, or any skill write). ``message`` is the
        user-facing "staged for approval" note.
    """

    __slots__ = ("allow", "blocked", "stage", "message")

    def __init__(self, *, allow=False, blocked=False, stage=False, message=""):
        self.allow = allow
        self.blocked = blocked
        self.stage = stage
        self.message = message


def evaluate_gate(subsystem: str, *, inline_summary: str = "",
                  inline_detail: str = "") -> GateDecision:
    """Decide what to do with a pending write for ``subsystem``.

    Args:
        subsystem: ``memory`` or ``skills``.
        inline_summary: short description used as the inline approval prompt
            header (memory foreground path only).
        inline_detail: full content shown in the inline prompt (memory entries
            are small; skills never take the inline path).

    Decision matrix:
        gate off (default)                    → allow (writes flow freely)
        gate on, memory + interactive CLI     → inline approve/deny prompt
        gate on, memory + gateway/script/bg   → stage
        gate on, skills (any origin)          → stage (too big to review inline)

    Note: there is no config-driven "blocked" outcome — the gate only ever
    delays a write for approval, never silently refuses it. ``blocked`` is
    still produced when the user *actively denies* an inline prompt.
    """
    if not write_approval_enabled(subsystem):
        return GateDecision(allow=True)

    background = is_background()

    # Skills always stage — a SKILL.md is too large to review inline, and a
    # background skill write happens in a daemon thread with no user present.
    if subsystem == SKILLS or background:
        where = "/skills pending" if subsystem == SKILLS else "/memory pending"
        return GateDecision(
            stage=True,
            message=(
                f"Staged for approval ({subsystem}.write_approval is on). "
                f"Not yet saved — review with {where}."
            ),
        )

    # Memory + foreground: if an interactive approval channel exists (a CLI
    # approval callback registered on this thread), prompt inline — entries
    # are small enough to show in full. Otherwise (gateway, script, batch,
    # no listener) stage instead of forcing a blind deny.
    if _interactive_approval_available():
        granted = _prompt_inline_memory_approval(inline_summary, inline_detail)
        if granted is True:
            return GateDecision(allow=True)
        if granted is False:
            return GateDecision(
                blocked=True,
                message="Memory write denied by user. The change was not saved.",
            )
        # granted is None → prompt failed; fall through to staging.

    return GateDecision(
        stage=True,
        message=(
            "Staged for approval (memory.write_approval is on). "
            "Not yet saved — review with /memory pending."
        ),
    )


def _interactive_approval_available() -> bool:
    """True when a foreground memory write can be approved inline.

    Inline prompting requires a per-thread approval callback registered by the
    interactive CLI (``tools.terminal_tool.set_approval_callback``). Every
    other surface stages instead:

    * **Gateway/API sessions** — the dangerous-command ``/approve`` round-trip
      lives in the pending-approval queue (``submit_pending`` +
      ``_await_gateway_decision``), which ``prompt_dangerous_approval`` never
      reaches; trying to prompt from a gateway session would hit the
      ``input()`` fallback and silently deny. Staging gives the user a real
      review affordance (``/memory pending``) instead.
    * Scripts, cron, and background threads — no user present.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
        return _get_approval_callback() is not None
    except Exception:
        return False


def _prompt_inline_memory_approval(summary: str, detail: str) -> Optional[bool]:
    """Prompt the user inline to approve a memory write.

    Returns True (approved), False (denied), or None (no interactive prompt
    available / prompt failed → caller should stage instead).

    Reuses the per-thread CLI approval callback registered for dangerous
    commands (``tools.terminal_tool.set_approval_callback``). The callback is
    invoked directly — NOT via ``prompt_dangerous_approval`` — because that
    wrapper falls back to ``input()`` (deadlock-prone under prompt_toolkit,
    see #15216) and converts callback errors into a silent deny; here a
    failed prompt must stage the write instead.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
    except Exception:
        return None

    callback = _get_approval_callback()
    if callback is None:
        # No interactive channel on this thread — stage rather than risk the
        # input() fallback (deadlock under prompt_toolkit, EOF-deny in tests).
        return None

    header = summary.strip() or "Save to memory?"
    body = detail.strip()
    description = f"Save to memory: {header}"
    command = body if body else header
    # Invoke the callback directly instead of via prompt_dangerous_approval:
    # that wrapper swallows callback exceptions into "deny", which would
    # silently refuse the write. Direct invocation lets a crashed prompt fall
    # back to staging (the gate only ever delays a write, never drops it).
    try:
        choice = callback(command, description, allow_permanent=False)
    except Exception as e:
        logger.error("Inline memory approval prompt failed: %s", e)
        return None

    if choice in {"once", "session"}:
        return True
    if choice == "deny":
        return False
    # Any other outcome (e.g. timeout that returns "deny" already handled) →
    # treat unknown as no-decision so we stage rather than silently drop.
    return None


# ---------------------------------------------------------------------------
# Skill-specific helpers (gist + diff for the review affordances)
# ---------------------------------------------------------------------------

def skill_gist(action: str, name: str, *, content: str = "",
               file_path: str = "", old_string: str = "",
               new_string: str = "") -> str:
    """Build a one-line human gist for a pending skill write.

    Heuristic, no model call — the gist surfaces enough to decide approve/reject
    in a chat bubble, while the full diff stays behind /skills diff (CLI/
    dashboard/file). For create/edit it pulls the frontmatter ``description:``;
    for patch/write_file it describes the size of the change.
    """
    if action in {"create", "edit"} and content:
        desc = _frontmatter_description(content)
        size = f"{len(content) // 1024 + 1} KB" if len(content) >= 1024 else f"{len(content)} chars"
        verb = "create" if action == "create" else "rewrite"
        if desc:
            return f"{verb} '{name}' — {desc} ({size})"
        return f"{verb} '{name}' ({size})"
    if action == "patch":
        target = file_path or "SKILL.md"
        removed = old_string.count("\n") + 1 if old_string else 0
        added = new_string.count("\n") + 1 if new_string else 0
        return f"patch '{name}' {target} (+{added}/-{removed} lines)"
    if action == "write_file":
        return f"write {file_path} in '{name}'"
    if action == "remove_file":
        return f"remove {file_path} from '{name}'"
    if action == "delete":
        return f"delete skill '{name}'"
    return f"{action} '{name}'"


def _frontmatter_description(content: str) -> str:
    """Extract the ``description:`` value from SKILL.md YAML frontmatter."""
    import re
    m = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
    if not m:
        return ""
    desc = m.group(1).strip().strip("'\"")
    return desc[:140]


def skill_pending_diff(record: Dict[str, Any]) -> str:
    """Build a full unified diff (or full content) for a staged skill write.

    Used by /skills diff <id> on a surface that can render it (CLI pager, web
    dashboard, or by opening the pending JSON file). For create this is the new
    file content; for edit/patch it is a unified diff against the current
    on-disk skill.
    """
    import difflib
    payload = record.get("payload", {})
    action = payload.get("action", "")
    name = payload.get("name", "")

    if action == "create":
        return (payload.get("content") or "")

    # Resolve current on-disk content for diffable actions.
    try:
        from tools.skill_manager_tool import _find_skill
    except Exception:
        _find_skill = None  # type: ignore

    current = ""
    target_label = "SKILL.md"
    if _find_skill is not None:
        found = _find_skill(name)
        if found:
            base = found["path"]
            if action == "edit":
                p = base / "SKILL.md"
            elif action in {"patch", "write_file"}:
                rel = payload.get("file_path") or "SKILL.md"
                p = base / rel
                target_label = rel
            else:
                p = base / "SKILL.md"
            try:
                if p.exists():
                    current = p.read_text(encoding="utf-8")
            except Exception:
                current = ""

    if action == "edit":
        new = payload.get("content") or ""
    elif action == "patch":
        old_s = payload.get("old_string") or ""
        new_s = payload.get("new_string") or ""
        new = current.replace(old_s, new_s) if current else f"(patch {old_s!r} → {new_s!r})"
    elif action == "write_file":
        new = payload.get("file_content") or ""
    elif action == "remove_file":
        return f"remove file: {payload.get('file_path')} from skill '{name}'"
    elif action == "delete":
        return f"delete skill '{name}'"
    else:
        return f"({action} on '{name}')"

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{target_label}",
        tofile=f"b/{target_label}",
    )
    text = "".join(diff)
    return text or "(no textual change)"
