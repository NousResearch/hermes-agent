"""Opt-in native skill handler receipts, not authenticated identity or commit proof.

Only the dispatcher uses this wrapper. Direct Python calls are not covered.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_WARNING = "Native skill-call audit unavailable; tool execution is unaffected."


@dataclass
class _Span:
    invocation_id: str
    key: bytes = field(repr=False)
    ledger_entries: list = field(default_factory=list)
    batch_rollback: str = "not_observed"


_current: ContextVar[_Span | None] = ContextVar("native_skill_audit_span", default=None)


def batch_rollback_observed(failed: bool) -> None:
    span = _current.get()
    if span is not None:
        span.batch_rollback = "rollback_failed" if failed else "rolled_back"


def ledger_invocation_id() -> str | None:
    span = _current.get()
    return span.invocation_id if span is not None else None


def ledger_appended(entry_id: str, evidence: dict) -> None:
    span = _current.get()
    if span is None:
        return
    try:
        span.ledger_entries.append({"id": entry_id, "evidence_hmac_sha256": _digest(span.key, evidence)})
    except Exception:  # health: allow BLE001 -- audit failure must preserve tool semantics; warn without private exception details
        _warn()


def _warn() -> None:
    # Never attach exception text, paths, arguments, results or traceback.
    try:
        logger.warning(_WARNING)
    except Exception:  # health: allow BLE001,S110 -- a broken logging sink must not block the tool or expose private exceptions
        # A broken logging sink is another audit failure, never a tool failure.
        pass


def _enabled() -> bool:
    from hermes_cli.config import cfg_get, load_config_readonly
    return cfg_get(load_config_readonly(), "skills", "native_call_audit", default=False) is True


def _digest(key: bytes, value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def _secure_stat(info) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("unsafe audit file")
    if info.st_uid != os.geteuid():  # windows-footgun: ok — guarded by _profile_directory primitives check
        raise ValueError("unsafe audit owner")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("unsafe audit mode")
    if info.st_nlink != 1:
        raise ValueError("unsafe audit links")


def _same_file(left, right) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


@contextmanager
def _profile_directory(home: Path):
    # Decline rather than silently using path-following or blocking fallbacks.
    required = ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK", "O_CLOEXEC", "geteuid")
    if (any(not hasattr(os, name) for name in required)
            or not {"open", "stat"}.issubset({fn.__name__ for fn in os.supports_dir_fd})
            or "stat" not in {fn.__name__ for fn in os.supports_follow_symlinks}):
        raise OSError("secure audit IO unsupported")
    path = Path(home)
    if ".." in path.parts:
        raise ValueError("unsafe audit directory traversal")
    path = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    descriptors = []
    links = []
    try:
        fd = os.open(path.anchor, flags)
        descriptors.append(fd)
        for component in path.parts[1:]:
            parent = fd
            fd = os.open(component, flags, dir_fd=parent)
            descriptors.append(fd)
            links.append((parent, component, fd))
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:  # windows-footgun: ok — checked above
            raise ValueError("unsafe audit directory")
        def stable():
            for parent, name, child in links:
                if not _same_file(os.fstat(child), os.stat(name, dir_fd=parent, follow_symlinks=False)):
                    raise ValueError("audit directory replaced")
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
        stable()
        yield fd, stable
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


@contextmanager
def _audit_file(directory: int, name: str, flags: int):
    try:
        before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        before = None
    if before is not None:
        _secure_stat(before)
    options = flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    if before is None:
        options |= os.O_CREAT | os.O_EXCL
    fd = os.open(name, options, 0o600, dir_fd=directory)
    try:
        current = os.fstat(fd)
        _secure_stat(current)
        if before is not None and not _same_file(before, current):
            raise ValueError("audit file replaced")
        if not _same_file(current, os.stat(name, dir_fd=directory, follow_symlinks=False)):
            raise ValueError("audit file replaced")
        yield fd, before is None
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("audit write made no progress")
        remaining = remaining[written:]


def _key(home: Path) -> bytes:
    with _profile_directory(home) as (directory, stable):
        with _audit_file(directory, "skill_native_audit.key", os.O_RDWR) as (fd, created):
            stable()
            if created:
                key = os.urandom(32)
                _write_all(fd, key)
                os.fsync(fd)
                os.fsync(directory)
            else:
                key = os.read(fd, 33)
            if len(key) != 32:
                raise ValueError("invalid key")
            return key


def _append(home: Path, row: dict) -> None:
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with _profile_directory(home) as (directory, stable):
        with _audit_file(directory, "skill_native_audit.jsonl", os.O_RDWR | os.O_APPEND) as (fd, created):
            stable()
            # Preserve incomplete evidence; never merge a new row into its tail.
            if os.fstat(fd).st_size:
                os.lseek(fd, -1, os.SEEK_END)
                if os.read(fd, 1) != b"\n":
                    raise ValueError("incomplete audit journal tail")
            _write_all(fd, data)
            os.fsync(fd)
            if created:
                os.fsync(directory)


def _handler_status(result) -> str:
    if not isinstance(result, dict):
        return "unknown"
    if result.get("staged") is True or result.get("approval_pending") is True:
        return "approval_pending"
    if result.get("success") is False or result.get("error"):
        return "failure"
    return "success" if result.get("success") is True else "unknown"


def _unaudited_dispatch(dispatch):
    # A nested unrecorded call must not attribute its mutations to its parent.
    token = _current.set(None)
    try:
        return dispatch()
    finally:
        _current.reset(token)


def dispatch_skill_call(args: dict, ids: dict, dispatch):
    """Record FINAL registry arguments; audit failures must never change tool semantics."""
    try:
        enabled = _enabled()
    except Exception:  # health: allow BLE001 -- audit failure must preserve tool semantics; warn without private exception details
        _warn()
        return _unaudited_dispatch(dispatch)
    if not enabled:
        return _unaudited_dispatch(dispatch)
    try:
        from tools.skill_ledger import ledger_enabled
        ledger_active = ledger_enabled()
        home = get_hermes_home()
        key = _key(home)
        invocation = uuid.uuid4().hex
        _append(home, {"version": 1, "event": "handler_entry", "invocation_id": invocation,
                       "ts": datetime.now(timezone.utc).isoformat(), "native_ids": ids,
                       "args_hmac_sha256": _digest(key, args)})
    except Exception:  # health: allow BLE001 -- audit failure must preserve tool semantics; warn without private exception details
        _warn()
        return _unaudited_dispatch(dispatch)
    span = _Span(invocation, key)
    token = _current.set(span)
    def complete(status, result):
        try:
            _append(home, {"version": 1, "event": "handler_completion", "invocation_id": invocation,
                           "ts": datetime.now(timezone.utc).isoformat(),
                           "handler_status": status,
                           "batch_rollback": span.batch_rollback,
                           "ledger_status": "appended" if span.ledger_entries else "missing" if ledger_active else "disabled",
                           "ledger_entries": span.ledger_entries,
                           "result_hmac_sha256": None if status == "exception" else _digest(key, result)})
        except Exception:  # health: allow BLE001 -- audit failure must preserve tool semantics; warn without private exception details
            _warn()

    try:
        try:
            result = dispatch()
        except Exception:  # health: allow BLE001 -- audit failure must preserve tool semantics; warn without private exception details
            complete("exception", None)
            raise
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            status = _handler_status(parsed)
        except Exception:  # health: allow BLE001 -- audit failure must preserve tool semantics; warn without private exception details
            _warn()
            return result
        complete(status, result)
        return result
    finally:
        _current.reset(token)
