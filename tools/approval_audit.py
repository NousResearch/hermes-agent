"""Opt-in, synchronous approval receipts; no filesystem access while disabled.

Configuration is published by the existing config loaders, never polled here.
All writers lock a separate file and close the JSONL before rotation (Windows).
"""

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_settings: dict[str, tuple[int, int]] = {}
_lock = threading.Lock()
_LOCK_TIMEOUT = 3.0
_warned_enabled = False
_GENESIS = "0" * 64
_SOURCES = {"cli": "interactive", "smart": "smart_approval", "gateway": "gateway_wait"}
_VERDICTS = {"once": "approved", "session": "approved", "always": "approved",
             "smart_approve": "approved", "smart_deny": "denied", "deny": "denied",
             "timeout": "timeout", "notify_failed": "notify_failed",
             "transport_timeout": "timeout",
             **{f"transport_{failure}": "denied" for failure in
                ("error", "invalid", "stale", "busy", "interrupted", "unavailable")}}


def configure(config: dict) -> None:
    """Snapshot only audit settings during an already-authorized config load."""
    global _warned_enabled
    approvals = config.get("approvals") or {}
    audit = approvals.get("audit_log") if isinstance(approvals, dict) else None
    enabled = audit.get("enabled", False) if isinstance(audit, dict) else False
    if (audit is not None and not isinstance(audit, dict)) or not isinstance(enabled, bool):
        if not _warned_enabled:
            logger.warning("approvals.audit_log.enabled must be a boolean; audit logging disabled")
            _warned_enabled = True
    if enabled is not True:
        # Cold/default-off loads don't even resolve the profile. A live disable must remove
        # this profile's earlier opt-in, without disabling other concurrently active profiles.
        if _settings:
            _settings.pop(str(get_hermes_home()), None)
        return
    home = str(get_hermes_home())
    log = config.get("logging") or {}
    try:
        limits = (max(1, int(log.get("max_size_mb", 5))) * 1024 * 1024,
                  max(1, int(log.get("backup_count", 3))))
    except (TypeError, ValueError, OverflowError, AttributeError):
        limits = (5 * 1024 * 1024, 3)
    _settings[home] = limits


def _digest(value) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def is_enabled() -> bool:
    return bool(_settings) and str(get_hermes_home()) in _settings


def decision_actor(payload: dict) -> str:
    choice = payload.get("choice", "")
    if choice in {"timeout", "transport_timeout"}:
        return "timeout"
    if payload.get("surface") == "smart":
        return "aux_llm"
    origin = payload.get("origin_surface", payload.get("surface"))
    return "user" if origin == "cli" else "gateway"


def _try_file_lock(handle) -> bool:
    if os.name == "nt":
        import portalocker
        try:
            portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.LockException:
            return False
    else:
        import fcntl
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


@contextmanager
def _writer_lock(path: Path):
    # One shared budget bounds contention on BOTH the in-process and OS locks.
    deadline = time.monotonic() + _LOCK_TIMEOUT
    if not _lock.acquire(timeout=_LOCK_TIMEOUT):
        raise TimeoutError("Approval audit thread lock timeout")
    try:
        with path.open("a+b") as handle:
            while not _try_file_lock(handle):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Approval audit file lock timeout")
                time.sleep(min(0.05, remaining))
            try:
                yield
            finally:
                if os.name == "nt":
                    import portalocker
                    portalocker.unlock(handle)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        _lock.release()


def _sync_directory(path: Path) -> None:
    # Windows doesn't expose POSIX directory fsync through os.open.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _tail_hash(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if not size:
                return _GENESIS
            handle.seek(size - 1)
            if handle.read(1) != b"\n":
                raise ValueError("Incomplete approval audit record")
            # Full pattern-key lists can exceed one block; read the whole final record.
            position = size - 1
            chunks = []
            while position:
                start = max(0, position - 65536)
                handle.seek(start)
                chunk = handle.read(position - start)
                chunks.append(chunk.rsplit(b"\n", 1)[-1])
                if b"\n" in chunk:
                    break
                position = start
    except FileNotFoundError:
        return _GENESIS
    record = json.loads(b"".join(reversed(chunks)))
    digest = record.pop("hash")
    if digest != hashlib.sha256(_canonical(record)).hexdigest():
        raise ValueError("Invalid approval audit tail hash")
    return digest


def _append(path: Path, record: dict, max_bytes: int, backups: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _writer_lock(path.with_suffix(".jsonl.lock")):
        # A rollover may have completed just before a previous writer crashed.
        predecessor = path if path.exists() and path.stat().st_size else Path(str(path) + ".1")
        record["prev_hash"] = _tail_hash(predecessor)
        record["hash"] = hashlib.sha256(_canonical(record)).hexdigest()
        line = _canonical(record) + b"\n"
        if path.exists() and path.stat().st_size and path.stat().st_size + len(line) > max_bytes:
            for index in range(backups, 0, -1):
                source = path if index == 1 else Path(f"{path}.{index - 1}")
                if source.exists():
                    os.replace(source, Path(f"{path}.{index}"))
            _sync_directory(path.parent)
        with path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def record_decision(payload: dict) -> None:
    """Observe a response, preserving the decision even when the sink fails."""
    if not _settings:
        return
    home = get_hermes_home()
    limits = _settings.get(str(home))
    if limits is None:
        return
    try:
        surface = payload.get("surface", "")
        choice = payload.get("choice", "")
        source = surface if surface.startswith("transport:") else _SOURCES.get(surface, surface or "other")
        origin = payload.get("origin_surface", surface)
        scope = payload.get("scope", choice if choice in {"once", "session", "always"} else None)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "session_id": str(payload["session_id"])[:256] if payload.get("session_id") else _digest(payload.get("session_key", "")),
            "turn_id": str(payload.get("turn_id") or "")[:256],
            "tool_call_id": str(payload.get("tool_call_id") or "")[:256],
            # Hooks currently lack reliable tool/target metadata. Never infer it from shell text.
            "tool": None, "target": None,
            "pattern_key": _digest(payload.get("pattern_key", "")),
            "pattern_keys": [_digest(key) for key in payload.get("pattern_keys", [payload.get("pattern_key", "")])],
            "raw_choice": choice if choice in _VERDICTS else _digest(choice),
            "scope": scope,
            "verdict": _VERDICTS.get(choice, "unknown"),
            "decided_by": decision_actor(payload),
            "source": source, "interactive": origin == "cli",
            "description": _digest(payload.get("description", "")),
            "content_redacted": _digest(payload.get("command", "")),
        }
        _append(home / "logs" / "approvals.jsonl", record, *limits)
    except Exception as exc:
        # Exception text can contain payloads or paths. Keep diagnostics content-free.
        logger.warning("Approval audit write failed (%s)", type(exc).__name__)
