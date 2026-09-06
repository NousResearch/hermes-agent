"""Opt-in, synchronous approval receipts; no filesystem access while disabled.

Configuration is published by the existing config loaders, never polled here.
All writers lock a separate file and close the JSONL before rotation (Windows).
"""

import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_settings: dict[str, tuple[int, int]] = {}
_lock = threading.Lock()
_GENESIS = "0" * 64
_SOURCES = {"cli": "interactive", "smart": "smart_approval", "gateway": "gateway_wait"}
_VERDICTS = {"once": "approved", "session": "approved", "always": "approved",
             "smart_approve": "approved", "smart_deny": "denied", "deny": "denied",
             "timeout": "timeout", "notify_failed": "notify_failed"}


def configure(config: dict) -> None:
    """Snapshot only audit settings during an already-authorized config load."""
    approvals = config.get("approvals") or {}
    audit = approvals.get("audit_log") if isinstance(approvals, dict) else None
    home = str(get_hermes_home())
    if not isinstance(audit, dict) or audit.get("enabled") is not True:
        _settings.pop(home, None)
        return
    log = config.get("logging") or {}
    try:
        limits = (max(1, int(log.get("max_size_mb", 5))) * 1024 * 1024,
                  max(1, int(log.get("backup_count", 3))))
    except (TypeError, ValueError, AttributeError):
        limits = (5 * 1024 * 1024, 3)
    _settings[home] = limits


def _digest(value) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@contextmanager
def _writer_lock(path: Path):
    with _lock, path.open("a+b") as handle:
        if os.name == "nt":
            import portalocker
            portalocker.lock(handle, portalocker.LOCK_EX)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                portalocker.unlock(handle)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _tail_hash(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if not size:
                return _GENESIS
            handle.seek(max(0, size - 65536))
            tail = handle.read()
    except FileNotFoundError:
        return _GENESIS
    if not tail.endswith(b"\n"):
        raise ValueError("Incomplete approval audit record")
    record = json.loads(tail.splitlines()[-1])
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
        source = _SOURCES.get(surface, "other")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "session_id": str(payload.get("session_id") or payload.get("session_key") or "")[:256],
            "turn_id": str(payload.get("turn_id") or "")[:256],
            "tool_call_id": str(payload.get("tool_call_id") or "")[:256],
            # Hooks currently lack reliable tool/target metadata. Never infer it from shell text.
            "tool": None, "target": None,
            "pattern_key": _digest(payload.get("pattern_key", "")),
            "verdict": _VERDICTS.get(choice, "unknown"),
            "decided_by": "aux_llm" if surface == "smart" else (
                "timeout" if choice == "timeout" else ("user" if surface == "cli" else "gateway")),
            "source": source, "interactive": surface == "cli",
            "description": _digest(payload.get("description", "")),
            "content_redacted": _digest(payload.get("command", "")),
        }
        _append(home / "logs" / "approvals.jsonl", record, *limits)
    except Exception as exc:
        # Exception text can contain payloads or paths. Keep diagnostics content-free.
        logger.warning("Approval audit write failed (%s)", type(exc).__name__)
