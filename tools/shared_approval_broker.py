"""File-backed approval broker shared by CLI and Gateway processes.

Gateway dangerous-command approvals are normally in-memory because the waiting
agent thread and /approve handler live in the same process.  CLI/TUI approvals
are different: the CLI process owns the prompt while Telegram /approve arrives
in the gateway process.  This module bridges that gap with small JSON records
under HERMES_HOME.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

_VALID_CHOICES = {"once", "session", "always", "deny"}
_LOCK = threading.RLock()


def _approval_dir() -> Path:
    return get_hermes_home() / "approvals" / "cli"


def _ensure_private_dir(path: Path) -> None:
    """Create the broker directory with owner-only permissions when possible."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _request_path(request_id: str) -> Path:
    safe = "".join(ch for ch in str(request_id) if ch.isalnum() or ch in ("-", "_"))
    if not safe:
        raise ValueError("request_id is empty")
    return _approval_dir() / f"{safe}.json"


def _now() -> float:
    return time.time()


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    _ensure_private_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _normalize_platform(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def _source_matches_target(source: Optional[dict[str, Any]], notify_target: str) -> bool:
    """Return whether a gateway source is allowed to resolve a CLI request.

    CLI approvals are only remotely resolvable from the platform/chat that was
    notified. This avoids a generic gateway /approve in one channel resolving a
    dangerous command prompt that was announced somewhere else.
    """
    if not source:
        return False
    target = str(notify_target or "").strip().lower()
    if not target:
        return False

    platform = _normalize_platform(source.get("platform"))
    chat_id = str(source.get("chat_id") or "").strip().lower()
    thread_id = str(source.get("thread_id") or "").strip().lower()
    parent_chat_id = str(source.get("parent_chat_id") or "").strip().lower()
    # send_message targets are commonly: "telegram:<chat>" or
    # "telegram:<chat>:<thread>".  Broker resolution requires a concrete
    # chat/thread target; a bare platform target such as "telegram" is only a
    # notification destination alias and is not specific enough to authorize a
    # remote dangerous-command approval.
    if target == platform:
        return False
    prefix = f"{platform}:"
    if not target.startswith(prefix):
        return False
    remainder = target[len(prefix):]
    parts = [part for part in remainder.split(":") if part]
    if not parts:
        return False
    if parts[0] not in {chat_id, parent_chat_id}:
        return False
    if len(parts) >= 2 and parts[1] != thread_id:
        return False
    return True


def _matches_resolver(data: dict[str, Any], source: Optional[dict[str, Any]]) -> bool:
    return _source_matches_target(source, str(data.get("notify_target") or ""))


def _read_request(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _is_expired(data: dict[str, Any], *, now: Optional[float] = None) -> bool:
    if data.get("status") != "pending":
        return False
    try:
        expires_at = float(data.get("expires_at", 0) or 0)
    except (TypeError, ValueError):
        expires_at = 0
    return expires_at > 0 and (now if now is not None else _now()) >= expires_at


def _mark_expired(path: Path, data: dict[str, Any], *, now: Optional[float] = None) -> None:
    data = dict(data)
    data["status"] = "expired"
    data["resolved_at"] = now if now is not None else _now()
    data.setdefault("choice", None)
    try:
        _atomic_write(path, data)
    except OSError:
        pass


def register_cli_approval(payload: dict[str, Any], ttl_seconds: int | float = 60) -> str:
    """Register a pending CLI approval and return its request id.

    ``payload`` should include session/cwd/pid/command details, but this helper
    is intentionally schema-light so older/newer CLI versions can coexist with
    the gateway.  The broker adds request_id/status/timestamps.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    request_id = str(payload.get("request_id") or uuid.uuid4().hex)
    now = _now()
    try:
        ttl = float(ttl_seconds)
    except (TypeError, ValueError):
        ttl = 60.0
    expires_at = 0 if ttl <= 0 else now + ttl
    data = dict(payload)
    data.update(
        {
            "request_id": request_id,
            "status": "pending",
            "choice": None,
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
        }
    )
    with _LOCK:
        _atomic_write(_request_path(request_id), data)
    return request_id


def get_cli_approval(request_id: str) -> Optional[dict[str, Any]]:
    """Return one CLI approval record, marking expired pending records."""
    path = _request_path(request_id)
    with _LOCK:
        data = _read_request(path)
        if data is None:
            return None
        if _is_expired(data):
            _mark_expired(path, data)
            data = _read_request(path) or data
        return data


def list_pending_cli_approvals() -> list[dict[str, Any]]:
    """List non-expired pending CLI approvals, oldest first."""
    base = _approval_dir()
    if not base.exists():
        return []
    pending: list[dict[str, Any]] = []
    now = _now()
    with _LOCK:
        for path in sorted(base.glob("*.json")):
            data = _read_request(path)
            if not data or data.get("status") != "pending":
                continue
            if _is_expired(data, now=now):
                _mark_expired(path, data, now=now)
                continue
            pending.append(data)
    pending.sort(key=lambda item: float(item.get("created_at", 0) or 0))
    return pending


def list_resolvable_cli_approvals(source: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """List pending CLI approvals that the given gateway source may resolve."""
    return [item for item in list_pending_cli_approvals() if _matches_resolver(item, source)]


def resolve_cli_approval(request_id: str, choice: str) -> bool:
    """Resolve a specific pending CLI approval."""
    normalized = (choice or "").strip().lower()
    if normalized not in _VALID_CHOICES:
        raise ValueError(f"invalid approval choice: {choice!r}")
    path = _request_path(request_id)
    with _LOCK:
        data = _read_request(path)
        if not data or data.get("status") != "pending" or _is_expired(data):
            if data and _is_expired(data):
                _mark_expired(path, data)
            return False
        data = dict(data)
        data["status"] = "resolved"
        data["choice"] = normalized
        data["resolved_at"] = _now()
        data["updated_at"] = data["resolved_at"]
        _atomic_write(path, data)
    return True


def resolve_oldest_cli_approval(
    choice: str,
    *,
    resolve_all: bool = False,
    source: Optional[dict[str, Any]] = None,
) -> int:
    """Resolve the oldest pending CLI approval(s) visible to the gateway."""
    pending = list_resolvable_cli_approvals(source)
    if not pending:
        return 0
    targets = pending if resolve_all else pending[:1]
    count = 0
    for item in targets:
        if resolve_cli_approval(str(item.get("request_id", "")), choice):
            count += 1
    return count


def wait_for_cli_approval(
    request_id: str,
    *,
    timeout_seconds: int | float = 60,
    poll_interval: int | float = 0.25,
) -> Optional[str]:
    """Poll until a CLI approval is resolved, expired, missing, or timed out."""
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        timeout = 60.0
    try:
        interval = max(float(poll_interval), 0.01)
    except (TypeError, ValueError):
        interval = 0.25
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    while True:
        data = get_cli_approval(request_id)
        if not data:
            return None
        status = data.get("status")
        if status == "resolved":
            choice = data.get("choice")
            return str(choice) if choice else None
        if status in ("expired", "cancelled"):
            return None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(interval, remaining))
        else:
            # Non-blocking check for callers that pass timeout_seconds=0.
            return None


def clear_cli_approval(request_id: str) -> None:
    """Remove one CLI approval file; best-effort cleanup for CLI finally blocks."""
    try:
        _request_path(request_id).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
