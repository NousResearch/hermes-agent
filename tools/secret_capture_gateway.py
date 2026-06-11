"""Gateway-side protected secret/sensitive-input capture.

A tool call registers a pending request, the gateway sends a user-facing prompt,
and the next typed message from the same user/session is captured as sensitive
payload instead of being appended to normal chat history.  Only safe metadata is
returned to the LLM/tool result; the raw value is written locally by the tool
callback.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

FinalizeCallback = Callable[["_SecretCaptureEntry", str], None]
NotifyCallback = Callable[["_SecretCaptureEntry"], None]


@dataclass
class _SecretCaptureEntry:
    secret_id: str
    session_key: str
    env_var: str
    prompt: str
    event: threading.Event = field(default_factory=threading.Event)
    value: Optional[str] = None
    cancelled: bool = False
    reason: str = ""
    resolved: bool = False
    # Bound by the gateway notification callback before the prompt is sent.
    platform: str = ""
    chat_id: str = ""
    thread_id: str = ""
    user_id: str = ""
    message_id: str = ""

    def signature(self) -> Dict[str, str]:
        return {
            "secret_id": self.secret_id,
            "session_key": self.session_key,
            "env_var": self.env_var,
            "prompt": self.prompt,
        }

    def bind_source(self, source: Any) -> None:
        """Attach routing/user identity metadata from a gateway SessionSource."""
        self.platform = str(getattr(getattr(source, "platform", None), "value", "") or "")
        self.chat_id = str(getattr(source, "chat_id", "") or "")
        self.thread_id = str(getattr(source, "thread_id", "") or "")
        self.user_id = str(getattr(source, "user_id", "") or "")
        self.message_id = str(getattr(source, "message_id", "") or "")

    def matches_source(self, source: Any) -> bool:
        """Return True if a reply/callback is allowed to resolve this entry."""
        if not self.platform and not self.chat_id and not self.user_id:
            # Backward-compatible for unit tests and non-gateway callers that do
            # not bind a source. Gateway-bound entries are always bound before
            # the prompt is sent.
            return True
        platform = str(getattr(getattr(source, "platform", None), "value", "") or "")
        chat_id = str(getattr(source, "chat_id", "") or "")
        thread_id = str(getattr(source, "thread_id", "") or "")
        user_id = str(getattr(source, "user_id", "") or "")
        if self.platform and platform != self.platform:
            return False
        if self.chat_id and chat_id != self.chat_id:
            return False
        if self.thread_id and thread_id != self.thread_id:
            return False
        if self.user_id and user_id != self.user_id:
            return False
        return True


_lock = threading.RLock()
_entries: Dict[str, _SecretCaptureEntry] = {}
_session_index: Dict[str, list[str]] = {}
_notify_cbs: Dict[str, NotifyCallback] = {}
_finalize_cbs: Dict[str, FinalizeCallback] = {}


def _remove_entry_locked(entry: _SecretCaptureEntry) -> None:
    _entries.pop(entry.secret_id, None)
    ids = _session_index.get(entry.session_key)
    if ids and entry.secret_id in ids:
        ids.remove(entry.secret_id)
        if not ids:
            _session_index.pop(entry.session_key, None)


def _finalize(entry: _SecretCaptureEntry, status: str) -> None:
    cb = None
    with _lock:
        cb = _finalize_cbs.get(entry.session_key)
    if cb is None:
        return
    try:
        cb(entry, status)
    except Exception:
        logger.debug("secret-capture finalize callback failed", exc_info=True)


def register(secret_id: str, session_key: str, env_var: str, prompt: str) -> _SecretCaptureEntry:
    entry = _SecretCaptureEntry(
        secret_id=secret_id,
        session_key=session_key or "",
        env_var=env_var,
        prompt=prompt,
    )
    with _lock:
        _entries[secret_id] = entry
        _session_index.setdefault(entry.session_key, []).append(secret_id)
    return entry


def wait_for_response(secret_id: str, timeout: float) -> Optional[_SecretCaptureEntry]:
    with _lock:
        entry = _entries.get(secret_id)
    if entry is None:
        return None

    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover
        touch_activity_if_due = None

    deadline = time.monotonic() + max(timeout, 0.0)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if entry.event.wait(timeout=min(1.0, remaining)):
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for user secret input")

    timed_out = False
    with _lock:
        current = _entries.get(secret_id)
        if current is entry:
            # No resolver/canceller won the race before timeout.
            entry.cancelled = True
            entry.reason = entry.reason or "timeout"
            timed_out = True
            _remove_entry_locked(entry)
    if timed_out:
        entry.event.set()
        _finalize(entry, "timeout")
    return entry


def resolve_gateway_secret(secret_id: str, value: str, source: Any = None) -> bool:
    with _lock:
        entry = _entries.get(secret_id)
        if entry is None or entry.resolved or entry.cancelled:
            return False
        if source is not None and not entry.matches_source(source):
            return False
        entry.value = str(value) if value is not None else ""
        entry.cancelled = False
        entry.resolved = True
        _remove_entry_locked(entry)
        entry.event.set()
    _finalize(entry, "received")
    return True


def cancel_gateway_secret(secret_id: str, reason: str = "cancelled", source: Any = None) -> bool:
    with _lock:
        entry = _entries.get(secret_id)
        if entry is None or entry.resolved or entry.cancelled:
            return False
        if source is not None and not entry.matches_source(source):
            return False
        entry.value = None
        entry.cancelled = True
        entry.reason = reason or "cancelled"
        _remove_entry_locked(entry)
        entry.event.set()
    _finalize(entry, entry.reason or "cancelled")
    return True


def get_pending_for_session(session_key: str, source: Any = None) -> Optional[_SecretCaptureEntry]:
    with _lock:
        ids = list(_session_index.get(session_key or "") or [])
        for sid in ids:
            entry = _entries.get(sid)
            if entry is not None and (source is None or entry.matches_source(source)):
                return entry
    return None


def has_pending(session_key: str) -> bool:
    return get_pending_for_session(session_key) is not None


def clear_session(session_key: str) -> int:
    with _lock:
        ids = list(_session_index.pop(session_key or "", []) or [])
        entries = [_entries.pop(sid, None) for sid in ids]
    cancelled = 0
    for entry in entries:
        if entry is None:
            continue
        entry.cancelled = True
        entry.reason = "session_cleared"
        entry.event.set()
        cancelled += 1
        _finalize(entry, "session_cleared")
    return cancelled


def register_notify(session_key: str, cb: NotifyCallback) -> None:
    with _lock:
        _notify_cbs[session_key or ""] = cb


def register_finalize(session_key: str, cb: FinalizeCallback) -> None:
    with _lock:
        _finalize_cbs[session_key or ""] = cb


def unregister_notify(session_key: str) -> None:
    # Clear pending entries while the finalize callback is still registered so
    # platform prompts can be edited/expired during agent interruption/cleanup.
    clear_session(session_key or "")
    with _lock:
        _notify_cbs.pop(session_key or "", None)
        _finalize_cbs.pop(session_key or "", None)


def get_notify(session_key: str) -> Optional[NotifyCallback]:
    with _lock:
        return _notify_cbs.get(session_key or "")


def get_secret_capture_timeout() -> int:
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        agent_cfg = cfg.get("agent", {}) or {}
        return int(agent_cfg.get("secret_capture_timeout", agent_cfg.get("clarify_timeout", 600)))
    except Exception:
        return 600
