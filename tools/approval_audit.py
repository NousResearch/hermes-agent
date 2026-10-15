"""Redacted, best-effort approval evidence; never changes an approval verdict."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import stat
import threading
import time

from hermes_constants import get_hermes_home, mkdir_under_hermes_home

logger = logging.getLogger(__name__)
_write_lock = threading.Lock()


def log_approval_event(pattern_key, description, command, result, surface=""):
    """Append one private JSONL record. Redaction or disk failure never leaks raw text."""
    try:
        from agent.redact import redact_sensitive_text
        from tools.approval_context import get_current_session_key

        text = {"pattern_key": pattern_key, "description": description, "command": command,
                "surface": surface}
        try:
            text = {key: redact_sensitive_text(str(value or ""), force=True)
                    for key, value in text.items()}
        except Exception:
            text = {key: "[redaction unavailable]" for key in text}
        result = result if isinstance(result, dict) else {}
        approved = bool(result.get("approved", False))
        outcome = result.get("outcome")
        if outcome not in {"approved", "denied", "blocked", "timeout", "pending", "cancelled", "notify_failed"}:
            # Only classify structured verdicts; messages may be localized and
            # must not turn a missing decision into a claimed user refusal.
            if approved:
                outcome = "approved"
            elif result.get("approval_pending") or result.get("status") in {"approval_required", "pending_approval"}:
                outcome = "pending"
            else:
                outcome = "blocked" if result.get("approved") is False else "unknown"
        session = get_current_session_key(default="")
        record = {"ts": time.time(), **text, "approved": approved, "outcome": outcome,
                  "session_key": "sha256:" + hashlib.sha256(session.encode()).hexdigest()[:12] if session else ""}
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode()
        path = get_hermes_home() / "logs" / "approvals.jsonl"
        mkdir_under_hermes_home(path.parent)
        # Nonblocking also refuses a planted FIFO without hanging the approval path.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        with _write_lock:
            if path.is_symlink():
                return
            fd = os.open(path, flags, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    return
                if hasattr(os, "fchmod"):
                    os.fchmod(fd, 0o600)
                # One append syscall avoids interleaving records between processes.
                if os.write(fd, payload) != len(payload):
                    logger.debug("approval audit append was incomplete")
            finally:
                os.close(fd)
    except Exception:
        logger.debug("approval audit unavailable")


def audited(surface, *, always=False):
    """Observe public approval entry points without changing their return contract."""
    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            result = fn(*args, **kwargs)
            if always or result != {"approved": True, "message": None}:
                values = signature.bind(*args, **kwargs).arguments
                log_approval_event(
                    result.get("pattern_key", values.get("rule_key", "")),
                    result.get("description", values.get("reason", "")),
                    values.get("command", values.get("code", values.get("tool_name", ""))),
                    result, surface,
                )
            return result
        return wrapped
    return decorate
