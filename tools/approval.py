"""Thread-safe per-session approval management for dangerous commands.

Replaces the module-level globals (_last_pending_approval, _session_approved_patterns)
that were previously in terminal_tool.py. Those globals were shared across all
concurrent gateway sessions, creating race conditions where one session's approval
could overwrite another's.

This module provides session-scoped state keyed by session_key, with proper locking.
"""

import threading
from typing import Optional


_lock = threading.Lock()

# Pending approval requests: session_key -> approval_dict
_pending: dict[str, dict] = {}

# Session-scoped approved patterns: session_key -> set of pattern_keys
_session_approved: dict[str, set] = {}

# Permanent allowlist (loaded from config, shared across sessions intentionally)
_permanent_approved: set = set()


def submit_pending(session_key: str, approval: dict):
    """Store a pending approval request for a session.

    Called by _check_dangerous_command when a gateway session hits a
    dangerous command. The gateway picks it up later via pop_pending().
    """
    with _lock:
        _pending[session_key] = approval


def pop_pending(session_key: str) -> Optional[dict]:
    """Retrieve and remove a pending approval for a session.

    Returns the approval dict if one was pending, None otherwise.
    Atomic: no other thread can read the same pending approval.
    """
    with _lock:
        return _pending.pop(session_key, None)


def has_pending(session_key: str) -> bool:
    """Check if a session has a pending approval request."""
    with _lock:
        return session_key in _pending


def approve_session(session_key: str, pattern_key: str):
    """Approve a dangerous command pattern for this session only.

    The approval is scoped to the session -- other sessions are unaffected.
    """
    with _lock:
        _session_approved.setdefault(session_key, set()).add(pattern_key)


def is_approved(session_key: str, pattern_key: str) -> bool:
    """Check if a pattern is approved (session-scoped or permanent)."""
    with _lock:
        if pattern_key in _permanent_approved:
            return True
        return pattern_key in _session_approved.get(session_key, set())


def approve_permanent(pattern_key: str):
    """Add a pattern to the permanent (cross-session) allowlist."""
    with _lock:
        _permanent_approved.add(pattern_key)


def load_permanent(patterns: set):
    """Bulk-load permanent allowlist entries from config."""
    with _lock:
        _permanent_approved.update(patterns)


def clear_session(session_key: str):
    """Clear all approvals and pending requests for a session (e.g., on /reset)."""
    with _lock:
        _session_approved.pop(session_key, None)
        _pending.pop(session_key, None)
