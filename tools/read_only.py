"""Read-only mode — per-session flag that blocks all non-whitelisted tools.

Usage
-----
    from tools.read_only import (
        enable_read_only, disable_read_only,
        is_read_only, is_tool_allowed, READ_ONLY_WHITELIST,
    )

    if is_read_only(session_key) and not is_tool_allowed(function_name):
        return tool_error(f"Read-only mode: '{function_name}' is blocked. "
                          "Only whitelisted tools are permitted. "
                          "Ask the user to run /ro to disable read-only mode.")

Thread-safe via a module-level lock (same pattern as tools/approval.py).
"""

from __future__ import annotations

import logging
import threading
from typing import FrozenSet

logger = logging.getLogger(__name__)

# ── Whitelist ──────────────────────────────────────────────────────────
# Tools allowed in read-only mode.  Every name here is treated as a
# case-sensitive exact match.  Add new tools here if they are pure reads
# that cannot modify system state.
#
# NOTE: ``session_search``, ``todo``, ``memory`` are dispatched by the
# agent loop (run_agent.py) and never reach the central tool dispatcher,
# so they are NOT listed here — they are never blocked by read-only mode
# in the first place and adding them would be a no-op.
_READ_ONLY_WHITELIST: FrozenSet[str] = frozenset({
    # File read
    "read_file",
    "search_files",

    # Web / network — pure reads
    "web_search",
    "web_extract",

    # Browser automation — all browser_* tools are read-only interactions
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_scroll",
    "browser_type",
    "browser_press",
    "browser_console",
    "browser_get_images",
    "browser_back",

    # Session history — pure read
    "session_search",

    # Skill browsing — pure reads
    "skill_view",
    "skills_list",

    # Clarify — asking the user questions, no modification
    "clarify",

    # Vision — read-only analysis of images
    "vision_analyze",
})

# ── Per-session state ──────────────────────────────────────────────────
_lock = threading.Lock()
_read_only_sessions: set[str] = set()

# ContextVar-based current session key (set by the agent loop)
import contextvars  # noqa: E402
_current_read_only_session: contextvars.ContextVar[str] = (
    contextvars.ContextVar("_current_read_only_session", default="")
)


def set_current_session_key(key: str) -> contextvars.Token[str]:
    """Bind the read-only check to a specific session key.
    
    Returns a token that can be passed to ``reset_current_session_key``
    to restore the prior value.
    """
    return _current_read_only_session.set(key)


def reset_current_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior read-only session key context."""
    _current_read_only_session.reset(token)


def get_current_session_key(default: str = "") -> str:
    """Return the active session key, or *default* when unset."""
    return _current_read_only_session.get(default)


def enable_read_only(session_key: str) -> None:
    """Enable read-only mode for a single session key."""
    if not session_key:
        return
    with _lock:
        _read_only_sessions.add(session_key)
    logger.info("Read-only mode ENABLED for session '%s'", session_key)


def disable_read_only(session_key: str) -> None:
    """Disable read-only mode for a single session key."""
    if not session_key:
        return
    with _lock:
        _read_only_sessions.discard(session_key)
    logger.info("Read-only mode DISABLED for session '%s'", session_key)


def is_read_only(session_key: str) -> bool:
    """Return True when read-only mode is active for a specific session."""
    if not session_key:
        return False
    with _lock:
        return session_key in _read_only_sessions


def is_current_read_only() -> bool:
    """Return True when the active session has read-only mode enabled."""
    return is_read_only(get_current_session_key(default=""))


def is_tool_allowed(function_name: str) -> bool:
    """Return True when *function_name* is on the read-only whitelist.

    Always returns True when read-only mode is *not* active for the
    current session, so this check can be inserted unconditionally::

        if not is_tool_allowed(function_name):
            return tool_error(...)

    When no session key is bound (current session key is empty),
    read-only mode is considered inactive.
    """
    key = get_current_session_key(default="")
    if not key or not is_read_only(key):
        return True  # read-only not active → all tools allowed
    return function_name in _READ_ONLY_WHITELIST


# ── Public constant for introspection ──────────────────────────────────
READ_ONLY_WHITELIST: FrozenSet[str] = _READ_ONLY_WHITELIST
