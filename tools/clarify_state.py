"""Gateway clarify bridge -- per-session blocking queue for interactive questions.

Mirrors the approval system's architecture (tools/approval.py):
- Agent thread blocks on a ``threading.Event`` waiting for user input.
- Gateway platform adapters send the prompt (card, button, text) and
  register a resolve callback.
- When the user answers (clicks a button, types text), the adapter calls
  ``resolve_gateway_clarify()`` to unblock the agent thread with the answer.

This module is the *core* bridge; platform-specific UI lives in each adapter
(e.g. ``gateway/platforms/feishu.py`` sends interactive cards).
"""

import contextvars
import logging
import threading
import uuid
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum time (seconds) to wait for a user response before giving up.
# Matches the approval system's _APPROVAL_TIMEOUT_SECONDS.
_CLARIFY_TIMEOUT_SECONDS = 300

# Per-thread/per-task gateway session identity (mirrors approval.py).
_clarify_session_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "clarify_session_key",
    default="",
)


def set_clarify_session_key(session_key: str) -> contextvars.Token[str]:
    """Bind the active clarify session key to the current context."""
    return _clarify_session_key.set(session_key or "")


def reset_clarify_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior clarify session key context."""
    _clarify_session_key.reset(token)


def get_clarify_session_key(default: str = "default") -> str:
    """Return the active session key for clarify, preferring context-local state."""
    session_key = _clarify_session_key.get()
    if session_key:
        return session_key
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", default)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class _ClarifyEntry:
    """One pending clarify question inside a gateway session."""
    __slots__ = ("event", "data", "result", "clarify_id")

    def __init__(self, data: dict):
        self.event = threading.Event()
        self.data = data  # question, choices, clarify_id, …
        self.result: Optional[str] = None  # user's answer text
        self.clarify_id: str = data.get("clarify_id", "")


_lock = threading.Lock()

# session_key → [_ClarifyEntry, …]
_gateway_queues: Dict[str, List[_ClarifyEntry]] = {}

# session_key → callable(clarify_data: dict) -> None
# Called from the agent thread to send the prompt to the user.
_gateway_notify_cbs: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Public API -- used by gateway/run.py
# ---------------------------------------------------------------------------

def register_clarify_notify(session_key: str, cb: Callable) -> None:
    """Register a per-session callback for sending clarify prompts to the user.

    The callback signature is ``cb(clarify_data: dict) -> None`` where
    *clarify_data* contains ``question``, ``choices``, and ``clarify_id``.
    The callback bridges sync→async (runs in the agent thread, must schedule
    the actual send on the event loop).
    """
    with _lock:
        _gateway_notify_cbs[session_key] = cb


def unregister_clarify_notify(session_key: str) -> None:
    """Unregister the per-session gateway clarify callback.

    Signals ALL blocked threads for this session so they don't hang forever
    (e.g. when the agent run finishes or is interrupted).
    """
    with _lock:
        _gateway_notify_cbs.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        entry.result = ""
        entry.event.set()


def resolve_gateway_clarify(session_key: str, answer: str,
                            clarify_id: Optional[str] = None) -> int:
    """Called by the gateway platform adapter to unblock the waiting agent thread.

    When *clarify_id* is provided, only the matching entry is resolved
    (enables multiple concurrent clarify prompts).  Otherwise the oldest
    pending entry is resolved (FIFO, backward compatible).

    Returns the number of entries resolved (0 means nothing was pending).
    """
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return 0

        # If clarify_id given, find and resolve that specific entry.
        if clarify_id:
            target = None
            for i, entry in enumerate(queue):
                if entry.clarify_id == clarify_id:
                    target = queue.pop(i)
                    break
            if target is None:
                return 0
            targets = [target]
        else:
            # FIFO: resolve the oldest pending clarify.
            targets = [queue.pop(0)]

        if not queue:
            _gateway_queues.pop(session_key, None)

    for entry in targets:
        entry.result = answer
        entry.event.set()
    return len(targets)


def has_blocking_clarify(session_key: str) -> bool:
    """Check if a session has one or more blocking gateway clarify prompts waiting."""
    with _lock:
        return bool(_gateway_queues.get(session_key))


# ---------------------------------------------------------------------------
# Blocking clarify callback -- passed to AIAgent as clarify_callback
# ---------------------------------------------------------------------------

def gateway_clarify_callback(question: str, choices: Optional[List[str]] = None) -> str:
    """Blocking callback suitable for passing to ``AIAgent(clarify_callback=…)``.

    This function:
    1. Creates a ``_ClarifyEntry`` with a unique ``clarify_id``.
    2. Enqueues it on the session's queue.
    3. Calls the registered notify callback to send the prompt to the user.
    4. Blocks the agent thread until the user answers (or the session is torn down).

    Returns the user's answer string.
    """
    session_key = get_clarify_session_key()
    clarify_id = f"clr_{uuid.uuid4().hex[:8]}"

    entry = _ClarifyEntry({
        "question": question,
        "choices": choices,
        "clarify_id": clarify_id,
        "session_key": session_key,
    })

    with _lock:
        queue = _gateway_queues.setdefault(session_key, [])
        queue.append(entry)
        notify_cb = _gateway_notify_cbs.get(session_key)

    if notify_cb:
        try:
            notify_cb(entry.data)
        except Exception as exc:
            logger.error("Clarify notify callback failed: %s", exc)

    # Block until resolved or session teardown.
    # Timeout prevents the agent thread from hanging indefinitely if the
    # user never responds (e.g. closes the chat, goes offline).
    if not entry.event.wait(timeout=_CLARIFY_TIMEOUT_SECONDS):
        # Timed out — remove from queue so it doesn't leak.
        with _lock:
            queue = _gateway_queues.get(session_key)
            if queue:
                try:
                    queue.remove(entry)
                except ValueError:
                    pass
                if not queue:
                    _gateway_queues.pop(session_key, None)
        logger.warning("Clarify prompt timed out after %ds (session=%s, id=%s)",
                        _CLARIFY_TIMEOUT_SECONDS, session_key, clarify_id)

    return entry.result if entry.result is not None else ""
