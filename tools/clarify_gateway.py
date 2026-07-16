"""Gateway-side clarify primitive (blocking event-based queue).

The ``clarify`` tool needs to ask the user a question and block the agent
thread until they respond.  In CLI mode this is trivial — ``input()`` is
synchronous.  In gateway mode the agent runs on a worker thread while the
event loop handles the user's reply, so we need a thread-safe primitive
that:

  * stores a pending clarify request (with a generated ``clarify_id``),
  * blocks the agent thread on an ``Event``,
  * resolves the wait when the gateway's button-callback or text-intercept
    fires ``resolve_gateway_clarify(clarify_id, response)``,
  * supports timeouts so a user who never responds does NOT hang the agent
    thread forever (which would also pin the gateway's running-agent guard).

State is module-level (same shape as ``tools.approval``) so platform
adapters can call ``resolve_gateway_clarify`` without holding a back-
reference to the ``GatewayRunner`` instance.

Two delivery paths from the adapter:

  1. **Button UI** — adapters override ``send_clarify`` to render inline
     buttons (e.g. Telegram ``InlineKeyboardMarkup``).  The button
     callback resolves with the chosen string.  A final "Other (type
     answer)" button enters text-capture mode for free-form responses.

  2. **Text fallback** — adapters without rich UI render a numbered list.
     The user replies with a number ("2") or with free text; the gateway's
     ``_handle_message`` intercepts the reply and resolves directly.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# =========================================================================
# Module-level state
# =========================================================================

@dataclass
class _ClarifyEntry:
    """One pending clarify request inside a gateway session."""
    clarify_id: str
    session_key: str
    question: str
    choices: Optional[List[str]]
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Any] = None
    awaiting_text: bool = False  # set when user picked "Other" or clarify is open-ended
    multiple: bool = False
    generation: Optional[int] = None

    def signature(self) -> Dict[str, object]:
        return {
            "clarify_id": self.clarify_id,
            "session_key": self.session_key,
            "question": self.question,
            "choices": list(self.choices) if self.choices else None,
            "multiple": self.multiple,
            "generation": self.generation,
        }


_lock = threading.RLock()
# clarify_id → _ClarifyEntry  (primary lookup for button callbacks)
_entries: Dict[str, _ClarifyEntry] = {}
# session_key → list[clarify_id]  (FIFO; for text-fallback intercept and session cleanup)
_session_index: Dict[str, List[str]] = {}


# =========================================================================
# Public API — agent-thread side
# =========================================================================

def register(
    clarify_id: str,
    session_key: str,
    question: str,
    choices: Optional[List[str]],
    *,
    multiple: bool = False,
    generation: Optional[int] = None,
) -> _ClarifyEntry:
    """Register a pending clarify request and return the entry.

    The caller (gateway clarify_callback) will then send the prompt to the
    user and block on ``wait_for_response(clarify_id, timeout)``.
    """
    entry = _ClarifyEntry(
        clarify_id=clarify_id,
        session_key=session_key,
        question=question,
        choices=list(choices) if choices else None,
        # Open-ended (no choices) → next message IS the response, no buttons needed.
        awaiting_text=not bool(choices),
        multiple=multiple,
        generation=generation,
    )
    with _lock:
        _entries[clarify_id] = entry
        _session_index.setdefault(session_key, []).append(clarify_id)
    return entry


def register_select_many(
    select_id: str,
    session_key: str,
    question: str,
    choices: List[str],
    *,
    generation: Optional[int] = None,
) -> _ClarifyEntry:
    """Register an explicit multi-select request in the shared wait queue."""
    return register(
        clarify_id=select_id,
        session_key=session_key,
        question=question,
        choices=choices,
        multiple=True,
        generation=generation,
    )


def wait_for_response(clarify_id: str, timeout: float) -> Optional[Any]:
    """Block on the entry's event until resolved or timeout fires.

    Polls in 1-second slices so the agent's inactivity heartbeat keeps
    firing — without this, ``Event.wait(timeout=600)`` blocks the thread
    for 10 minutes with zero activity touches and the gateway's inactivity
    watchdog kills the agent while the user is still typing.

    Returns the resolved response string, or ``None`` on timeout.
    """
    with _lock:
        entry = _entries.get(clarify_id)
    if entry is None:
        return None

    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - optional
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
            touch_activity_if_due(activity_state, "waiting for user clarify response")

    with _lock:
        # Remove only the entry this waiter owns. A replacement using the same
        # id must survive a late old-worker cleanup.
        if _entries.get(clarify_id) is entry:
            _entries.pop(clarify_id, None)
            ids = _session_index.get(entry.session_key)
            if ids and clarify_id in ids:
                ids.remove(clarify_id)
                if not ids:
                    _session_index.pop(entry.session_key, None)

    return entry.response


# =========================================================================
# Public API — gateway / adapter side
# =========================================================================

def resolve_gateway_clarify(clarify_id: str, response: str) -> bool:
    """Unblock the agent thread waiting on ``clarify_id``.

    Returns True if an entry was found and resolved, False otherwise
    (already resolved, expired, or never existed).
    """
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None or entry.multiple or entry.event.is_set():
            return False
        # Keep the accepted value and the event publication under the same
        # lock.  Button and text callbacks can arrive on different threads;
        # checking ``is_set`` and writing separately lets a late responder
        # overwrite the first answer.
        entry.response = str(response) if response is not None else ""
        entry.event.set()
        return True


def resolve_gateway_select_many(select_id: str, responses: Sequence[str]) -> bool:
    """Resolve a multi-select request with canonical offered choice labels."""
    if not isinstance(responses, (list, tuple)):
        return False
    with _lock:
        entry = _entries.get(select_id)
        if entry is None or not entry.multiple or entry.event.is_set():
            return False
        offered = set(entry.choices or [])
        selected: List[str] = []
        for response in responses:
            value = str(response).strip()
            if value not in offered:
                return False
            if value not in selected:
                selected.append(value)
        entry.response = selected
        entry.event.set()
        return True


def cancel_gateway_select_many(select_id: str) -> bool:
    """Resolve a multi-select request as an explicit user cancellation."""
    return resolve_gateway_select_many(select_id, [])


def get_pending_for_session(
    session_key: str,
    *,
    include_choice_prompts: bool = False,
) -> Optional[_ClarifyEntry]:
    """Return the oldest pending clarify entry for a session, or None.

    By default this only returns entries awaiting free-form text (open-ended
    clarifies, or a multi-choice clarify after the user picked ``Other``).
    Gateways may pass ``include_choice_prompts=True`` when the user has typed
    directly in response to an active multi-choice prompt; in that case the
    oldest unresolved clarify is returned so the text can resolve it instead
    of being queued as an unrelated follow-up turn.
    """
    with _lock:
        ids = _session_index.get(session_key) or []
        for cid in ids:
            entry = _entries.get(cid)
            if entry is None:
                continue
            if include_choice_prompts or entry.awaiting_text:
                return entry
        return None


def _coerce_text_response(entry: _ClarifyEntry, response: str) -> str:
    """Map typed choice replies to canonical choice text, otherwise keep custom text."""
    text = str(response).strip()
    if entry.choices:
        try:
            idx = int(text) - 1
        except ValueError:
            idx = -1
        if 0 <= idx < len(entry.choices):
            return entry.choices[idx]
        for choice in entry.choices:
            if text.casefold() == str(choice).strip().casefold():
                return str(choice).strip()
    return text


def _coerce_select_many_text(
    entry: _ClarifyEntry,
    response: str,
) -> Optional[List[str]]:
    """Parse a numbered multi-select reply, returning None when it is invalid."""
    text = str(response or "").strip()
    if text.casefold() in {"cancel", "cancelled", "canceled", "取消"}:
        return []
    if not text or not entry.choices:
        return None

    tokens = [token for token in re.split(r"[\s,，、]+", text) if token]
    if not tokens:
        return None

    selected: List[str] = []
    for token in tokens:
        if not token.isdigit():
            return None
        index = int(token) - 1
        if not 0 <= index < len(entry.choices):
            return None
        choice = str(entry.choices[index])
        if choice not in selected:
            selected.append(choice)
    return selected


def select_many_text_help(entry: _ClarifyEntry) -> str:
    """Return retry guidance for an invalid numbered multi-select reply."""
    count = len(entry.choices or [])
    return (
        f"Reply with one or more numbers from 1 to {count}, separated by spaces "
        "(for example: 1 2 3), or reply `cancel`."
    )


def resolve_text_response_for_session(session_key: str, response: str) -> bool:
    """Resolve the oldest pending clarify in ``session_key`` from typed text."""
    entry = get_pending_for_session(session_key, include_choice_prompts=True)
    if entry is None:
        return False
    if entry.multiple:
        selected = _coerce_select_many_text(entry, response)
        if selected is None:
            return False
        return resolve_gateway_select_many(entry.clarify_id, selected)
    return resolve_gateway_clarify(
        entry.clarify_id,
        _coerce_text_response(entry, response),
    )


def mark_awaiting_text(clarify_id: str) -> bool:
    """Flip an entry into text-capture mode (user picked the 'Other' button).

    Returns True if the entry exists and was flipped, False otherwise.
    """
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None or entry.event.is_set():
            return False
        entry.awaiting_text = True
        return True


def has_pending(session_key: str) -> bool:
    """Return True when this session has at least one pending clarify entry."""
    with _lock:
        ids = _session_index.get(session_key) or []
        return any(_entries.get(cid) is not None for cid in ids)


def clear_session(session_key: str, *, generation: Optional[int] = None) -> int:
    """Resolve and drop every pending clarify for a session.

    Used by session-boundary cleanup (e.g. ``/new``, gateway shutdown,
    cached-agent eviction) so blocked agent threads don't hang past the
    end of their session.  Returns the number of entries cancelled.
    """
    with _lock:
        ids = list(_session_index.get(session_key, []) or [])
        remaining_ids = []
        cancelled = 0
        for cid in ids:
            entry = _entries.get(cid)
            if entry is None:
                continue
            if generation is not None and entry.generation != generation:
                remaining_ids.append(cid)
                continue
            _entries.pop(cid, None)
            if not entry.event.is_set():
                # Removal and cancellation publication share the resolver lock:
                # whichever decision acquired it first keeps its response.
                entry.response = ""
                entry.event.set()
                cancelled += 1
        if remaining_ids:
            _session_index[session_key] = remaining_ids
        else:
            _session_index.pop(session_key, None)
        return cancelled


# =========================================================================
# Config
# =========================================================================

def get_clarify_timeout() -> int:
    """Read the clarify response timeout (seconds) from config.

    Defaults to 3600 (1 hour) — long enough that a user who steps away
    (meeting, AFK, slow to read) still finds a live entry when they tap
    the button, short enough that a genuinely abandoned prompt eventually
    unblocks the agent thread instead of pinning the running-agent guard
    forever.  The old 600s default evicted the entry mid-think, so a late
    tap landed on a dead entry and the agent hung on ``running: clarify``
    (#32762).

    Reads ``agent.clarify_timeout`` from config.yaml.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        agent_cfg = cfg.get("agent", {}) or {}
        return int(agent_cfg.get("clarify_timeout", 3600))
    except Exception:
        return 3600


# =========================================================================
# Per-session notify hook (gateway → adapter bridge)
# =========================================================================
# Mirrors tools.approval's _gateway_notify_cbs: the gateway registers a
# per-session callback that sends the clarify prompt to the user.  The
# callback bridges sync→async (runs on the agent thread; schedules the
# adapter ``send_clarify`` call on the event loop).

_notify_cbs: Dict[str, Callable[[_ClarifyEntry], None]] = {}


def register_notify(session_key: str, cb: Callable[[_ClarifyEntry], None]) -> None:
    """Register a per-session notify callback used by ``clarify_callback``."""
    with _lock:
        _notify_cbs[session_key] = cb


def unregister_notify(session_key: str) -> None:
    """Drop the per-session notify callback and cancel any pending clarify entries."""
    with _lock:
        _notify_cbs.pop(session_key, None)
    # Cancel any pending entries so blocked threads unwind when the run
    # ends (interrupt, completion, gateway shutdown).
    clear_session(session_key)


def get_notify(session_key: str) -> Optional[Callable[[_ClarifyEntry], None]]:
    with _lock:
        return _notify_cbs.get(session_key)
