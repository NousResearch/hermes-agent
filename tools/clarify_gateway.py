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
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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
    multi_select: bool = False
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[str] = None
    awaiting_text: bool = False  # set when user picked "Other" or clarify is open-ended

    def signature(self) -> Dict[str, object]:
        return {
            "clarify_id": self.clarify_id,
            "session_key": self.session_key,
            "question": self.question,
            "choices": list(self.choices) if self.choices else None,
            "multi_select": bool(self.multi_select),
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
    multi_select: bool = False,
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
        multi_select=bool(multi_select) and bool(choices),
        # Open-ended (no choices) → next message IS the response, no buttons needed.
        awaiting_text=not bool(choices),
    )
    with _lock:
        _entries[clarify_id] = entry
        _session_index.setdefault(session_key, []).append(clarify_id)
    return entry


def wait_for_response(clarify_id: str, timeout: float) -> Optional[str]:
    """Block on the entry's event until resolved or timeout fires.

    Polls in 1-second slices so the agent's inactivity heartbeat keeps
    firing — without this, ``Event.wait(timeout=600)`` blocks the thread
    for 10 minutes with zero activity touches and the gateway's inactivity
    watchdog kills the agent while the user is still typing.

    ``timeout <= 0`` means an unlimited wait (never auto-skip mid-think); the
    heartbeat still fires each slice so inactivity watchdogs don't kill a live
    prompt.

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

    # 0 / negative → unlimited: no deadline, poll forever in 1s slices.
    unlimited = timeout is None or float(timeout) <= 0.0
    deadline = None if unlimited else time.monotonic() + float(timeout)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    while True:
        if deadline is None:
            slice_s = 1.0
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            slice_s = min(1.0, remaining)
        if entry.event.wait(timeout=slice_s):
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for user clarify response")

    with _lock:
        # Remove from indices regardless of resolution outcome.
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

def get_pending_by_id(
    clarify_id: str,
    *,
    session_key: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """Return a snapshot of an unresolved clarify request.

    ``session_key`` lets transports bind an opaque request identifier to the
    run/session that owns it without exposing the mutable queue entry.
    """
    with _lock:
        entry = _entries.get(clarify_id)
        if (
            entry is None
            or entry.event.is_set()
            or (session_key is not None and entry.session_key != session_key)
        ):
            return None
        return entry.signature()


def resolve_gateway_clarify(
    clarify_id: str,
    response: str,
    *,
    session_key: Optional[str] = None,
) -> bool:
    """Unblock the agent thread waiting on ``clarify_id``.

    Returns True if an entry was found and resolved, False otherwise
    (wrong session, already resolved, expired, or never existed).
    """
    with _lock:
        entry = _entries.get(clarify_id)
        if (
            entry is None
            or entry.event.is_set()
            or (session_key is not None and entry.session_key != session_key)
        ):
            return False
        entry.response = str(response) if response is not None else ""
        entry.event.set()
        return True


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


def _label_matches(text: str, choice: object) -> bool:
    """Case-insensitive label match that ignores the '(Recommended)' suffix.

    The first choice reaches adapters already decorated (see
    ``tools.clarify_tool.mark_recommended``), so a user who types the option
    text as the agent worded it — without the label — must still resolve the
    prompt.
    """
    from tools.clarify_tool import strip_recommended

    return strip_recommended(text).casefold() == strip_recommended(str(choice)).casefold()


# Outcomes for typed clarify replies. Gateway uses these to decide whether to
# cancel a pending prompt (free prose deadlock break) or keep it armed so the
# user can retry a selection-like invalid reply (out-of-range / bad list).
TEXT_RESOLVED = "resolved"
TEXT_REJECTED_PROSE = "rejected_prose"
TEXT_REJECTED_SELECTION = "rejected_selection"
TEXT_NO_PENDING = "no_pending"


def _selection_attempt_tokens(
    text: str,
    choices: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Return tokens when ``text`` looks like a typed selection attempt.

    Selection-shaped input includes:
      - a bare integer ("2", "99")
      - comma-separated numbers/labels ("1,3", "staging, prod", "1,99")
      - space-separated all-numeric lists ("1 3")

    Free prose ("just checking the visual UI, no need to pass any data") returns
    None even when it contains commas, so the gateway can release the clarify
    and continue normal routing instead of forcing a retry.

    Multi-word choice labels are allowed in comma-lists up to the longest
    choice's word count (e.g. "Send to SOL, Keep with Enoch").
    """
    stripped = str(text).strip()
    if not stripped:
        return None

    max_choice_words = 1
    if choices:
        max_choice_words = max(
            (len(str(choice).split()) for choice in choices),
            default=1,
        )
        max_choice_words = max(1, max_choice_words)

    if "," in stripped:
        tokens = [t.strip() for t in stripped.split(",") if t.strip()]
        if not tokens:
            return None
        # Natural-language clauses with commas are not selection lists.
        # Each selection token is either a number or at most as many words
        # as the longest configured choice label.
        for token in tokens:
            if token.isdigit():
                continue
            words = token.split()
            if len(words) == 0 or len(words) > max_choice_words:
                return None
        return tokens

    parts = stripped.split()
    if len(parts) > 1 and all(p.strip().isdigit() for p in parts):
        return [p.strip() for p in parts]

    # Bare integer (in-range or out-of-range) is always a selection attempt.
    if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
        return [stripped]

    try:
        int(stripped)
        return [stripped]
    except ValueError:
        return None


def _coerce_text_response(entry: _ClarifyEntry, response: str) -> Optional[str]:
    """Map typed choice replies to canonical choice text, otherwise keep or reject custom text.

    Thin wrapper over :func:`_coerce_text_response_detailed` for callers that
    only need the accepted value (or ``None`` on any rejection).
    """
    coerced, _reason = _coerce_text_response_detailed(entry, response)
    return coerced


def _coerce_text_response_detailed(
    entry: _ClarifyEntry,
    response: str,
) -> tuple[Optional[str], Optional[str]]:
    """Map typed replies and classify rejections.

    Returns ``(value, None)`` when the reply is accepted.

    Returns ``(None, reason)`` when rejected:
      - ``"invalid_selection"`` — selection-shaped but unusable (out-of-range
        number, unrecognised comma-list). Keep the pending clarify so the
        user can retry.
      - ``"prose"`` — free text that is not a selection attempt. Gateway may
        cancel the clarify and continue normal busy-message routing so a
        redirect-to-steer path cannot deadlock behind the waiting tool.

    For native interactive multi-choice clarifies (button UI, awaiting_text=False):
      - Accept numeric selections ("2" → choice[1])
      - Accept exact choice label matches (case-insensitive)
      - Reject arbitrary prose so the message can continue as a normal turn

    For multi-select clarifies (entry.multi_select=True):
      - Accept several numbers separated by commas and/or spaces ("1,3" / "1 3")
      - Accept exact choice label matches (single or comma-separated)
      - Out-of-range numbers / unrecognised lists reject the whole reply so the
        user can retry instead of silently getting a partial selection
      - Selections are returned as a JSON array string, which the clarify
        tool's ``_parse_multi_select_response`` decodes back into a list

    For text fallback or awaiting_text mode:
      - Accept any text (numeric/label/custom) after passing through coercion

    For open-ended clarifies (no choices):
      - Accept any text
    """
    text = str(response).strip()

    if not entry.choices:
        # Open-ended: accept any text
        return text, None

    if entry.multi_select:
        coerced = _coerce_multi_select_text(entry, text)
        if coerced is not None:
            return coerced, None
        # Not a parseable selection — accept as custom text only in
        # awaiting_text mode (the "Other" path); otherwise classify reject.
        if entry.awaiting_text:
            return text, None
        if _selection_attempt_tokens(text, entry.choices) is not None:
            return None, "invalid_selection"
        return None, "prose"

    # Try numeric selection first (always valid for multi-choice)
    try:
        idx = int(text) - 1
        is_int = True
    except ValueError:
        idx = -1
        is_int = False

    if is_int and 0 <= idx < len(entry.choices):
        return entry.choices[idx], None

    # Try exact choice label match (always valid for multi-choice)
    for choice in entry.choices:
        if _label_matches(text, choice):
            return str(choice).strip(), None

    # For text fallback or awaiting_text mode, accept custom text
    # For native interactive multi-choice mode, reject with a reason
    if entry.awaiting_text:
        return text, None

    # Out-of-range / non-canonical integer is a failed selection, not prose.
    if is_int:
        return None, "invalid_selection"
    return None, "prose"


def _coerce_multi_select_text(entry: _ClarifyEntry, text: str) -> Optional[str]:
    """Parse a typed multi-select reply into a JSON array of choice labels.

    Accepts numbers and/or exact labels separated by commas (and, for
    all-numeric replies, bare spaces): "1,3", "1 3", "staging, prod".
    Returns ``None`` when any token is out of range or unrecognised so the
    caller can reject the reply cleanly instead of resolving a partial or
    wrong selection.
    """
    import json as _json

    if not text:
        return None
    choices = entry.choices or []

    # Split on commas first; if no commas and every whitespace-separated
    # token is numeric, treat spaces as separators too ("1 3").
    if "," in text:
        tokens = [t.strip() for t in text.split(",") if t.strip()]
    else:
        parts = text.split()
        if len(parts) > 1 and all(p.strip().isdigit() for p in parts):
            tokens = [p.strip() for p in parts]
        else:
            tokens = [text]

    selected: List[str] = []
    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(choices):
                label = str(choices[idx]).strip()
                if label not in selected:
                    selected.append(label)
                continue
            return None  # out-of-range number → reject whole reply
        # Exact label match (case-insensitive)
        matched = None
        for choice in choices:
            if _label_matches(token, choice):
                matched = str(choice).strip()
                break
        if matched is None:
            return None
        if matched not in selected:
            selected.append(matched)

    if not selected:
        return None
    return _json.dumps(selected, ensure_ascii=False)


def attempt_text_response_for_session(session_key: str, response: str) -> str:
    """Try to resolve the oldest pending clarify in ``session_key`` from typed text.

    Returns one of:
      - ``TEXT_RESOLVED`` — accepted; waiter unblocked
      - ``TEXT_REJECTED_PROSE`` — free prose on a native choice prompt; caller
        may cancel the clarify and continue ordinary message routing
      - ``TEXT_REJECTED_SELECTION`` — selection-shaped but invalid; leave the
        pending clarify armed so the user can retry
      - ``TEXT_NO_PENDING`` — no interceptable clarify for this session
    """
    entry = get_pending_for_session(session_key, include_choice_prompts=True)
    if entry is None:
        return TEXT_NO_PENDING

    coerced, reason = _coerce_text_response_detailed(entry, response)
    if coerced is None:
        if reason == "invalid_selection":
            return TEXT_REJECTED_SELECTION
        return TEXT_REJECTED_PROSE

    if resolve_gateway_clarify(entry.clarify_id, coerced):
        return TEXT_RESOLVED
    # Lost a race with a button/callback resolution — treat as no work left.
    return TEXT_NO_PENDING


def resolve_text_response_for_session(session_key: str, response: str) -> bool:
    """Resolve the oldest pending clarify in ``session_key`` from typed text.

    Returns True only when the reply was accepted and the waiter unblocked.
    Rejected prose, rejected selections, and missing prompts all return False;
    use :func:`attempt_text_response_for_session` when the caller must
    distinguish those cases (gateway deadlock vs multi-select retry).
    """
    return attempt_text_response_for_session(session_key, response) == TEXT_RESOLVED


def mark_awaiting_text(clarify_id: str) -> bool:
    """Flip an entry into text-capture mode (user picked the 'Other' button).

    Returns True if the entry exists and was flipped, False otherwise.
    """
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None:
            return False
        entry.awaiting_text = True
        return True


def has_pending(session_key: str) -> bool:
    """Return True when this session has at least one pending clarify entry."""
    with _lock:
        ids = _session_index.get(session_key) or []
        return any(_entries.get(cid) is not None for cid in ids)


def clear_session(session_key: str) -> int:
    """Resolve and drop every pending clarify for a session.

    Used by session-boundary cleanup (e.g. ``/new``, gateway shutdown,
    cached-agent eviction) so blocked agent threads don't hang past the
    end of their session.  Returns the number of entries actually
    cancelled (i.e. whose event had not yet been set).  Already-resolved
    entries are dropped from the registry but their response is preserved.

    First-writer-wins: an entry whose event is already set has been resolved
    by a real response (button callback or text intercept).  Session cleanup
    must NOT overwrite that response with the empty cancellation sentinel —
    the waiting agent thread would observe a cancelled prompt even though the
    user answered.  Only unresolved entries are cancelled here.
    """
    with _lock:
        ids = list(_session_index.pop(session_key, []) or [])
        entries = [_entries.pop(cid, None) for cid in ids]
        # The mutation loop must stay inside the lock: the pop above and the
        # event.is_set() check below have to be atomic with respect to
        # resolve_gateway_clarify, or a button callback could win between the
        # pop and the check and have its answer clobbered by the sentinel.
        cancelled = 0
        for entry in entries:
            if entry is None:
                continue
            # Entry is removed from the global registry regardless of its
            # state — a cleared session must not be resurrected by late
            # callbacks — but a resolved entry keeps its real response.
            if entry.event.is_set():
                continue
            # Empty string sentinel — agent code can distinguish from a real
            # response by inspecting the wait_for_response return value
            # alongside its own timeout deadline.  Most callers just treat any
            # falsy result as "user did not respond".
            entry.response = ""
            entry.event.set()
            cancelled += 1
    return cancelled


# =========================================================================
# Config
# =========================================================================

def resolve_clarify_timeout(config: dict) -> int:
    """Resolve the clarify timeout (seconds) from an already-loaded config dict.

    Single source of truth shared by every surface (messaging gateway, CLI,
    TUI/desktop) so the timeout can't drift between them.  Resolution order:

    1. legacy top-level ``clarify.timeout`` if a user explicitly set it,
    2. else the canonical ``agent.clarify_timeout``,
    3. else 3600 (1 hour).

    ``<= 0`` is preserved verbatim and means *unlimited* to callers (never
    auto-skip while the user is still deciding); the waiting loops translate
    that into a null deadline.  A non-numeric value falls back to 3600.
    """
    raw = (config.get("clarify") or {}).get("timeout")
    if raw is None:
        raw = (config.get("agent") or {}).get("clarify_timeout", 3600)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 3600


def get_clarify_timeout() -> int:
    """Read the clarify response timeout (seconds) from config.

    Defaults to 3600 (1 hour) — long enough that a user who steps away
    (meeting, AFK, slow to read) still finds a live entry when they tap
    the button, short enough that a genuinely abandoned prompt eventually
    unblocks the agent thread instead of pinning the running-agent guard
    forever.  The old 600s default evicted the entry mid-think, so a late
    tap landed on a dead entry and the agent hung on ``running: clarify``
    (#32762).

    Reads ``agent.clarify_timeout`` from config.yaml (see
    :func:`resolve_clarify_timeout` for the full resolution order).  Set to
    ``0`` (or negative) for an unlimited wait — never auto-skip while the user
    is still deciding.
    """
    try:
        from hermes_cli.config import load_config
        return resolve_clarify_timeout(load_config() or {})
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
