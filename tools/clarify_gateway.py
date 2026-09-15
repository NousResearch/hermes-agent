"""Gateway-side clarify primitive (blocking event-based queue). The agent runs on a worker
thread while the event loop handles the user's reply, so a pending clarify is stored
module-level (same shape as ``tools.approval``) and the agent thread blocks on an ``Event``
until an adapter button callback or the gateway text-intercept resolves it, or the timeout
fires. Adapters render inline buttons (an "Other" row flips the entry into text-capture
mode) or a numbered-list text fallback."""

from __future__ import annotations
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


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


_lock = threading.RLock()
_entries: Dict[str, _ClarifyEntry] = {}  # clarify_id -> entry (button callbacks)
_session_index: Dict[str, List[str]] = {}  # session_key -> [clarify_id] FIFO (text intercept, cleanup)
# Per-session notify callbacks (gateway -> adapter bridge); mirrors tools.approval. Tests clear it.
_notify_cbs: Dict[str, Callable[[_ClarifyEntry], None]] = {}

# Outcomes for typed clarify replies. Gateway cancels the pending prompt on
# free prose (deadlock break) but keeps it armed for a retryable bad selection.
TEXT_RESOLVED = "resolved"
TEXT_REJECTED_PROSE = "rejected_prose"
TEXT_REJECTED_SELECTION = "rejected_selection"
TEXT_NO_PENDING = "no_pending"


def register(clarify_id: str, session_key: str, question: str, choices: Optional[List[str]],
             multi_select: bool = False) -> _ClarifyEntry:
    """Register a pending clarify request; caller then blocks on ``wait_for_response``.
    Open-ended (no choices) entries start in text mode: the next message IS the response."""
    entry = _ClarifyEntry(clarify_id, session_key, question, list(choices) if choices else None,
                          bool(multi_select) and bool(choices), awaiting_text=not bool(choices))
    with _lock:
        _entries[clarify_id] = entry
        _session_index.setdefault(session_key, []).append(clarify_id)
    return entry


def wait_for_response(clarify_id: str, timeout: float) -> Optional[str]:
    """Block until the entry resolves or ``timeout`` (``<= 0`` = unlimited) elapses; None on
    timeout/unknown id. Polls in 1s slices so the inactivity heartbeat keeps firing (a
    single long ``Event.wait`` would let the gateway watchdog kill a live prompt)."""
    with _lock:
        entry = _entries.get(clarify_id)
    if entry is None:
        return None
    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - optional
        touch_activity_if_due = None
    deadline = None if timeout is None or float(timeout) <= 0.0 else time.monotonic() + float(timeout)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    while True:
        remaining = 1.0 if deadline is None else deadline - time.monotonic()
        if remaining <= 0 or entry.event.wait(timeout=min(1.0, remaining)):
            break
        # Periodic activity touch so the gateway's inactivity timeout doesn't kill the agent during long
        # code execution (#10807).
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for user clarify response")
    with _lock:
        _entries.pop(clarify_id, None)  # regardless of outcome
        ids = _session_index.get(entry.session_key) or []
        if clarify_id in ids:
            ids.remove(clarify_id)
            if not ids:
                _session_index.pop(entry.session_key, None)
    return entry.response


def resolve_gateway_clarify(clarify_id: str, response: str) -> bool:
    """Unblock the waiter on ``clarify_id``; False if already resolved/expired/unknown."""
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None or entry.event.is_set():
            return False
        entry.response = str(response) if response is not None else ""
        entry.event.set()
        return True


def _get_pending_for_session_locked(
    session_key: str, *, include_choice_prompts: bool = False
) -> Optional[_ClarifyEntry]:
    """Oldest unresolved matching entry; caller holds ``_lock``."""
    for cid in _session_index.get(session_key) or []:
        entry = _entries.get(cid)
        if entry is not None and not entry.event.is_set() and (
            include_choice_prompts or entry.awaiting_text
        ):
            return entry
    return None


def get_pending_for_session(session_key: str, *, include_choice_prompts: bool = False) -> Optional[_ClarifyEntry]:
    """Oldest pending entry awaiting free text (open-ended, or after "Other");
    ``include_choice_prompts=True`` returns the oldest unresolved entry of any kind (user
    typed at an active choice prompt: resolve it rather than queue a follow-up turn)."""
    with _lock:
        return _get_pending_for_session_locked(
            session_key, include_choice_prompts=include_choice_prompts
        )


def _match_label(text: str, choices: List[str]) -> Optional[str]:
    """Stripped choice text matching ``text`` case-insensitively, ignoring the '(Recommended)'
    suffix the first choice carries by the time it reaches adapters; None if no match."""
    from tools.clarify_tool import strip_recommended
    wanted = strip_recommended(text).casefold()
    for choice in choices:
        if strip_recommended(str(choice)).casefold() == wanted:
            return str(choice).strip()
    return None


def _split_tokens(text: str) -> Optional[List[str]]:
    """Comma-separated tokens, or space-separated all-numeric tokens ("1 3"); else None."""
    if "," in text:
        return [t.strip() for t in text.split(",")]
    parts = text.split()
    return parts if len(parts) > 1 and all(p.isdigit() for p in parts) else None


def _is_int(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False


def _looks_int(text: str) -> bool:
    """True for parseable integers and digit-like inputs that ``int`` rejects."""
    stripped = text[1:] if text.startswith("-") else text
    return bool(stripped) and (stripped.isdigit() or _is_int(text))


def _selection_attempt_tokens(text: str, choices: Optional[List[str]] = None) -> Optional[List[str]]:
    """Selection tokens when numeric or exact-label evidence exists; else None."""
    stripped = str(text).strip()
    if not stripped:
        return None
    tokens = _split_tokens(stripped)
    if tokens is None:
        return [stripped] if _looks_int(stripped) else None
    if "," not in stripped:
        return tokens or None
    nonempty = [token for token in tokens if token]
    if not nonempty:
        return None
    evidence = any(
        _looks_int(token) or _match_label(token, choices or []) is not None
        for token in nonempty
    )
    return tokens if evidence else None


def _coerce_text_response(entry: _ClarifyEntry, response: str) -> Optional[str]:
    """Accepted value for a typed reply, or None on any rejection."""
    return _coerce_text_response_detailed(entry, response)[0]


def _coerce_text_response_detailed(entry: _ClarifyEntry, response: str) -> tuple[Optional[str], Optional[str]]:
    """Map a typed reply to ``(value, None)`` or ``(None, reason)``: ``"invalid_selection"``
    (selection-shaped but out of range/unrecognised — keep the clarify armed for a retry) or
    ``"prose"`` (free text on a native choice prompt — the gateway may cancel and route normally
    so a redirect-to-steer path cannot deadlock behind the waiting tool). Open-ended entries and
    ``awaiting_text`` accept any text; numeric picks and exact labels always resolve; multi-select
    returns a JSON array string (decoded tool-side); one bad token rejects the whole reply."""
    text = str(response).strip()
    if not entry.choices:
        return text, None
    if entry.multi_select:
        coerced = _coerce_multi_select_text(entry, text)
        selection_shaped = _selection_attempt_tokens(text, entry.choices) is not None
    else:
        selection_shaped = _looks_int(text)
        try:
            idx = int(text) - 1
        except ValueError:
            idx = -1
        coerced = entry.choices[idx] if 0 <= idx < len(entry.choices) else _match_label(text, entry.choices)
    if coerced is not None:
        return coerced, None
    if entry.awaiting_text:
        return text, None
    return None, "invalid_selection" if selection_shaped else "prose"


def _coerce_multi_select_text(entry: _ClarifyEntry, text: str) -> Optional[str]:
    """Parse "1,3" / "1 3" / "staging, prod" into a JSON array of choice labels;
    None when any token is out of range or unrecognised (reject the whole reply)."""
    choices, selected = entry.choices or [], []
    if not text:
        return None
    whole_label = _match_label(text, choices)
    if whole_label is not None:
        return json.dumps([whole_label], ensure_ascii=False)
    tokens = _split_tokens(text)
    for token in [text] if tokens is None else tokens:
        if not token:
            return None
        if _looks_int(token):
            try:
                idx = int(token) - 1
            except ValueError:
                return None
            label = str(choices[idx]).strip() if 0 <= idx < len(choices) else None
        else:
            label = _match_label(token, choices)
        if label is None:
            return None
        if label not in selected:
            selected.append(label)
    return json.dumps(selected, ensure_ascii=False) if selected else None


def attempt_text_response_for_session(session_key: str, response: str) -> str:
    """Atomically classify and resolve the oldest pending clarify from typed text."""
    with _lock:
        entry = _get_pending_for_session_locked(session_key, include_choice_prompts=True)
        if entry is None:
            return TEXT_NO_PENDING
        coerced, reason = _coerce_text_response_detailed(entry, response)
        if coerced is None:
            if reason == "invalid_selection":
                return TEXT_REJECTED_SELECTION
            resolve_gateway_clarify(entry.clarify_id, "")
            return TEXT_REJECTED_PROSE
        return (
            TEXT_RESOLVED
            if resolve_gateway_clarify(entry.clarify_id, coerced)
            else TEXT_NO_PENDING
        )


def resolve_text_response_for_session(session_key: str, response: str) -> bool:
    """True only when the typed reply was accepted and the waiter unblocked."""
    return attempt_text_response_for_session(session_key, response) == TEXT_RESOLVED


def mark_awaiting_text(clarify_id: str) -> bool:
    """Flip an entry into text-capture mode (user picked 'Other'); False if unknown."""
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is not None:
            entry.awaiting_text = True
        return entry is not None


def has_pending(session_key: str) -> bool:
    """True when this session has at least one unresolved clarify entry."""
    with _lock:
        return any(
            entry is not None and not entry.event.is_set()
            for cid in _session_index.get(session_key) or []
            if (entry := _entries.get(cid)) is not None
        )


def clear_session(session_key: str) -> int:
    """Drop every pending clarify for a session (``/new``, shutdown, cached-agent eviction) so
    blocked agent threads don't outlive it; returns how many were cancelled. Cancelled waiters
    see "" (callers tell it from a real reply only via their own timeout bookkeeping; most treat
    any falsy result as no response). First-writer-wins: an already-set entry was answered for
    real, so it is dropped but its response preserved. The loop stays inside the lock so a button
    callback cannot slip between pop and check; entries go regardless of state so a cleared
    session is never resurrected by late callbacks."""
    with _lock:
        cancelled = 0
        for entry in (_entries.pop(cid, None) for cid in list(_session_index.pop(session_key, []) or [])):
            if entry is None or entry.event.is_set():
                continue
            entry.response = ""
            entry.event.set()
            cancelled += 1
    return cancelled


def resolve_clarify_timeout(config: dict) -> int:
    """Clarify timeout (seconds): legacy ``clarify.timeout`` if explicitly set, else
    ``agent.clarify_timeout``, else 3600 — the single source of truth for every surface
    (gateway, CLI, TUI). ``<= 0`` is kept verbatim (unlimited); non-numeric -> 3600."""
    raw = (config.get("clarify") or {}).get("timeout")
    if raw is None:
        raw = (config.get("agent") or {}).get("clarify_timeout", 3600)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 3600


def get_clarify_timeout() -> int:
    """Clarify timeout from config.yaml; 0/negative = unlimited. Default 3600: long enough
    that a user who stepped away still finds a live entry when they tap, short enough that
    an abandoned prompt eventually unblocks the agent thread instead of pinning the guard.

    The old 600s default evicted the entry mid-think, so a late tap landed on a dead entry and the agent
    hung on ``running: clarify`` (#32762).
    """
    try:
        from hermes_cli.config import load_config
        return resolve_clarify_timeout(load_config() or {})
    except Exception:
        return 3600


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def get_notify(session_key: str) -> Optional[Callable[[_ClarifyEntry], None]]:
    with _lock:
        return _notify_cbs.get(session_key)

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
# ---- END PLUGIN-COMPAT ----
