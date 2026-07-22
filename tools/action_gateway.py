"""Gateway-side action-menu registry for inline action buttons on messages.

Mirrors :mod:`tools.clarify_gateway`, but for **fire-and-forget action menus**
rather than a blocking clarify prompt. A message (e.g. a scheduled digest) declares
one or more follow-up actions; the Telegram adapter renders them as inline buttons; a
later tap resolves the chosen action and hands it to a registered dispatch callback.

Design notes
------------
* **No platform imports.** This module is pure data + a small callback registry, so it
  is unit-testable without a live Telegram client (see
  ``tests/gateway/test_telegram_action_buttons.py``).
* **Callback budget.** Telegram caps ``callback_data`` at 64 bytes. We never inline the
  action text into the callback; instead we key a short ``set_id`` + integer action id
  (``ca:<set_id>:<action_id>``) and keep the human label server-side here — the same
  short-id indirection the clarify/approval flows already use.
* **Label de-duplication.** Two distinct actions must never render as identical buttons
  (the bug that motivated #52252): if labels collide we disambiguate them.

The remaining integration seam — *where* outbound scheduled/agent messages call
``send_action_menu`` and what a tap dispatches back into the agent — is intentionally
left to the gateway and tracked in #52252; this module provides the mechanism.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

CALLBACK_PREFIX = "ca"
CALLBACK_MAX_BYTES = 64  # Telegram callback_data hard cap
LABEL_MAX_CHARS = 40     # button labels truncate for display (mirrors model picker)


@dataclass
class Action:
    """One tappable follow-up action attached to a message."""

    id: int
    label: str            # human description shown on the button
    op: str = "do"        # do | details | skip | retry | <custom>
    payload: Optional[str] = None  # opaque; the dispatch handler interprets it


@dataclass
class _ActionSet:
    set_id: str
    actions: List[Action]
    session_key: Optional[str] = None
    resolved: bool = False


_lock = threading.RLock()
# set_id → _ActionSet (primary lookup for button callbacks)
_sets: Dict[str, _ActionSet] = {}
# session_key (or "*") → dispatch callback registered by the gateway (the seam)
_dispatch_cbs: Dict[str, Callable[[str, Action], None]] = {}


def _new_set_id() -> str:
    return uuid.uuid4().hex[:10]


# =========================================================================
# Registration
# =========================================================================

def register(
    actions: Iterable,
    session_key: Optional[str] = None,
    set_id: Optional[str] = None,
) -> _ActionSet:
    """Register an action set and return it.

    ``actions`` is an iterable of :class:`Action` or dicts. Dict keys accepted:
    ``id``, ``label`` (falls back to ``object`` then ``verb``), ``op``, ``payload``.
    Visible labels are de-duplicated so two distinct actions never render as the
    same button.
    """
    norm: List[Action] = []
    seen: Dict[str, int] = {}
    for i, a in enumerate(actions):
        if isinstance(a, dict):
            label = str(a.get("label") or a.get("object") or a.get("verb") or f"action {i + 1}").strip()
            a = Action(
                id=int(a.get("id", i + 1)),
                label=label or f"action {i + 1}",
                op=str(a.get("op", "do")),
                payload=a.get("payload"),
            )
        count = seen.get(a.label, 0)
        if count:
            a = Action(id=a.id, label=f"{a.label} ({count + 1})", op=a.op, payload=a.payload)
        seen[a.label] = count + 1
        norm.append(a)

    sid = set_id or _new_set_id()
    entry = _ActionSet(set_id=sid, actions=norm, session_key=session_key)
    with _lock:
        _sets[sid] = entry
    return entry


# =========================================================================
# Rendering helpers (adapter side)
# =========================================================================

def build_keyboard_rows(set_id: str, include_skip: bool = True) -> List[List[Tuple[str, str]]]:
    """Return rows of ``(button_label, callback_data)`` for the adapter to render.

    Each ``callback_data`` is ``ca:<set_id>:<action_id>`` and is guaranteed to fit
    inside Telegram's 64-byte cap. Labels are truncated for display only.
    """
    with _lock:
        entry = _sets.get(set_id)
    if not entry:
        return []
    rows: List[List[Tuple[str, str]]] = []
    for a in entry.actions:
        cb = f"{CALLBACK_PREFIX}:{set_id}:{a.id}"
        if len(cb.encode("utf-8")) > CALLBACK_MAX_BYTES:
            logger.warning("action callback_data over 64 bytes, skipping: %s", cb)
            continue
        label = a.label if len(a.label) <= LABEL_MAX_CHARS else a.label[: LABEL_MAX_CHARS - 1] + "…"
        rows.append([(label, cb)])
    if include_skip and rows:
        rows.append([("Skip", f"{CALLBACK_PREFIX}:{set_id}:skip")])
    return rows


def parse_callback(data: str) -> Optional[Tuple[str, str]]:
    """Parse ``ca:<set_id>:<token>`` → ``(set_id, token)``; ``None`` if not ours."""
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        return None
    return parts[1], parts[2]


# =========================================================================
# Resolution (callback side)
# =========================================================================

def resolve(set_id: str, token: str) -> Optional[Action]:
    """Resolve a tapped ``token`` to its :class:`Action`.

    Returns the chosen Action, or ``None`` for ``skip`` / unknown / already-resolved.
    Marks the set resolved so a stale button cannot fire twice.
    """
    with _lock:
        entry = _sets.get(set_id)
        if not entry or entry.resolved:
            return None
        if token == "skip":
            entry.resolved = True
            return None
        try:
            aid = int(token)
        except (TypeError, ValueError):
            return None
        for a in entry.actions:
            if a.id == aid:
                entry.resolved = True
                return a
    return None


def clear(set_id: str) -> None:
    with _lock:
        _sets.pop(set_id, None)


# =========================================================================
# Dispatch seam (gateway → agent). Tracked in #52252.
# =========================================================================

def register_dispatch(key: str, cb: Callable[[str, Action], None]) -> None:
    """Register a callback invoked when an action is tapped. ``key`` is a session_key
    or ``"*"`` for a global default."""
    with _lock:
        _dispatch_cbs[key] = cb


def get_dispatch(session_key: Optional[str]) -> Optional[Callable[[str, Action], None]]:
    with _lock:
        return _dispatch_cbs.get(session_key or "") or _dispatch_cbs.get("*")
