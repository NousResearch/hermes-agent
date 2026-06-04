"""Gateway-side reaction-menu primitive (non-blocking, consumer-agnostic).

Unlike the ``clarify`` and ``approval`` primitives, ``present_menu`` does **not**
block the agent thread.  The tool presents a menu — a message with one seeded
reaction per option — and ends the turn.  When the user later taps an option,
the platform adapter forges a synthetic ``[menu-choice]`` user turn and dispatches
it through its own message handler; the FIFO / promote machinery in the runner
queues that turn behind any busy agent for free (see ``GatewayRunner._enqueue_fifo``).

So this module is pure bookkeeping — there is no ``threading.Event`` here:

  * a per-message registry of live menus, keyed by ``menu_id``;
  * option validation (1–5 options, unique emoji, ``silent`` rejected in v1);
  * a resolve/dedup primitive so a double-tap injects exactly one turn;
  * a per-session "newest menu" pointer used **only** by the text fallback, where
    a bare numeric reply can't disambiguate between several live menus;
  * optional SQLite persistence so unresolved menus survive a gateway restart.

Two layers (spec §2, deliberately consumer-agnostic):

  * The core registry is keyed by ``menu_id`` (one per message) and supports
    MANY live menus in a single session simultaneously.  Reaction and (future)
    button consumers drive it directly.
  * The ``newest``-per-session pointer is a thin convenience for the text
    fallback only.  It does not constrain the core.

State is module-level (same shape as ``tools.approval`` / ``tools.clarify_gateway``)
so platform adapters can resolve a menu without holding a back-reference to the
``GatewayRunner``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# Synthetic-turn marker (mirrors the /goal continuation marker in run.py)
# =========================================================================
# A tapped menu injects a normal user-role turn whose body is prefixed with
# this marker.  The runner recognises the prefix to treat it like a goal
# continuation for pause/clear purposes; the model reads the marker as a
# provenance signal for the chosen path.

MENU_CHOICE_MARKER = "[menu-choice]"


def build_menu_choice_body(payload: str) -> str:
    """Build the synthetic-turn body for a chosen option payload."""
    return f"{MENU_CHOICE_MARKER}\n{payload}"


def is_menu_choice_text(text_or_event: Any) -> bool:
    """Return True for a synthetic ``[menu-choice]`` turn (text or event)."""
    text = getattr(text_or_event, "text", text_or_event) or ""
    return str(text).startswith(MENU_CHOICE_MARKER)


def parse_menu_choice_payload(text: str) -> str:
    """Extract the payload from a ``[menu-choice]\\n{payload}`` body."""
    raw = str(text or "")
    if not raw.startswith(MENU_CHOICE_MARKER):
        return raw
    return raw[len(MENU_CHOICE_MARKER):].lstrip("\n")


# =========================================================================
# Option validation
# =========================================================================

MIN_OPTIONS = 1
MAX_OPTIONS = 5

# Reaction reserved by the menu choreography itself (reload / regenerate).
# An option may NOT claim it.
RELOAD_EMOJI = "♻️"  # ♻️


class MenuValidationError(ValueError):
    """Raised when an option list fails validation."""


def validate_options(options: Any) -> List[Dict[str, Any]]:
    """Validate and normalise a menu's option list.

    Each option is a dict with:
      * ``emoji``    (str, required) — the reaction key / number anchor.
      * ``label``    (str, required) — human-readable choice text.
      * ``payload``  (str, required) — injected as the synthetic turn body.
      * ``terminal`` (bool, optional, default False) — when True the chosen
        path ends the menu lifecycle: no ``♻️`` reload reaction is seeded.

    Rules: 1–5 options; ``emoji``/``label``/``payload`` non-empty; emoji unique
    within the menu and never the reserved ``♻️``; ``silent: true`` is reserved
    and rejected in v1 (spec §8).

    Returns the normalised list.  Raises :class:`MenuValidationError` otherwise.
    """
    if not isinstance(options, list):
        raise MenuValidationError("options must be a list")
    if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
        raise MenuValidationError(
            f"a menu needs {MIN_OPTIONS}–{MAX_OPTIONS} options, got {len(options)}"
        )

    normalized: List[Dict[str, Any]] = []
    seen_emoji: set[str] = set()
    for idx, opt in enumerate(options):
        if not isinstance(opt, dict):
            raise MenuValidationError(f"option {idx} must be an object")
        if opt.get("silent"):
            raise MenuValidationError(
                "silent menus are reserved and not supported in this version"
            )
        emoji = str(opt.get("emoji", "")).strip()
        label = str(opt.get("label", "")).strip()
        payload = str(opt.get("payload", "")).strip()
        if not emoji:
            raise MenuValidationError(f"option {idx} is missing 'emoji'")
        if not label:
            raise MenuValidationError(f"option {idx} is missing 'label'")
        if not payload:
            raise MenuValidationError(f"option {idx} is missing 'payload'")
        if emoji == RELOAD_EMOJI:
            raise MenuValidationError(
                f"option {idx} uses the reserved reload reaction {RELOAD_EMOJI}"
            )
        if emoji in seen_emoji:
            raise MenuValidationError(f"duplicate emoji {emoji!r} in menu")
        seen_emoji.add(emoji)
        normalized.append({
            "emoji": emoji,
            "label": label,
            "payload": payload,
            "terminal": bool(opt.get("terminal", False)),
        })
    return normalized


# =========================================================================
# Registry entry
# =========================================================================

@dataclass
class _MenuEntry:
    """One live menu, keyed by ``menu_id`` (per message, consumer-agnostic)."""

    menu_id: str
    session_key: str
    platform: str
    chat_id: str
    message_id: str
    prompt: str
    options: List[Dict[str, Any]]
    context_id: Optional[str] = None
    # Serialised SessionSource (see gateway.session.SessionSource.to_dict) so
    # the adapter can forge a routed synthetic turn — even after a restart.
    source: Optional[Dict[str, Any]] = None
    resolved: bool = False
    created_ts: float = field(default_factory=time.time)

    def option_for_emoji(self, emoji: str) -> Optional[Dict[str, Any]]:
        for opt in self.options:
            if opt["emoji"] == emoji:
                return opt
        return None

    def to_row(self) -> Tuple:
        return (
            self.menu_id,
            self.session_key,
            self.platform,
            self.chat_id,
            self.message_id,
            self.prompt,
            json.dumps(self.options, ensure_ascii=False),
            self.context_id,
            json.dumps(self.source, ensure_ascii=False) if self.source else None,
            1 if self.resolved else 0,
            self.created_ts,
        )

    @classmethod
    def from_row(cls, row: Tuple) -> "_MenuEntry":
        return cls(
            menu_id=row[0],
            session_key=row[1],
            platform=row[2],
            chat_id=row[3],
            message_id=row[4],
            prompt=row[5],
            options=json.loads(row[6]) if row[6] else [],
            context_id=row[7],
            source=json.loads(row[8]) if row[8] else None,
            resolved=bool(row[9]),
            created_ts=row[10],
        )


# =========================================================================
# Module-level state
# =========================================================================

_lock = threading.RLock()
_menus: Dict[str, _MenuEntry] = {}                  # menu_id → entry
_session_index: Dict[str, List[str]] = {}           # session_key → [menu_id, …]
_session_newest: Dict[str, str] = {}                # session_key → newest menu_id (text fallback)
_notify_cbs: Dict[str, Callable[[_MenuEntry], None]] = {}  # session_key → send callback

# GC thresholds (spec §6).
_UNRESOLVED_TTL_SECONDS = 30 * 24 * 3600   # 30 days
_RESOLVED_TTL_SECONDS = 7 * 24 * 3600      # 7 days (only relevant if kept)


# =========================================================================
# Persistence (SQLite)
# =========================================================================

_db_path_override: Optional[Path] = None


def set_db_path(path: Optional[str | Path]) -> None:
    """Override the persistence DB path (used by tests).  None resets to default."""
    global _db_path_override
    _db_path_override = Path(path) if path is not None else None


def _db_path() -> Path:
    if _db_path_override is not None:
        return _db_path_override
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "reaction_menus.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reaction_menus (
            menu_id     TEXT PRIMARY KEY,
            session_key TEXT,
            platform    TEXT,
            chat_id     TEXT,
            message_id  TEXT,
            prompt      TEXT,
            options     TEXT,
            context_id  TEXT,
            source      TEXT,
            resolved    INTEGER DEFAULT 0,
            created_ts  REAL
        )
        """
    )
    return conn


def _persist(entry: _MenuEntry) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reaction_menus "
                "(menu_id, session_key, platform, chat_id, message_id, prompt, "
                " options, context_id, source, resolved, created_ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                entry.to_row(),
            )
    except Exception as exc:  # persistence is best-effort
        logger.debug("reaction_menu: persist failed for %s: %s", entry.menu_id, exc)


def _delete_persisted(menu_id: str) -> None:
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM reaction_menus WHERE menu_id = ?", (menu_id,))
    except Exception as exc:
        logger.debug("reaction_menu: delete failed for %s: %s", menu_id, exc)


def load_persisted(platform: Optional[str] = None) -> List[_MenuEntry]:
    """Load unresolved persisted menus into the in-memory registry.

    Called by an adapter on startup so it can rebuild its platform-specific
    tracking (e.g. ``_menu_prompts_by_event``).  Returns the restored entries,
    optionally filtered to one ``platform``.  Expired rows are GC'd here.
    """
    restored: List[_MenuEntry] = []
    now = time.time()
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT menu_id, session_key, platform, chat_id, message_id, "
                "prompt, options, context_id, source, resolved, created_ts "
                "FROM reaction_menus WHERE resolved = 0"
            ).fetchall()
    except Exception as exc:
        logger.debug("reaction_menu: load failed: %s", exc)
        return restored

    for row in rows:
        try:
            entry = _MenuEntry.from_row(row)
        except Exception:
            continue
        if now - entry.created_ts > _UNRESOLVED_TTL_SECONDS:
            _delete_persisted(entry.menu_id)
            continue
        if platform is not None and entry.platform != platform:
            continue
        with _lock:
            _menus[entry.menu_id] = entry
            self_ids = _session_index.setdefault(entry.session_key, [])
            if entry.menu_id not in self_ids:
                self_ids.append(entry.menu_id)
            _session_newest[entry.session_key] = entry.menu_id
        restored.append(entry)
    return restored


# =========================================================================
# Registry API
# =========================================================================

def register(
    menu_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    message_id: str,
    prompt: str,
    options: List[Dict[str, Any]],
    context_id: Optional[str] = None,
    source: Optional[Dict[str, Any]] = None,
    persist: bool = True,
) -> _MenuEntry:
    """Register a live menu and return the entry.

    ``options`` must already be normalised (call :func:`validate_options`).
    Adds the menu to the per-session index and marks it the newest live menu
    for that session (text-fallback disambiguation).
    """
    entry = _MenuEntry(
        menu_id=menu_id,
        session_key=session_key,
        platform=platform,
        chat_id=chat_id,
        message_id=message_id,
        prompt=prompt,
        options=list(options),
        context_id=context_id,
        source=source,
    )
    with _lock:
        _menus[menu_id] = entry
        _session_index.setdefault(session_key, []).append(menu_id)
        _session_newest[session_key] = menu_id
    if persist:
        _persist(entry)
    return entry


def get(menu_id: str) -> Optional[_MenuEntry]:
    with _lock:
        return _menus.get(menu_id)


def get_by_message(message_id: str) -> Optional[_MenuEntry]:
    """Return the live menu attached to a platform message, or None."""
    if not message_id:
        return None
    with _lock:
        for entry in _menus.values():
            if entry.message_id == message_id and not entry.resolved:
                return entry
    return None


def mark_resolved(menu_id: str) -> bool:
    """Flip a menu to resolved.  Returns True only on the first transition.

    The dedup primitive: a double-tap calls this twice but only the first
    call returns True, so exactly one synthetic turn is injected.  The
    resolved entry is dropped from the registry and from persistence.
    """
    with _lock:
        entry = _menus.get(menu_id)
        if entry is None or entry.resolved:
            return False
        entry.resolved = True
        _drop_locked(menu_id)
    _delete_persisted(menu_id)
    return True


def remove(menu_id: str) -> None:
    """Remove a menu from the registry and persistence (no resolve semantics)."""
    with _lock:
        _drop_locked(menu_id)
    _delete_persisted(menu_id)


def _drop_locked(menu_id: str) -> None:
    """Remove a menu from the in-memory indices.  Caller holds ``_lock``."""
    entry = _menus.pop(menu_id, None)
    if entry is None:
        return
    ids = _session_index.get(entry.session_key)
    if ids and menu_id in ids:
        ids.remove(menu_id)
        if not ids:
            _session_index.pop(entry.session_key, None)
    if _session_newest.get(entry.session_key) == menu_id:
        # Promote the next-most-recent surviving menu, if any.
        remaining = _session_index.get(entry.session_key) or []
        if remaining:
            _session_newest[entry.session_key] = remaining[-1]
        else:
            _session_newest.pop(entry.session_key, None)


def get_newest_for_session(session_key: str) -> Optional[_MenuEntry]:
    """Return the newest live menu for a session (text-fallback resolution)."""
    with _lock:
        menu_id = _session_newest.get(session_key)
        if not menu_id:
            return None
        entry = _menus.get(menu_id)
        return entry if entry and not entry.resolved else None


def list_for_session(session_key: str) -> List[_MenuEntry]:
    with _lock:
        return [
            _menus[mid]
            for mid in _session_index.get(session_key, [])
            if mid in _menus
        ]


def clear_session(session_key: str) -> int:
    """Drop every live menu for a session (e.g. ``/new``, eviction).  Returns count."""
    with _lock:
        ids = list(_session_index.get(session_key, []) or [])
        for mid in ids:
            _drop_locked(mid)
    for mid in ids:
        _delete_persisted(mid)
    return len(ids)


# =========================================================================
# Per-session notify hook (gateway → adapter send bridge)
# =========================================================================
# Mirrors tools.approval / tools.clarify_gateway: the gateway runner registers
# a per-session callback that sends the menu to the user.  ``present_menu``
# invokes it synchronously on the agent thread; the callback schedules the
# adapter's ``send_reaction_menu`` on the event loop and returns once delivery
# is confirmed (non-blocking — it does NOT wait for the user to tap).

def register_notify(session_key: str, cb: Callable[[_MenuEntry], None]) -> None:
    with _lock:
        _notify_cbs[session_key] = cb


def unregister_notify(session_key: str) -> None:
    with _lock:
        _notify_cbs.pop(session_key, None)


def get_notify(session_key: str) -> Optional[Callable[[_MenuEntry], None]]:
    with _lock:
        return _notify_cbs.get(session_key)


# =========================================================================
# Test / lifecycle helpers
# =========================================================================

def reset_state() -> None:
    """Drop all in-memory state.  For tests and full gateway resets."""
    with _lock:
        _menus.clear()
        _session_index.clear()
        _session_newest.clear()
        _notify_cbs.clear()
