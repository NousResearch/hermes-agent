"""Chat-origin scope for ``session_search``.

A gateway turn belongs to one conversation (a DM, a group, a channel or thread). Without a scope,
``session_search`` recalled from every conversation in the profile and returned hits with no
visible provenance, so a hit from another chat read like this chat's history, and a group chat
could be shown a different chat's transcript.

Rules (``scope`` argument):

* default: ``"chat"`` when the turn has a chat origin on a human messaging surface (Signal, Telegram,
  Discord, ...), else ``"all"`` (CLI, TUI/desktop, API server, cron, kanban, no origin): those bind a
  per-conversation session key too, but cross-session recall is what their user asks for;
* ``"chat"`` filters every search route in SQL (FTS/LIKE, title lookup, browse) by the origin's
  ``session_key`` (else its ``chat_id``) plus its platform, and fails closed when no origin is known; legacy rows with no
  ``session_key``/``chat_id`` are outside every chat scope (reachable with ``scope="all"``);
* ``"all"`` is explicit cross-conversation recall; every result then carries ``origin`` and,
  when the turn has an origin, ``same_origin`` (same-origin results rank first).

The origin comes from the current session's row, falling back to the turn's session context
(``HERMES_SESSION_KEY`` / ``HERMES_SESSION_CHAT_ID`` ...). ``profile=`` recall never applies the
current chat's origin to another profile's database.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCOPES = ("chat", "all")

CROSS_CHAT_NOTICE = ("Some results are from a different conversation (same_origin=false). Treat them as "
                     "cross-chat history; in a group chat, ask before relaying them here.")
CROSS_CHAT_WARNING = ("This session belongs to a different conversation from the active chat. Do not present "
                      "it as this chat's history without confirmation.")
NO_ORIGIN_MESSAGE = ("No active chat origin is available for scope='chat'. Pass scope='all' to search "
                     "other conversations.")
_ENV_FALLBACK = (("session_key", "HERMES_SESSION_KEY"), ("chat_id", "HERMES_SESSION_CHAT_ID"),
                 ("chat_type", "HERMES_SESSION_CHAT_TYPE"), ("display_name", "HERMES_SESSION_CHAT_NAME"),
                 ("source", "HERMES_SESSION_PLATFORM"))


def current_origin(db, current_session_id: Optional[str]) -> Optional[dict[str, Any]]:
    """The active turn's chat origin (session row first, session context second), or None."""
    meta: dict[str, Any] = {}
    if current_session_id:
        try:
            meta = db.get_session(current_session_id) or {}
        except Exception:
            logger.debug("session_search: origin lookup failed for %s", current_session_id, exc_info=True)
    origin = {key: meta.get(key) or None for key, _env in _ENV_FALLBACK}
    if not (origin["session_key"] or origin["chat_id"]):
        try:
            from gateway.session_context import NON_MESSAGING_SESSION_SURFACES, get_session_env
            # The chat comes from the context, so its platform must too: a delegated child's
            # row says source="subagent", which no chat row carries.
            if origin["source"] not in NON_MESSAGING_SESSION_SURFACES | {"cron"}:
                origin["source"] = None
            for key, env in _ENV_FALLBACK:
                origin[key] = origin[key] or get_session_env(env, "") or None
        except Exception:
            logger.debug("session_search: session-context origin lookup failed", exc_info=True)
    return origin if (origin["session_key"] or origin["chat_id"]) else None


def _on_messaging_surface(origin: dict[str, Any]) -> bool:
    """The origin's platform when known, else the turn's session context, is a human messaging channel."""
    from gateway.session_context import NON_MESSAGING_SESSION_SURFACES, session_is_messaging_surface
    source = str(origin.get("source") or "").strip().lower()
    if source:
        return source not in NON_MESSAGING_SESSION_SURFACES and source != "cron"
    return session_is_messaging_surface()


def resolve_scope(scope: Any, origin: Optional[dict[str, Any]]) -> str:
    """Explicit ``chat``/``all`` wins; else ``chat`` only for an origin on a messaging surface."""
    norm = scope.strip().lower() if isinstance(scope, str) else ""
    return norm if norm in SCOPES else ("chat" if origin and _on_messaging_surface(origin) else "all")


def origin_filters(scope: str, origin: Optional[dict[str, Any]]) -> dict[str, Any]:
    """SQL filter kwargs for title lookup / browse: the session key (else chat id) plus the source."""
    if scope != "chat" or not origin:
        return {}
    key = {"session_key": origin["session_key"]} if origin.get("session_key") else {"chat_id": origin["chat_id"]}
    return {**key, **({"source": origin["source"]} if origin.get("source") else {})}


def scope_filters(scope: str, origin: Optional[dict[str, Any]]) -> dict[str, Any]:
    """:func:`origin_filters` shaped for ``search_messages`` (the source becomes ``source_filter``)."""
    filters = origin_filters(scope, origin)
    if "source" in filters:
        filters["source_filter"] = [filters.pop("source")]
    return filters


def same_origin(meta: dict[str, Any], origin: Optional[dict[str, Any]]) -> Optional[bool]:
    """True/False against the active origin; None when the turn has none (nothing to compare)."""
    if not origin:
        return None
    if origin.get("source") and meta.get("source") != origin["source"]:
        return False
    if origin.get("session_key"):
        return bool(meta.get("session_key")) and meta.get("session_key") == origin["session_key"]
    return bool(meta.get("chat_id")) and meta.get("chat_id") == origin["chat_id"]


def origin_label(meta: dict[str, Any]) -> dict[str, Any]:
    """Compact provenance for one result: platform, chat type, chat name; no raw chat ids or session keys.
    In a DM the chat name is usually the other person's display name."""
    return {k: meta.get(k) or None for k in ("source", "chat_type", "display_name")}


def annotate(entry: dict[str, Any], meta: dict[str, Any], origin: Optional[dict[str, Any]]) -> Optional[bool]:
    """Add ``origin`` (+ ``same_origin`` when comparable) to *entry*; returns the same-origin verdict."""
    entry["origin"] = origin_label(meta)
    verdict = same_origin(meta, origin)
    if verdict is not None:
        entry["same_origin"] = verdict
    return verdict


def recall_scope(scope: str, origin: Optional[dict[str, Any]], **counts: Optional[int]) -> dict[str, Any]:
    """``recall_scope`` payload block: the level, the active origin's label, optional match counts."""
    return {"scope_level": scope, **{k: v for k, v in origin_label(origin or {}).items() if v},
            **{k: v for k, v in counts.items() if v is not None}}


def payload_fields(results: list[dict[str, Any]], scope: str, origin: Optional[dict[str, Any]],
                   **counts: Optional[int]) -> dict[str, Any]:
    """``scope``/``recall_scope`` (+ the cross-chat ``notice``) for a discover/browse payload."""
    fields: dict[str, Any] = {"scope": scope, "recall_scope": recall_scope(scope, origin, **counts)}
    if any(e.get("same_origin") is False for e in results):
        fields["notice"] = CROSS_CHAT_NOTICE
    return fields


def warn_if_foreign(response: dict[str, Any], meta: dict[str, Any], origin: Optional[dict[str, Any]]) -> None:
    """Read/scroll by explicit id stay allowed, but a foreign session is labelled as such."""
    verdict = same_origin(meta, origin)
    if verdict is not None:
        response["same_origin"] = verdict
        if verdict is False:
            response["cross_context_warning"] = CROSS_CHAT_WARNING
