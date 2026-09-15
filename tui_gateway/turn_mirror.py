"""Mirror locally-served turns of gateway-owned sessions back to their platform chat.

A session whose state.db ``source`` resolves to a gateway platform (``telegram``, ``discord``,
...) belongs to the messaging gateway.  When such a session is continued from a local surface —
the Desktop chat, the TUI, the dashboard — the exchange exists only on that surface; the
platform's own chat never sees it, so the user's record on the platform is incomplete.

With ``display.platforms.<platform>.mirror_local_turns`` enabled, each completed local turn is
forwarded to the session's own chat (the ``chat_id``/``thread_id`` the gateway stored on the
session row) through the platform plugin's standalone sender — the same ``hermes send`` /
cron-delivery helper chain, so no live gateway connection is required:

    📲

    > the user's message, quoted line by line

    the assistant's reply

Only complete exchanges are mirrored; nothing is sent for synthetic turns (compaction markers,
auto-continue, background notices), silent responses, or empty replies.  Delivery happens on a
detached daemon thread and every failure is contained to a log line — a mirror must never delay,
break, or alter the turn it mirrors.

Config (resolved like every display setting: ``display.platforms.<platform>.<key>`` wins, the
top-level ``display.<key>`` is the fallback; defaults in parentheses):

    mirror_local_turns: false          # master switch (false)
    mirror_local_turns_silent: true    # ``disable_notification`` on the mirrored sends (true)
    mirror_local_turns_4096_split: false
                                       # false: keep one message where possible; a longer
                                       #   payload paginates past the platform cap
                                       # true: split into standalone <=4096 messages
    mirror_local_turns_label: "📲"     # first-line marker; "" hides it
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The silent-response filter token: a reply that is only this token means "nothing was said".
_SILENT_REPLY = "[SILENT]"


@dataclass(frozen=True)
class MirrorPlan:
    """One resolved mirror delivery: where, what, and how."""

    platform: str
    chat_id: str
    message: str
    thread_id: Optional[str] = None
    silent: bool = True
    chunk_indicators: bool = True  # False in the "split into standalone messages" mode


def quote_block(text: str) -> str:
    """Blockquote of *text*: every line gets a ``> `` prefix; blank lines keep the quote open."""
    lines = str(text).rstrip().splitlines()
    return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)


def compose_mirror_message(user_text: str, reply_text: str, label: str = "") -> str:
    """The mirrored copy: optional label line, the quoted user message, then the reply."""
    parts = []
    if str(label).strip():
        parts.append(str(label).strip())
    parts.append(quote_block(user_text))
    parts.append(str(reply_text).rstrip())
    return "\n\n".join(parts)


def mirror_settings(cfg: dict, platform: str) -> Optional[dict]:
    """``{silent, chunk_indicators, label}`` when mirroring is enabled for *platform*, else None."""
    from gateway.display_config import resolve_display_setting

    if not resolve_display_setting(cfg, platform, "mirror_local_turns", False):
        return None
    return {
        "silent": bool(resolve_display_setting(cfg, platform, "mirror_local_turns_silent", True)),
        "chunk_indicators": not bool(
            resolve_display_setting(cfg, platform, "mirror_local_turns_4096_split", False)),
        "label": str(resolve_display_setting(cfg, platform, "mirror_local_turns_label", "📲") or ""),
    }


def build_mirror_plan(*, platform: str, chat_id: str, thread_id: Optional[str],
                      user_text: Any, reply_text: Any, display_kind: Optional[str],
                      rid: str, cfg: dict) -> Optional[MirrorPlan]:
    """The one decision point; returns what to send, or None. No I/O — pure and testable."""
    if display_kind or str(rid).startswith("__"):
        return None  # synthetic turn: compaction marker, auto-continue, relay, ...
    if not isinstance(user_text, str) or not user_text.strip():
        return None
    reply = reply_text if isinstance(reply_text, str) else ""
    if not reply.strip() or reply.strip() == _SILENT_REPLY:
        return None
    if not platform or not chat_id:
        return None
    settings = mirror_settings(cfg, platform)
    if settings is None:
        return None
    return MirrorPlan(
        platform=platform,
        chat_id=str(chat_id),
        thread_id=thread_id or None,
        message=compose_mirror_message(user_text, reply, settings["label"]),
        silent=settings["silent"],
        chunk_indicators=settings["chunk_indicators"],
    )


def resolve_mirror_target(session: dict, agent: Any) -> tuple[str, str, Optional[str]]:
    """``(platform, chat_id, thread_id)`` for the session's own chat, or ``("", "", None)``.

    Reads the gateway-stored ``source``/``chat_id``/``thread_id`` off the session row — the same
    facts the gateway itself would deliver to.  Fail-closed: any missing piece skips the mirror
    rather than guessing a target.
    """
    session_id = str(getattr(agent, "session_id", "") or "").strip() or str(session.get("session_key") or "").strip()
    if not session_id:
        return "", "", None
    row: dict = {}
    try:
        from tui_gateway.session_workdir import _session_db

        with _session_db(session) as db:
            getter = getattr(db, "get_session", None) if db is not None else None
            fetched = getter(session_id) if callable(getter) else None
            row = fetched if isinstance(fetched, dict) else {}
    except Exception:
        logger.debug("mirror target lookup failed", exc_info=True)
        return "", "", None
    platform = str(row.get("source") or "").strip().lower()
    chat_id = str(row.get("chat_id") or "").strip()
    thread_id = str(row.get("thread_id") or "").strip() or None
    return platform, chat_id, thread_id


def mirror_turn(session: dict, agent: Any, user_text: Any, reply_text: Any, *,
                display_kind: Optional[str] = None, rid: str = "") -> None:
    """Entry point: decide, then forward one completed local turn on a detached thread.

    Never raises and never blocks the caller (the turn's own worker thread).
    """
    try:
        platform, chat_id, thread_id = resolve_mirror_target(session, agent)
        if not platform:
            return
        from tui_gateway.session_lifecycle import _is_gateway_owned_source

        if not _is_gateway_owned_source(platform):
            return
        from tui_gateway.server import _load_cfg

        plan = build_mirror_plan(
            platform=platform, chat_id=chat_id, thread_id=thread_id,
            user_text=user_text, reply_text=reply_text,
            display_kind=display_kind, rid=rid, cfg=_load_cfg(),
        )
        if plan is None:
            return
    except Exception:
        logger.debug("mirror decision failed", exc_info=True)
        return
    _start_delivery(plan)


def _start_delivery(plan: MirrorPlan) -> None:
    """Hand the plan to a detached daemon thread (the sender is network-bound)."""
    threading.Thread(target=_deliver, args=(plan,), name="turn-mirror", daemon=True).start()


def _deliver(plan: MirrorPlan) -> None:
    """Send the mirror through the platform plugin's standalone sender; contain all failures."""
    try:
        from gateway.config import PlatformConfig
        from tools.send_message_senders import _registry_standalone_send

        result = asyncio.run(_registry_standalone_send(
            plan.platform, PlatformConfig(enabled=True), plan.chat_id, plan.message,
            thread_id=plan.thread_id, silent=plan.silent, chunk_indicators=plan.chunk_indicators,
        ))
        if isinstance(result, dict) and result.get("error"):
            logger.warning("turn mirror send failed (%s:%s): %s",
                           plan.platform, plan.chat_id, result.get("error"))
        else:
            logger.debug("turn mirror sent to %s:%s", plan.platform, plan.chat_id)
    except Exception:
        logger.warning("turn mirror send failed (%s:%s)", plan.platform, plan.chat_id, exc_info=True)
