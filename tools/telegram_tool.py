"""Telegram platform tools — native sticker send + collection management.

Two service-gated tools (Footprint Ladder rung 3):

- ``tg_send_sticker`` — send a *native* Telegram sticker (Bot API ``file_id``
  semantics) from the persistent collection in
  ``plugins/platforms/telegram/sticker_collection.py``. Exposed only inside
  Telegram gateway sessions, and delivery is scoped to the active session's
  chat (derived from session context, never a model-supplied target) — the
  same no-arbitrary-outbound rule that keeps ``send_message`` out of the
  agent-callable registry.
- ``tg_manage_stickers`` — curate that collection (list / update / remove /
  add_set). Exposed in Telegram gateway sessions or when
  ``TELEGRAM_BOT_TOKEN`` is configured (``add_set`` works from the CLI too).
  Management surface only; NOT a global Telegram sticker search.

Mirrors the registration style of ``tools/yuanbao_tools.py`` and the
one-shot Bot + proxy pattern of ``tools/send_message_tool.py::_send_telegram``.
``telegram.Bot`` (python-telegram-bot) is imported lazily inside functions so
minimal installs without PTB can still load this module and evaluate
``check_fn`` without crashing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from agent.redact import redact_sensitive_text
from tools.registry import registry, tool_result

logger = logging.getLogger(__name__)

_TOOLSET = "telegram"


def _sticker_collection():
    """Lazy access to the sticker-collection store.

    ``plugins.platforms.telegram``'s package ``__init__`` imports the adapter,
    which pulls in python-telegram-bot — an optional dependency and a heavy
    import. Importing it here at module top would make every CLI startup (and
    every PTB-less install) pay that cost at tool-discovery time, so we import
    only when a tool actually runs.
    """
    from plugins.platforms.telegram import sticker_collection

    return sticker_collection

# https://t.me/addstickers/<short_name> (optionally without the scheme).
_ADDSTICKERS_URL_RE = re.compile(r"(?:https?://)?t\.me/addstickers/([A-Za-z0-9_]+)")


# ---------------------------------------------------------------------------
# Availability gate + one-shot Bot construction
# ---------------------------------------------------------------------------


def _check_telegram_session() -> bool:
    """Availability check for tools that act on the ACTIVE Telegram session.

    True only inside a Telegram gateway session (session env is task-local).
    Used for tg_send_sticker, whose delivery target is derived from session
    context — outside a Telegram session the tool could not send anywhere, so
    it stays hidden rather than erroring at call time.
    """
    try:
        from gateway.session_context import get_session_env

        return get_session_env("HERMES_SESSION_PLATFORM", "") == "telegram"
    except Exception:
        return False


def _check_telegram() -> bool:
    """Toolset availability check.

    True when running inside a Telegram gateway session, or when a bot token
    is configured (CLI / cron one-shot usage). Must never crash when the
    gateway or python-telegram-bot is absent.
    """
    try:
        from gateway.session_context import get_session_env

        if get_session_env("HERMES_SESSION_PLATFORM", "") == "telegram":
            return True
    except Exception:
        pass
    try:
        from agent.secret_scope import get_secret

        if (get_secret("TELEGRAM_BOT_TOKEN", "") or "").strip():
            return True
    except Exception:
        pass
    return False


def _resolve_bot_token() -> str:
    """Resolve the Telegram bot token (profile-scoped secret → env fallback)."""
    try:
        from agent.secret_scope import get_secret

        token = get_secret("TELEGRAM_BOT_TOKEN", "")
    except Exception:
        token = ""
    return (token or "").strip()


def _build_bot(token: str):
    """Construct a one-shot ``telegram.Bot`` honouring the configured proxy.

    Mirrors ``tools/send_message_tool.py::_send_telegram``: ``TELEGRAM_PROXY``
    (resolved via ``resolve_proxy_url``) routes the Bot through
    ``HTTPXRequest``; a failure to attach the proxy falls back to a direct
    connection. Raises ImportError when python-telegram-bot is absent.
    """
    from telegram import Bot  # lazy: PTB may be absent in minimal installs

    try:
        from gateway.platforms.base import resolve_proxy_url

        proxy = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org"])
    except Exception:
        proxy = None

    if proxy:
        try:
            from telegram.request import HTTPXRequest

            logger.info("telegram_tool: one-shot Bot routed through proxy %s", proxy)
            return Bot(
                token=token,
                request=HTTPXRequest(proxy=proxy),
                get_updates_request=HTTPXRequest(proxy=proxy),
            )
        except Exception as exc:
            logger.warning(
                "telegram_tool: failed to attach Telegram proxy (%s); using direct connection",
                exc,
            )
    return Bot(token=token)


# ---------------------------------------------------------------------------
# Session-context defaults + Telegram id mapping
# ---------------------------------------------------------------------------


def _session_env(name: str) -> str:
    """Read a HERMES_SESSION_* value; "" when the gateway context is absent."""
    try:
        from gateway.session_context import get_session_env

        return (get_session_env(name, "") or "").strip()
    except Exception:
        return ""


def _map_thread_id_for_send(thread_id: str) -> Optional[int]:
    """Map a session thread id through the adapter's General-topic rule.

    Telegram forum supergroups address the General topic as thread id "1" on
    incoming updates, but Bot API sends reject ``message_thread_id=1`` with
    "Message thread not found" — the adapter's
    ``_message_thread_id_for_send`` maps "1" → None so the send lands in
    General. Falls back to the explicit mapping when the adapter import
    fails (e.g. python-telegram-bot missing in this venv).
    """
    if not thread_id:
        return None
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter

        return TelegramAdapter._message_thread_id_for_send(str(thread_id))
    except Exception:
        return None if str(thread_id) == "1" else int(thread_id)


def _redact_token(text: str, token: str) -> str:
    """Strip the bot token (and other secrets) from API error text."""
    if token:
        text = text.replace(token, "***")
    return redact_sensitive_text(text)


def _entry_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Selector-facing view of a collection entry (no per-bot file_id)."""
    return {
        "file_unique_id": entry["file_unique_id"],
        "emoji": entry["emoji"],
        "set_name": entry["set_name"],
        "kind": entry["kind"],
        "description": entry["description"],
    }


def _resolve_selector(file_unique_id: str, sticker: str) -> Optional[Dict[str, Any]]:
    """Locate a collection entry by exact file_unique_id, else a resolve query."""
    store = _sticker_collection()
    fuid = (file_unique_id or "").strip()
    if fuid:
        return store.resolve(fuid)
    query = (sticker or "").strip()
    if query:
        return store.resolve(query)
    return None


def _parse_set_short_name(raw: str) -> str:
    """Extract a sticker-pack short name, accepting t.me/addstickers URLs."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    match = _ADDSTICKERS_URL_RE.search(raw)
    if match:
        return match.group(1)
    return raw


# ---------------------------------------------------------------------------
# Tool 1: tg_send_sticker
# ---------------------------------------------------------------------------


async def tg_send_sticker(sticker: str, set_name: str = "") -> dict:
    """Send a native Telegram sticker from the collection to the current chat."""
    query = (sticker or "").strip()
    if not query:
        return {
            "success": False,
            "error": "sticker is required (emoji, set_name:emoji, file_unique_id, or file_id).",
        }

    entry = _sticker_collection().resolve(query, set_name=(set_name or "").strip())
    if entry is None:
        return {
            "success": False,
            "error": (
                f"Sticker {query!r} is not in your collection. You can only send stickers "
                "that are already in the collection — stickers users have sent you (recorded "
                "automatically), or packs you imported with tg_manage_stickers action='add_set'."
            ),
        }

    # Delivery is scoped to the ACTIVE Telegram session only. Mirroring the
    # send_message precedent (tools/send_message_tool.py), the model must not
    # get an arbitrary outbound target, so there is deliberately no chat_id
    # parameter — the destination is derived from session context.
    target = _session_env("HERMES_SESSION_CHAT_ID")
    if not target:
        return {
            "success": False,
            "error": "No active Telegram chat in this session — tg_send_sticker only works inside a Telegram conversation.",
        }

    token = _resolve_bot_token()
    if not token:
        return {
            "success": False,
            "error": "Telegram bot token is not configured (TELEGRAM_BOT_TOKEN).",
        }

    from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

    # Numeric ids pass through as ints; @username targets stay strings —
    # both are valid Bot API chat identifiers.
    normalized_chat_id = normalize_telegram_chat_id(target)
    thread_id = _map_thread_id_for_send(_session_env("HERMES_SESSION_THREAD_ID"))

    try:
        bot = _build_bot(token)
    except ImportError:
        return {
            "success": False,
            "error": "python-telegram-bot is not installed in this environment.",
        }

    send_kwargs: Dict[str, Any] = {"chat_id": normalized_chat_id, "sticker": entry["file_id"]}
    if thread_id is not None:
        send_kwargs["message_thread_id"] = thread_id

    try:
        message = await bot.send_sticker(**send_kwargs)
    except Exception as exc:
        logger.exception("[telegram_tool] tg_send_sticker error")
        return {"success": False, "error": _redact_token(str(exc), token)}

    return {
        "success": True,
        "chat_id": str(normalized_chat_id),
        "sticker": {
            "file_unique_id": entry["file_unique_id"],
            "emoji": entry["emoji"],
            "set_name": entry["set_name"],
        },
        "message_id": getattr(message, "message_id", None),
        "note": "Sticker delivered to the chat. If you have additional text to say, reply now; otherwise end your turn without generating text.",
    }


async def _handle_tg_send_sticker(args, **kw):
    return tool_result(await tg_send_sticker(
        sticker=args.get("sticker", ""),
        set_name=args.get("set_name", ""),
    ))


# ---------------------------------------------------------------------------
# Tool 2: tg_manage_stickers
# ---------------------------------------------------------------------------


async def tg_manage_stickers(
    action: str,
    sticker: str = "",
    file_unique_id: str = "",
    description: str = "",
    set_name: str = "",
    limit: int = 100,
) -> dict:
    """Curate the local Telegram sticker collection (list/update/remove/add_set)."""
    action = (action or "").strip().lower()
    store = _sticker_collection()

    if action == "list":
        set_filter = (set_name or "").strip()
        # The collection is capped at MAX_STICKERS entries, so one unbounded
        # listing gives both the page and the true total.
        entries = store.list_stickers(set_name=set_filter, limit=store.MAX_STICKERS)
        total = len(entries)
        try:
            cap = max(int(limit), 0)
        except (TypeError, ValueError):
            cap = 100
        shown = entries[:cap]
        result: Dict[str, Any] = {
            "success": True,
            "action": "list",
            "total": total,
            "returned": len(shown),
            "stickers": [_entry_summary(e) for e in shown],
        }
        if set_filter:
            result["set_name"] = set_filter
        if len(shown) < total:
            result["note"] = f"Showing {len(shown)} of {total} entries; raise limit to see more."
        return result

    if action in ("update", "remove"):
        if not (file_unique_id or "").strip() and not (sticker or "").strip():
            return {
                "success": False,
                "error": f"action={action!r} requires file_unique_id (preferred) or a sticker query (emoji / set_name:emoji).",
            }
        entry = _resolve_selector(file_unique_id, sticker)
        if entry is None:
            selector = (file_unique_id or "").strip() or (sticker or "").strip()
            return {
                "success": False,
                "error": f"No collection entry matches {selector!r}. Call tg_manage_stickers action='list' to see current file_unique_ids.",
            }
        if action == "update":
            if not store.update_description(entry["file_unique_id"], description or ""):
                return {
                    "success": False,
                    "error": f"Could not update {entry['file_unique_id']!r} (entry missing or corrupt).",
                }
            updated = dict(_entry_summary(entry))
            updated["description"] = description or ""
            return {"success": True, "action": "update", "updated": 1, "entry": updated}
        if not store.remove_sticker(entry["file_unique_id"]):
            return {
                "success": False,
                "error": f"Could not remove {entry['file_unique_id']!r} (entry already gone).",
            }
        return {"success": True, "action": "remove", "removed": 1, "entry": _entry_summary(entry)}

    if action == "add_set":
        name = _parse_set_short_name(set_name)
        if not name:
            return {
                "success": False,
                "error": "action='add_set' requires set_name — a pack short name or a https://t.me/addstickers/<name> link.",
            }
        token = _resolve_bot_token()
        if not token:
            return {
                "success": False,
                "error": "Telegram bot token is not configured (TELEGRAM_BOT_TOKEN).",
            }
        try:
            bot = _build_bot(token)
        except ImportError:
            return {
                "success": False,
                "error": "python-telegram-bot is not installed in this environment.",
            }
        try:
            summary = await store.refresh_from_sets(bot, [name])
        except Exception as exc:
            logger.exception("[telegram_tool] tg_manage_stickers add_set error")
            return {"success": False, "error": _redact_token(str(exc), token)}
        if summary.get("sets", 0) == 0:
            return {
                "success": False,
                "error": (
                    f"Could not import sticker set {name!r} — the Bot API fetch failed. "
                    "Check the short name (it must be exact) and the bot token."
                ),
                "summary": summary,
            }
        return {
            "success": True,
            "action": "add_set",
            "set_name": name,
            "sets": summary["sets"],
            "stickers": summary["stickers"],
            "new": summary["new"],
            "note": (
                f"Imported set {name!r}: {summary['stickers']} stickers "
                f"({summary['new']} new). They are now in your collection and "
                "sendable with tg_send_sticker."
            ),
        }

    return {
        "success": False,
        "error": f"Unknown action {action!r}. Valid actions: list, update, remove, add_set.",
    }


async def _handle_tg_manage_stickers(args, **kw):
    return tool_result(await tg_manage_stickers(
        action=args.get("action", ""),
        sticker=args.get("sticker", ""),
        file_unique_id=args.get("file_unique_id", ""),
        description=args.get("description", ""),
        set_name=args.get("set_name", ""),
        limit=args.get("limit", 100),
    ))


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


registry.register(
    name="tg_send_sticker",
    toolset=_TOOLSET,
    schema={
        "name": "tg_send_sticker",
        "description": (
            "Send a native Telegram sticker to the current Telegram chat. Stickers are "
            "resolved from your sticker collection — stickers users have sent you, plus "
            "packs you imported with tg_manage_stickers action='add_set'. The collection "
            "listing is injected into your context at session start; pick from it by emoji "
            "(optionally disambiguated with set_name) instead of guessing. "
            "CRITICAL: Whenever the user asks you to send a sticker, you MUST use this "
            "tool. DO NOT draw or generate a sticker image yourself and send it as a "
            "photo — that produces a fake 'sticker' picture instead of a real "
            "Telegram sticker and is the WRONG path. You can only send stickers that are "
            "already in your collection; if none fits, say so, or ask the user to send you "
            "one (or a https://t.me/addstickers/<name> link you can import)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sticker": {
                    "type": "string",
                    "description": (
                        "Which sticker to send: its emoji (e.g. '😀'), 'set_name:emoji' "
                        "for an exact pack match, or a file_unique_id / file_id from the "
                        "collection."
                    ),
                },
                "set_name": {
                    "type": "string",
                    "description": (
                        "Optional pack short name to disambiguate when several stickers "
                        "share the same emoji."
                    ),
                },
            },
            "required": ["sticker"],
        },
    },
    handler=_handle_tg_send_sticker,
    check_fn=_check_telegram_session,
    is_async=True,
    emoji="🎨",
)


registry.register(
    name="tg_manage_stickers",
    toolset=_TOOLSET,
    schema={
        "name": "tg_manage_stickers",
        "description": (
            "Manage your Telegram sticker collection (the stickers tg_send_sticker can "
            "send). Actions: 'update' annotates a sticker's description in your own words "
            "(e.g. 'use when the user is being sarcastic') so future picks are more "
            "accurate; 'remove' deletes an entry (also the cleanup path for stickers "
            "whose file_id went stale); 'add_set' imports a whole sticker pack by short "
            "name or https://t.me/addstickers/<name> link; 'list' shows the local "
            "collection. "
            "IMPORTANT about 'list': the collection listing is already injected into your "
            "context at session start — rely on that injected listing to pick stickers. "
            "Call 'list' ONLY to verify the result of a manage action you just performed "
            "(update/remove/add_set), or when you have a concrete reason to believe the "
            "collection changed mid-session. Never call it just to browse. This is a "
            "local collection, not a global Telegram sticker search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "update", "remove", "add_set"],
                    "description": "Which management operation to perform.",
                },
                "sticker": {
                    "type": "string",
                    "description": (
                        "Selector for update/remove when you don't know the "
                        "file_unique_id: emoji or set_name:emoji, resolved against the "
                        "collection."
                    ),
                },
                "file_unique_id": {
                    "type": "string",
                    "description": (
                        "Selector for update/remove: the entry's file_unique_id "
                        "(exact match, preferred over sticker)."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "New description for action='update' (\"\" clears the annotation).",
                },
                "set_name": {
                    "type": "string",
                    "description": (
                        "Pack short name: filter for 'list', pack to import for 'add_set' "
                        "(https://t.me/addstickers/<name> links accepted)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entries returned by 'list' (default 100).",
                },
            },
            "required": ["action"],
        },
    },
    handler=_handle_tg_manage_stickers,
    check_fn=_check_telegram,
    is_async=True,
    emoji="🗂️",
)
