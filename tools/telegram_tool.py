"""Telegram chat introspection and management tool.

Provides the agent with the ability to enumerate and inspect Telegram
chats, members, and user profiles when running on (or alongside) the
Telegram gateway. Uses the official Telegram Bot API directly with the
bot token — no dependency on the gateway adapter's client.

Only included in the ``telegram`` toolset (and the cross-platform
``messaging`` toolset), so it has zero cost for users on other
platforms. The tool is gated by ``check_fn`` on ``TELEGRAM_BOT_TOKEN``
being set, so it disappears from the schema entirely when the env var
is missing.

Caveats the schema description is honest about:

* The Bot API has no ``getAllChats`` endpoint. ``list_chats`` returns
  the deduped set of Telegram chats in Hermes' gateway sessions file,
  with a ``getUpdates`` fallback only when no gateway sessions exist.
  This is fundamentally a "chats the gateway has seen" view, not an
  exhaustive global directory.

* Bots can only resolve user profiles for users they have at least one
  interaction with in a shared chat. ``get_user`` therefore returns
  best-effort metadata, and may return ``{"ok": false, "reason":
  "no shared chat"}`` for unknown IDs.

* All API errors are surfaced as ``{"error": "..."}`` JSON so the model
  can see the failure mode and retry / give up intelligently.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bot_token() -> Optional[str]:
    """Resolve the Telegram bot token from environment."""
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None


def _telegram_request(
    method: str,
    path: str,
    token: str,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Any:
    """Make a request to the Telegram Bot API.

    Telegram's Bot API uses GET with query params for read operations
    and POST with JSON body for write operations. We accept both shapes
    via the ``params`` and ``body`` arguments.
    """
    url = f"{TELEGRAM_API_BASE}/bot{token}/{path}"

    data = None
    headers = {"User-Agent": "Hermes-Agent (https://github.com/NousResearch/hermes-agent)"}

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif params and method.upper() == "GET":
        url += "?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None and v != ""}
        )

    req = urllib.request.Request(url, data=data, method=method, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise TelegramAPIError(e.code, body_text) from e
    except urllib.error.URLError as e:
        raise TelegramAPIError(0, f"Network error: {e.reason}") from e

    if not payload.get("ok", False):
        # Telegram's own protocol: ok:false, error_code, description
        raise TelegramAPIError(
            int(payload.get("error_code", 0) or 0),
            str(payload.get("description", "unknown error")),
        )
    return payload.get("result")


class TelegramAPIError(Exception):
    """Raised when a Telegram Bot API call fails."""
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Telegram API error {status}: {body}")


# ---------------------------------------------------------------------------
# Chat-type mapping (mirrors the official Bot API docs)
# ---------------------------------------------------------------------------

_CHAT_TYPE_NAMES = {
    "private": "dm",
    "group": "group",
    "supergroup": "supergroup",
    "channel": "channel",
}

_CHAT_TYPE_LABELS = {
    "dm": "private DM",
    "group": "group",
    "supergroup": "supergroup",
    "channel": "broadcast channel",
}


def _chat_type_name(t: str) -> str:
    return _CHAT_TYPE_NAMES.get(t, t)


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _get_me(token: str, **_kwargs: Any) -> str:
    """Return the bot's own identity (useful for verifying token + scope)."""
    me = _telegram_request("GET", "getMe", token)
    return json.dumps({
        "id": me.get("id"),
        "is_bot": me.get("is_bot"),
        "first_name": me.get("first_name"),
        "username": me.get("username"),
        "can_join_groups": me.get("can_join_groups"),
        "can_read_all_group_messages": me.get("can_read_all_group_messages"),
        "supports_inline_queries": me.get("supports_inline_queries"),
    })


def _list_chats(token: str, limit: int = 100, include_channels: bool = True, **_kwargs: Any) -> str:
    """Return the deduped set of chats the bot has interacted with.

    Strategy: read ``~/.hermes/sessions/sessions.json`` first, which the
    gateway maintains as it handles each conversation. This gives the
    actual "chats the bot is in / has talked to" directory with chat_id,
    display name, and chat_type — and crucially does **not** require a
    ``getUpdates`` call, so it never conflicts with the running
    gateway's long-poll.

    Fallback: if the sessions file is missing or empty (e.g. the user
    is running this tool standalone without the gateway), call
    ``getUpdates`` once with a short timeout. This will fail with a 409
    "terminated by other getUpdates request" if a gateway *is* running,
    which is correct — and surfaces as a structured error.

    Chat-type labels:

    * ``dm`` — private 1:1 chat (chat_id == user_id)
    * ``group`` — basic group
    * ``supergroup`` — supergroup (chat_id starts with -100)
    * ``channel`` — broadcast channel

    Channels are included by default; pass ``include_channels=false`` to
    drop them. The Bot API has no full directory; the result is "chats
    the gateway has seen", which for personal use is the complete set.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, 100))

    # ---- Primary: read from the gateway's sessions directory ----
    by_id: Dict[int, Dict[str, Any]] = {}
    source = "gateway_sessions"
    note = "Chats the running gateway has seen, from ~/.hermes/sessions/sessions.json."

    sessions_file = Path.home() / ".hermes" / "sessions" / "sessions.json"
    if sessions_file.exists():
        try:
            data = json.loads(sessions_file.read_text())
        except Exception as exc:
            logger.warning("telegram_tool: could not parse sessions.json: %s", exc)
            data = {}

        for session_key, meta in data.items():
            # Key format: agent:<profile>:<platform>:<chat_type>:<chat_id>
            parts = session_key.split(":")
            if len(parts) < 5 or parts[2] != "telegram":
                continue
            try:
                chat_id = int(parts[4])
            except (TypeError, ValueError):
                continue
            chat_type = _chat_type_name(parts[3] if len(parts) > 3 else "dm")
            display_name = (meta or {}).get("display_name") or f"chat_{chat_id}"
            username = (meta or {}).get("username")  # may be absent
            entry: Dict[str, Any] = {
                "id": chat_id,
                "type": chat_type,
                "type_label": _CHAT_TYPE_LABELS.get(chat_type, chat_type),
                "title": display_name,
            }
            if username:
                entry["username"] = username
            by_id[chat_id] = entry

    # ---- Fallback: getUpdates peek (will 409 if gateway is running) ----
    if not by_id:
        source = "getUpdates_fallback"
        note = "No gateway sessions found; queried getUpdates directly. This will conflict if a gateway is running."
        try:
            updates = _telegram_request(
                "GET", "getUpdates", token,
                params={"limit": str(limit), "timeout": 0, "allowed_updates": ["message", "edited_message", "channel_post", "edited_channel_post", "my_chat_member", "chat_member"]},
                timeout=20,
            )
        except TelegramAPIError as exc:
            # 409 means the gateway is holding the lease — return a useful
            # hint rather than the raw error.
            if exc.status == 409:
                return json.dumps({
                    "chats": [],
                    "count": 0,
                    "source": source,
                    "note": note,
                    "error": "getUpdates conflicted with the running gateway (409). Start a conversation in Telegram to populate ~/.hermes/sessions/sessions.json, or stop the gateway first if you really want a direct getUpdates read.",
                })
            raise

        for upd in updates or []:
            ch = upd.get("message", {}).get("chat") \
                or upd.get("edited_message", {}).get("chat") \
                or upd.get("channel_post", {}).get("chat") \
                or upd.get("edited_channel_post", {}).get("chat") \
                or upd.get("my_chat_member", {}).get("chat") \
                or upd.get("chat_member", {}).get("chat") \
                or upd.get("callback_query", {}).get("message", {}).get("chat")
            if not ch or "id" not in ch:
                continue
            cid = ch["id"]
            if cid in by_id:
                continue
            by_id[cid] = {
                "id": cid,
                "type": _chat_type_name(ch.get("type", "")),
                "type_label": _CHAT_TYPE_LABELS.get(_chat_type_name(ch.get("type", "")), ch.get("type")),
                "title": ch.get("title") or ch.get("first_name") or ch.get("username") or f"chat_{cid}",
                **({"username": ch["username"]} if ch.get("username") else {}),
            }

    chats = list(by_id.values())
    if not include_channels:
        chats = [c for c in chats if c["type"] != "channel"]

    # Stable order: DMs first, then by title
    chats.sort(key=lambda c: (c["type"] != "dm", (c.get("title") or "").lower()))

    return json.dumps({
        "chats": chats,
        "count": len(chats),
        "source": source,
        "note": note,
    })


def _chat_info(token: str, chat_id: str, **_kwargs: Any) -> str:
    """Return details about a single chat."""
    if not chat_id:
        return json.dumps({"error": "chat_id is required for chat_info"})
    ch = _telegram_request("GET", "getChat", token, params={"chat_id": chat_id})
    out: Dict[str, Any] = {
        "id": ch.get("id"),
        "type": _chat_type_name(ch.get("type", "")),
        "type_label": _CHAT_TYPE_LABELS.get(_chat_type_name(ch.get("type", "")), ch.get("type")),
        "title": ch.get("title"),
        "username": ch.get("username"),
        "first_name": ch.get("first_name"),
        "last_name": ch.get("last_name"),
        "is_forum": ch.get("is_forum"),
        "description": ch.get("description"),
        "pinned_message_id": (ch.get("pinned_message") or {}).get("message_id"),
        "permissions": ch.get("permissions"),
    }
    # Drop empty fields for a cleaner response
    out = {k: v for k, v in out.items() if v not in (None, "", [], {})}
    return json.dumps(out)


def _list_members(token: str, chat_id: str, limit: int = 50, **_kwargs: Any) -> str:
    """Return a best-effort list of chat members.

    The Bot API exposes ``getChatMembersCount`` for the total and the
    iterator ``getChatAdministrators`` for admins. For *all* members in
    a large supergroup, the API doesn't return a complete list — the
    bot only sees admins (always) and members it has explicitly
    promoted. The ``chat_member`` event stream fills in the rest as
    users interact, but is not a directory.

    What this action returns:
    * ``admins`` — every administrator of the chat
    * ``owner`` — the chat owner (if it surfaces in the admin list)
    * ``total_count`` — total member count (may be approximate)
    * ``scope`` — the limitations of what the bot can see

    Note: private (DM) chats do not have administrators. Calling this
    on a DM returns a friendly hint rather than the raw Telegram 400.
    """
    if not chat_id:
        return json.dumps({"error": "chat_id is required for list_members"})
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 50))

    # DMs have no admins. Detect early to avoid a 400 from the API.
    try:
        cid_int = int(chat_id)
        if cid_int > 0:  # user IDs and DM chat_ids are positive; groups are negative
            # Confirm via getChat — for DMs the type will be "private"
            chat = _telegram_request("GET", "getChat", token, params={"chat_id": chat_id}, timeout=5)
            if (chat.get("type") or "").lower() == "private":
                return json.dumps({
                    "chat_id": str(chat_id),
                    "error": "list_members is not meaningful for private (DM) chats — they have no administrators. Use get_user to look up a specific user instead.",
                    "scope": "DM chats have no admin roster.",
                })
    except (ValueError, TelegramAPIError):
        # If we can't pre-classify, fall through and let the API answer.
        pass

    admins = _telegram_request(
        "GET", "getChatAdministrators", token, params={"chat_id": chat_id}
    ) or []
    try:
        total = _telegram_request(
            "GET", "getChatMemberCount", token, params={"chat_id": chat_id}
        )
    except TelegramAPIError:
        total = None

    admin_list: List[Dict[str, Any]] = []
    owner: Optional[Dict[str, Any]] = None
    for m in admins:
        user = m.get("user", {})
        entry = {
            "user_id": user.get("id"),
            "username": user.get("username"),
            "first_name": user.get("first_name"),
            "last_name": user.get("last_name"),
            "is_bot": user.get("is_bot", False),
            "status": m.get("status"),
            "custom_title": m.get("custom_title"),
        }
        if m.get("status") == "creator":
            owner = entry
        else:
            admin_list.append(entry)
        if len(admin_list) >= limit:
            break

    return json.dumps({
        "chat_id": str(chat_id),
        "owner": owner,
        "admins": admin_list,
        "admin_count": len(admin_list),
        "admin_limit": limit,
        "total_count": total,
        "scope": "Bot API returns admins always and members only on interaction. Not a complete member directory.",
    })


def _get_user(token: str, user_id: str, chat_id: str = "", **_kwargs: Any) -> str:
    """Best-effort user profile lookup.

    Telegram's Bot API has no general ``getUser(user_id)``. The two
    options are:

    1. ``getChatMember(chat_id, user_id)`` — returns the user object
       only if the user is a member of ``chat_id`` (private chats work
       when ``chat_id`` is the same as ``user_id``). Requires a chat
       the user is in.

    2. Use a recent message's ``from`` field — not exposed here.

    If the bot shares no chat with the user, this returns a structured
    "no shared chat" response rather than guessing.
    """
    if not user_id:
        return json.dumps({"error": "user_id is required for get_user"})

    # For private chats, the user_id is also the chat_id
    target_chat = chat_id or user_id
    try:
        m = _telegram_request(
            "GET", "getChatMember", token,
            params={"chat_id": target_chat, "user_id": user_id},
        )
    except TelegramAPIError as e:
        # 400 with "USER_NOT_FOUND" or "chat not found" -> not a shared chat
        return json.dumps({
            "ok": False,
            "user_id": user_id,
            "reason": "no shared chat" if "USER_NOT_FOUND" in e.body or "chat not found" in e.body.lower() else f"api_error: {e.body}",
        })
    user = m.get("user", {})
    return json.dumps({
        "ok": True,
        "user": {
            "id": user.get("id"),
            "is_bot": user.get("is_bot", False),
            "username": user.get("username"),
            "first_name": user.get("first_name"),
            "last_name": user.get("last_name"),
            "language_code": user.get("language_code"),
            "is_premium": user.get("is_premium"),
        },
        "chat_id": str(target_chat),
        "status_in_chat": m.get("status"),
    })


# ---------------------------------------------------------------------------
# Action dispatch + metadata
# ---------------------------------------------------------------------------

_ACTIONS = {
    "get_me": _get_me,
    "list_chats": _list_chats,
    "chat_info": _chat_info,
    "list_members": _list_members,
    "get_user": _get_user,
}

_ACTION_MANIFEST: List[tuple] = [
    ("get_me",      "()",                "verify bot identity and capabilities (token health check)"),
    ("list_chats",  "(limit=100)",        "deduped chats from Hermes gateway sessions; DMs first, then groups/channels"),
    ("chat_info",   "(chat_id)",         "single chat details: title, type, username, description, permissions"),
    ("list_members","(chat_id, limit=50)","admins + member count for a chat; full member directory not available"),
    ("get_user",    "(user_id)",         "user profile (best-effort; requires a shared chat)"),
]

_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "chat_info": ["chat_id"],
    "list_members": ["chat_id"],
    "get_user": ["user_id"],
}


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

def check_telegram_tool_requirements() -> bool:
    """Tool is available only when a Telegram bot token is configured."""
    return bool(_get_bot_token())


# ---------------------------------------------------------------------------
# Handler + registration
# ---------------------------------------------------------------------------

_HANDLER_DEFAULTS = {
    "action": "",
    "chat_id": "",
    "user_id": "",
    "limit": 100,
    "include_channels": True,
}


def _as_bool(value: Any, default: bool = True) -> bool:
    """Parse common string/JSON boolean shapes."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _telegram_handler(args: Dict[str, Any], **_kw: Any) -> str:
    """Registry-compatible handler for the ``telegram`` tool."""
    token = _get_bot_token()
    if not token:
        return json.dumps({"error": "TELEGRAM_BOT_TOKEN not configured."})

    action = (args.get("action") or "").strip()
    if not action:
        return json.dumps({
            "error": "action is required",
            "available_actions": list(_ACTIONS.keys()),
        })

    fn = _ACTIONS.get(action)
    if not fn:
        return json.dumps({
            "error": f"Unknown action: {action}",
            "available_actions": list(_ACTIONS.keys()),
        })

    missing = [p for p in _REQUIRED_PARAMS.get(action, []) if not args.get(p)]
    if missing:
        return json.dumps({
            "error": f"Missing required parameters for '{action}': {', '.join(missing)}",
        })

    try:
        return fn(
            token=token,
            chat_id=args.get("chat_id", ""),
            user_id=args.get("user_id", ""),
            limit=int(args.get("limit", 100)),
            include_channels=_as_bool(args.get("include_channels"), True),
        )
    except TelegramAPIError as e:
        logger.warning("Telegram API error in action '%s': %s", action, e)
        return json.dumps({"error": str(e), "status": e.status})
    except Exception as e:
        logger.exception("Unexpected error in telegram action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


# Build the schema once at import time. The schema is gated on token
# presence via check_fn, so if the token is missing the registry hides
# the tool entirely. We always build a full schema here; the gate is at
# registration lookup, not schema build.

_manifest_lines = "\n".join(
    f"  {name}{sig}  — {desc}"
    for name, sig, desc in _ACTION_MANIFEST
)

_TELEGRAM_SCHEMA = {
    "name": "telegram",
    "description": (
        "Telegram chat and member introspection via the Bot API.\n\n"
        "Available actions:\n"
        f"{_manifest_lines}\n\n"
        "Call get_me first to verify the bot is configured and reachable. "
        "Then list_chats to discover chat_ids, chat_info for details, "
        "list_members for admin rosters, and get_user for individual profiles.\n\n"
        "Limitations the model should be aware of: "
        "(1) the Bot API has no full chat directory — list_chats returns "
        "Telegram chats recorded in Hermes gateway sessions, with a getUpdates "
        "fallback only when no gateway sessions exist. "
        "(2) get_user requires a shared chat; if the user is unknown to the "
        "bot it returns ok:false with reason 'no shared chat'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS.keys()),
            },
            "chat_id": {
                "type": "string",
                "description": "Numeric Telegram chat ID (use list_chats or chat_info to discover).",
            },
            "user_id": {
                "type": "string",
                "description": "Numeric Telegram user ID (use get_user or a member listing to discover).",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Max results (list_chats default 100; list_members cap 50).",
            },
            "include_channels": {
                "type": "boolean",
                "default": True,
                "description": "If false, exclude broadcast channels from list_chats.",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="telegram",
    toolset="telegram",
    schema=_TELEGRAM_SCHEMA,
    handler=lambda args, **kw: _telegram_handler(args, **kw),
    check_fn=check_telegram_tool_requirements,
    requires_env=["TELEGRAM_BOT_TOKEN"],
)
