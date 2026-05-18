"""Discord server introspection and management tool.

Provides the agent with the ability to interact with Discord servers
when running on the Discord gateway. Uses Discord REST API directly
with the bot token — no dependency on the gateway adapter's client.

Only included in the hermes-discord toolset, so it has zero cost
for users on other platforms.

The schema exposed to the model is filtered by two gates:

1. Privileged intents detected from GET /applications/@me at schema
   build time. Actions that require an intent the bot doesn't have
   (search_members / member_info → GUILD_MEMBERS intent) are hidden.
   fetch_messages is kept regardless of MESSAGE_CONTENT intent, but
   its description is annotated when the intent is missing.

2. User config allowlist at ``discord.server_actions``. If the user
   sets a comma-separated list (or YAML list) of action names, only
   those appear in the schema. Empty/unset means all intent-available
   actions are exposed.

Per-guild permissions (MANAGE_ROLES etc.) are NOT pre-checked — Discord
returns a 403 at call time and :func:`_enrich_403` maps it to
actionable guidance the model can relay to the user.
"""

import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"

# Application flag bits (from GET /applications/@me → "flags").
# Source: https://discord.com/developers/docs/resources/application#application-object-application-flags
_FLAG_GATEWAY_GUILD_MEMBERS = 1 << 14
_FLAG_GATEWAY_GUILD_MEMBERS_LIMITED = 1 << 15
_FLAG_GATEWAY_MESSAGE_CONTENT = 1 << 18
_FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED = 1 << 19

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bot_token() -> Optional[str]:
    """Resolve the Discord bot token from environment."""
    return os.getenv("DISCORD_BOT_TOKEN", "").strip() or None


def _discord_request(
    method: str,
    path: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Any:
    """Make a request to the Discord REST API."""
    url = f"{DISCORD_API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent (https://github.com/NousResearch/hermes-agent)",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise DiscordAPIError(e.code, error_body) from e


class DiscordAPIError(Exception):
    """Raised when a Discord API call fails."""
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Discord API error {status}: {body}")


# ---------------------------------------------------------------------------
# Channel type mapping
# ---------------------------------------------------------------------------

_CHANNEL_TYPE_NAMES = {
    0: "text",
    2: "voice",
    4: "category",
    5: "announcement",
    10: "announcement_thread",
    11: "public_thread",
    12: "private_thread",
    13: "stage",
    15: "forum",
    16: "media",
}


def _channel_type_name(type_id: int) -> str:
    return _CHANNEL_TYPE_NAMES.get(type_id, f"unknown({type_id})")


# ---------------------------------------------------------------------------
# Capability detection (application intents)
# ---------------------------------------------------------------------------

# Module-level cache so the app/me endpoint is hit at most once per process.
_capability_cache: Dict[str, Dict[str, Any]] = {}


def _detect_capabilities(token: str, *, force: bool = False) -> Dict[str, Any]:
    """Detect the bot's app-wide capabilities via GET /applications/@me.

    Returns a dict with keys:

    - ``has_members_intent``: GUILD_MEMBERS intent is enabled
    - ``has_message_content``: MESSAGE_CONTENT intent is enabled
    - ``detected``: detection succeeded (False means exposing everything
      and letting runtime errors handle it)

    Cached in a module-global. Pass ``force=True`` to re-fetch.
    """
    global _capability_cache
    if token in _capability_cache and not force:
        return _capability_cache[token]

    caps: Dict[str, Any] = {
        "has_members_intent": True,
        "has_message_content": True,
        "detected": False,
    }

    try:
        app = _discord_request("GET", "/applications/@me", token, timeout=5)
        flags = int(app.get("flags", 0) or 0)
        caps["has_members_intent"] = bool(
            flags & (_FLAG_GATEWAY_GUILD_MEMBERS | _FLAG_GATEWAY_GUILD_MEMBERS_LIMITED)
        )
        caps["has_message_content"] = bool(
            flags & (_FLAG_GATEWAY_MESSAGE_CONTENT | _FLAG_GATEWAY_MESSAGE_CONTENT_LIMITED)
        )
        caps["detected"] = True
    except Exception as exc:  # nosec — detection is best-effort
        logger.info(
            "Discord capability detection failed (%s); exposing all actions.", exc,
        )

    _capability_cache[token] = caps
    return caps


def _reset_capability_cache() -> None:
    """Test hook: clear the detection cache."""
    global _capability_cache
    _capability_cache = {}


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _list_guilds(token: str, guild_id: str = "", **_kwargs: Any) -> str:
    """List guilds, scoped to the current guild when session context is present."""
    if guild_id:
        guild = _discord_request("GET", f"/guilds/{guild_id}", token)
        guilds = [guild]
    else:
        guilds = _discord_request("GET", "/users/@me/guilds", token)
    result = []
    for g in guilds:
        result.append({
            "id": g["id"],
            "name": g["name"],
            "icon": g.get("icon"),
            "owner": g.get("owner", False),
            "permissions": g.get("permissions"),
        })
    return json.dumps({"guilds": result, "count": len(result)})


def _server_info(token: str, guild_id: str, **_kwargs: Any) -> str:
    """Get detailed information about a guild."""
    g = _discord_request("GET", f"/guilds/{guild_id}", token, params={"with_counts": "true"})
    return json.dumps({
        "id": g["id"],
        "name": g["name"],
        "description": g.get("description"),
        "icon": g.get("icon"),
        "owner_id": g.get("owner_id"),
        "member_count": g.get("approximate_member_count"),
        "online_count": g.get("approximate_presence_count"),
        "features": g.get("features", []),
        "premium_tier": g.get("premium_tier"),
        "premium_subscription_count": g.get("premium_subscription_count"),
        "verification_level": g.get("verification_level"),
    })


def _list_channels(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all channels in a guild, organized by category."""
    channels = _discord_request("GET", f"/guilds/{guild_id}/channels", token)

    # Organize: categories first, then channels under each
    categories: Dict[Optional[str], Dict[str, Any]] = {}
    uncategorized: List[Dict[str, Any]] = []

    # First pass: collect categories
    for ch in channels:
        if ch["type"] == 4:  # category
            categories[ch["id"]] = {
                "id": ch["id"],
                "name": ch["name"],
                "position": ch.get("position", 0),
                "channels": [],
            }

    # Second pass: assign channels to categories
    for ch in channels:
        if ch["type"] == 4:
            continue
        entry = {
            "id": ch["id"],
            "name": ch.get("name", ""),
            "type": _channel_type_name(ch["type"]),
            "position": ch.get("position", 0),
            "topic": ch.get("topic"),
            "nsfw": ch.get("nsfw", False),
        }
        parent = ch.get("parent_id")
        if parent and parent in categories:
            categories[parent]["channels"].append(entry)
        else:
            uncategorized.append(entry)

    # Sort
    sorted_cats = sorted(categories.values(), key=lambda c: c["position"])
    for cat in sorted_cats:
        cat["channels"].sort(key=lambda c: c["position"])
    uncategorized.sort(key=lambda c: c["position"])

    result: List[Dict[str, Any]] = []
    if uncategorized:
        result.append({"category": None, "channels": uncategorized})
    for cat in sorted_cats:
        result.append({
            "category": {"id": cat["id"], "name": cat["name"]},
            "channels": cat["channels"],
        })

    total = sum(len(group["channels"]) for group in result)
    return json.dumps({"channel_groups": result, "total_channels": total})


def _channel_info(token: str, channel_id: str, **_kwargs: Any) -> str:
    """Get detailed info about a specific channel."""
    ch = _discord_request("GET", f"/channels/{channel_id}", token)
    return json.dumps({
        "id": ch["id"],
        "name": ch.get("name"),
        "type": _channel_type_name(ch["type"]),
        "guild_id": ch.get("guild_id"),
        "topic": ch.get("topic"),
        "nsfw": ch.get("nsfw", False),
        "position": ch.get("position"),
        "parent_id": ch.get("parent_id"),
        "rate_limit_per_user": ch.get("rate_limit_per_user", 0),
        "last_message_id": ch.get("last_message_id"),
    })


def _list_roles(token: str, guild_id: str, **_kwargs: Any) -> str:
    """List all roles in a guild."""
    roles = _discord_request("GET", f"/guilds/{guild_id}/roles", token)
    result = []
    for r in sorted(roles, key=lambda r: r.get("position", 0), reverse=True):
        result.append({
            "id": r["id"],
            "name": r["name"],
            "color": f"#{r.get('color', 0):06x}" if r.get("color") else None,
            "position": r.get("position", 0),
            "mentionable": r.get("mentionable", False),
            "managed": r.get("managed", False),
            "member_count": r.get("member_count"),
            "hoist": r.get("hoist", False),
        })
    return json.dumps({"roles": result, "count": len(result)})


def _member_info(token: str, guild_id: str, user_id: str, **_kwargs: Any) -> str:
    """Get info about a specific guild member."""
    m = _discord_request("GET", f"/guilds/{guild_id}/members/{user_id}", token)
    user = m.get("user", {})
    return json.dumps({
        "user_id": user.get("id"),
        "username": user.get("username"),
        "display_name": user.get("global_name"),
        "nickname": m.get("nick"),
        "avatar": user.get("avatar"),
        "bot": user.get("bot", False),
        "roles": m.get("roles", []),
        "joined_at": m.get("joined_at"),
        "premium_since": m.get("premium_since"),
    })


def _search_members(token: str, guild_id: str, query: str, limit: int = 20, **_kwargs: Any) -> str:
    """Search for guild members by name."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 20
    params = {"query": query, "limit": str(min(limit, 100))}
    members = _discord_request("GET", f"/guilds/{guild_id}/members/search", token, params=params)
    result = []
    for m in members:
        user = m.get("user", {})
        result.append({
            "user_id": user.get("id"),
            "username": user.get("username"),
            "display_name": user.get("global_name"),
            "nickname": m.get("nick"),
            "bot": user.get("bot", False),
            "roles": m.get("roles", []),
        })
    return json.dumps({"members": result, "count": len(result)})


def _fetch_messages(
    token: str, channel_id: str, limit: int = 50,
    before: Optional[str] = None, after: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Fetch recent messages from a channel."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    params: Dict[str, str] = {"limit": str(min(limit, 100))}
    if before:
        params["before"] = before
    if after:
        params["after"] = after
    messages = _discord_request("GET", f"/channels/{channel_id}/messages", token, params=params)
    result = []
    for msg in messages:
        author = msg.get("author", {})
        result.append({
            "id": msg["id"],
            "content": msg.get("content", ""),
            "author": {
                "id": author.get("id"),
                "username": author.get("username"),
                "display_name": author.get("global_name"),
                "bot": author.get("bot", False),
            },
            "timestamp": msg.get("timestamp"),
            "edited_timestamp": msg.get("edited_timestamp"),
            "attachments": [
                {"filename": a.get("filename"), "url": a.get("url"), "size": a.get("size")}
                for a in msg.get("attachments", [])
            ],
            "reactions": [
                {"emoji": r.get("emoji", {}).get("name"), "count": r.get("count", 0)}
                for r in msg.get("reactions", [])
            ] if msg.get("reactions") else [],
            "pinned": msg.get("pinned", False),
        })
    return json.dumps({"messages": result, "count": len(result)})


def _list_pins(token: str, channel_id: str, **_kwargs: Any) -> str:
    """List pinned messages in a channel."""
    messages = _discord_request("GET", f"/channels/{channel_id}/pins", token)
    result = []
    for msg in messages:
        author = msg.get("author", {})
        result.append({
            "id": msg["id"],
            "content": msg.get("content", "")[:200],  # Truncate for overview
            "author": author.get("username"),
            "timestamp": msg.get("timestamp"),
        })
    return json.dumps({"pinned_messages": result, "count": len(result)})


def _pin_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Pin a message in a channel."""
    _discord_request("PUT", f"/channels/{channel_id}/pins/{message_id}", token)
    return json.dumps({"success": True, "message": f"Message {message_id} pinned."})


def _unpin_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Unpin a message from a channel."""
    _discord_request("DELETE", f"/channels/{channel_id}/pins/{message_id}", token)
    return json.dumps({"success": True, "message": f"Message {message_id} unpinned."})


def _delete_message(token: str, channel_id: str, message_id: str, **_kwargs: Any) -> str:
    """Delete a message from a channel or thread."""
    _discord_request("DELETE", f"/channels/{channel_id}/messages/{message_id}", token)
    return json.dumps({"success": True, "message": f"Message {message_id} deleted."})


def _create_thread(
    token: str, channel_id: str, name: str,
    message_id: Optional[str] = None,
    auto_archive_duration: int = 1440,
    **_kwargs: Any,
) -> str:
    """Create a thread in a channel."""
    if message_id:
        # Create thread from an existing message
        path = f"/channels/{channel_id}/messages/{message_id}/threads"
        body: Dict[str, Any] = {
            "name": name,
            "auto_archive_duration": auto_archive_duration,
        }
    else:
        # Create a standalone thread
        path = f"/channels/{channel_id}/threads"
        body = {
            "name": name,
            "auto_archive_duration": auto_archive_duration,
            "type": 11,  # PUBLIC_THREAD
        }
    thread = _discord_request("POST", path, token, body=body)
    return json.dumps({
        "success": True,
        "thread_id": thread["id"],
        "name": thread.get("name"),
    })


def _add_role(token: str, guild_id: str, user_id: str, role_id: str, **_kwargs: Any) -> str:
    """Add a role to a guild member."""
    _discord_request("PUT", f"/guilds/{guild_id}/members/{user_id}/roles/{role_id}", token)
    return json.dumps({"success": True, "message": f"Role {role_id} added to user {user_id}."})


def _remove_role(token: str, guild_id: str, user_id: str, role_id: str, **_kwargs: Any) -> str:
    """Remove a role from a guild member."""
    _discord_request("DELETE", f"/guilds/{guild_id}/members/{user_id}/roles/{role_id}", token)
    return json.dumps({"success": True, "message": f"Role {role_id} removed from user {user_id}."})


# ---------------------------------------------------------------------------
# Action dispatch + metadata
# ---------------------------------------------------------------------------

_ACTIONS = {
    "list_guilds": _list_guilds,
    "server_info": _server_info,
    "list_channels": _list_channels,
    "channel_info": _channel_info,
    "list_roles": _list_roles,
    "member_info": _member_info,
    "search_members": _search_members,
    "fetch_messages": _fetch_messages,
    "list_pins": _list_pins,
    "pin_message": _pin_message,
    "unpin_message": _unpin_message,
    "delete_message": _delete_message,
    "create_thread": _create_thread,
    "add_role": _add_role,
    "remove_role": _remove_role,
}

_CORE_ACTION_NAMES = frozenset({"fetch_messages", "search_members", "create_thread"})
_ADMIN_ACTION_NAMES = frozenset(_ACTIONS.keys()) - _CORE_ACTION_NAMES

_CORE_ACTIONS = {k: v for k, v in _ACTIONS.items() if k in _CORE_ACTION_NAMES}
_ADMIN_ACTIONS = {k: v for k, v in _ACTIONS.items() if k in _ADMIN_ACTION_NAMES}

# Single-source-of-truth manifest: action → (signature, one-line description).
# Consumed by :func:`_build_schema` so the schema's top-level description
# always matches the registered action set.
_ACTION_MANIFEST: List[Tuple[str, str, str]] = [
    ("list_guilds", "()", "list servers the bot is in"),
    ("server_info", "(guild_id)", "server details + member counts"),
    ("list_channels", "(guild_id)", "all channels grouped by category"),
    ("channel_info", "(channel_id)", "single channel details"),
    ("list_roles", "(guild_id)", "roles sorted by position"),
    ("member_info", "(guild_id, user_id)", "lookup a specific member"),
    ("search_members", "(guild_id, query)", "find members by name prefix"),
    ("fetch_messages", "(channel_id)", "recent messages; optional before/after snowflakes"),
    ("list_pins", "(channel_id)", "pinned messages in a channel"),
    ("pin_message", "(channel_id, message_id)", "pin a message"),
    ("unpin_message", "(channel_id, message_id)", "unpin a message"),
    ("delete_message", "(channel_id, message_id)", "delete a message"),
    ("create_thread", "(channel_id, name)", "create a public thread; optional message_id anchor"),
    ("add_role", "(guild_id, user_id, role_id)", "assign a role"),
    ("remove_role", "(guild_id, user_id, role_id)", "remove a role"),
]

# Actions that require the GUILD_MEMBERS privileged intent.
_INTENT_GATED_MEMBERS = frozenset({"member_info", "search_members"})

# Per-action required params for runtime validation.
_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "server_info": ["guild_id"],
    "list_channels": ["guild_id"],
    "list_roles": ["guild_id"],
    "member_info": ["guild_id", "user_id"],
    "search_members": ["guild_id", "query"],
    "channel_info": ["channel_id"],
    "fetch_messages": ["channel_id"],
    "list_pins": ["channel_id"],
    "pin_message": ["channel_id", "message_id"],
    "unpin_message": ["channel_id", "message_id"],
    "delete_message": ["channel_id", "message_id"],
    "create_thread": ["channel_id", "name"],
    "add_role": ["guild_id", "user_id", "role_id"],
    "remove_role": ["guild_id", "user_id", "role_id"],
}

_PERMISSION_ADMINISTRATOR = 1 << 3
_PERMISSION_SEND_MESSAGES = 1 << 11
_PERMISSION_MANAGE_MESSAGES = 1 << 13
_PERMISSION_MANAGE_CHANNELS = 1 << 4
_PERMISSION_MANAGE_ROLES = 1 << 28
_PERMISSION_CREATE_PUBLIC_THREADS = 1 << 35

_GUILD_PERMISSION_REQUIREMENTS: Dict[str, int] = {
    "list_channels": _PERMISSION_MANAGE_CHANNELS,
    "pin_message": _PERMISSION_MANAGE_MESSAGES,
    "unpin_message": _PERMISSION_MANAGE_MESSAGES,
    "delete_message": _PERMISSION_MANAGE_MESSAGES,
    "create_thread": _PERMISSION_SEND_MESSAGES | _PERMISSION_CREATE_PUBLIC_THREADS,
    "add_role": _PERMISSION_MANAGE_ROLES,
    "remove_role": _PERMISSION_MANAGE_ROLES,
}

_CHANNEL_SCOPED_ACTIONS = frozenset({
    "channel_info", "fetch_messages", "list_pins", "pin_message",
    "unpin_message", "delete_message", "create_thread",
})
_GUILD_SCOPED_ACTIONS = frozenset({
    "server_info", "list_channels", "list_roles", "member_info",
    "search_members", "add_role", "remove_role",
})


def _read_discord_session_context() -> Dict[str, str]:
    """Return the current gateway session context relevant to Discord authz."""
    session_context = sys.modules.get("gateway.session_context")
    get_session_env = getattr(session_context, "get_session_env", os.getenv)
    return {
        "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
        "guild_id": get_session_env("HERMES_SESSION_GUILD_ID", ""),
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "parent_chat_id": get_session_env("HERMES_SESSION_PARENT_CHAT_ID", ""),
        "user_id": get_session_env("HERMES_SESSION_USER_ID", ""),
    }


def _parse_permissions(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _load_member_permissions(
    token: str, guild_id: str, user_id: str,
) -> Tuple[int, Dict[str, int], List[str]]:
    """Compute guild-level permissions and role positions for the invoking member."""
    member = _discord_request("GET", f"/guilds/{guild_id}/members/{user_id}", token)
    roles = _discord_request("GET", f"/guilds/{guild_id}/roles", token)
    role_by_id = {str(role.get("id")): role for role in roles}
    permissions = _parse_permissions(role_by_id.get(str(guild_id), {}).get("permissions"))
    positions: Dict[str, int] = {
        str(role.get("id")): int(role.get("position", 0) or 0)
        for role in roles
    }
    for role_id in member.get("roles", []):
        permissions |= _parse_permissions(role_by_id.get(str(role_id), {}).get("permissions"))
    return permissions, positions, [str(role_id) for role_id in member.get("roles", [])]


def _has_permissions(permissions: int, required: int) -> bool:
    return bool(permissions & _PERMISSION_ADMINISTRATOR) or (permissions & required) == required


def _authorize_discord_tool_call(args: Dict[str, Any], token: str) -> Optional[str]:
    """Bind registry-dispatched Discord tools to the invoking Discord session."""
    ctx = _read_discord_session_context()
    action = str(args.get("action") or "")
    platform = ctx.get("platform") or ""
    if platform != "discord":
        return "Discord tools are available only inside a Discord gateway session."

    current_guild_id = ctx.get("guild_id") or ""
    current_channel_ids = {v for v in (ctx.get("chat_id"), ctx.get("parent_chat_id")) if v}
    requester_id = ctx.get("user_id") or ""
    if not current_guild_id or not requester_id:
        return "Discord tools require a server message with a known invoking user."

    target_guild_id = str(args.get("guild_id") or "")
    if action == "list_guilds":
        args["guild_id"] = current_guild_id
        target_guild_id = current_guild_id
    elif action in _GUILD_SCOPED_ACTIONS:
        if not target_guild_id:
            args["guild_id"] = current_guild_id
            target_guild_id = current_guild_id
        if target_guild_id != current_guild_id:
            return "Discord tool access is limited to the current server."

    target_channel_id = str(args.get("channel_id") or "")
    if action in _CHANNEL_SCOPED_ACTIONS:
        if not target_channel_id:
            return f"Missing required parameters for '{action}': channel_id"
        if target_channel_id not in current_channel_ids:
            return "Discord channel actions are limited to the current channel or its parent channel."

    required = _GUILD_PERMISSION_REQUIREMENTS.get(action, 0)
    if required:
        permissions, role_positions, requester_role_ids = _load_member_permissions(
            token, current_guild_id, requester_id,
        )
        if not _has_permissions(permissions, required):
            return "The invoking Discord user lacks the permissions required for this action."
        if action in {"add_role", "remove_role"} and not (permissions & _PERMISSION_ADMINISTRATOR):
            role_id = str(args.get("role_id") or "")
            requester_positions = [
                role_positions.get(str(role_id), 0)
                for role_id in requester_role_ids
            ]
            requester_top = max(requester_positions or [0])
            if requester_top <= role_positions.get(role_id, 0):
                return "The invoking Discord user cannot manage that role due to role hierarchy."

    return None


# ---------------------------------------------------------------------------
# Config-based action allowlist
# ---------------------------------------------------------------------------

def _load_allowed_actions_config() -> Optional[List[str]]:
    """Read ``discord.server_actions`` from user config.

    Returns a list of allowed action names, or ``None`` if the user
    hasn't restricted the set (default: all actions allowed).

    Accepts either a comma-separated string or a YAML list.
    Unknown action names are dropped with a log warning.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.debug("discord: could not load config (%s); allowing all actions.", exc)
        return None

    raw = (cfg.get("discord") or {}).get("server_actions")
    if raw is None or raw == "":
        return None

    if isinstance(raw, str):
        names = [n.strip() for n in raw.split(",") if n.strip()]
    elif isinstance(raw, (list, tuple)):
        names = [str(n).strip() for n in raw if str(n).strip()]
    else:
        logger.warning(
            "discord.server_actions: unexpected type %s; ignoring.", type(raw).__name__,
        )
        return None

    valid = [n for n in names if n in _ACTIONS]
    invalid = [n for n in names if n not in _ACTIONS]
    if invalid:
        logger.warning(
            "discord.server_actions: unknown action(s) ignored: %s. "
            "Known: %s",
            ", ".join(invalid), ", ".join(_ACTIONS.keys()),
        )
    return valid


def _available_actions(
    caps: Dict[str, Any],
    allowlist: Optional[List[str]],
) -> List[str]:
    """Compute the visible action list from intents + config allowlist.

    Preserves the canonical order from :data:`_ACTIONS`.
    """
    actions: List[str] = []
    for name in _ACTIONS:
        # Intent filter
        if not caps.get("has_members_intent", True) and name in _INTENT_GATED_MEMBERS:
            continue
        # Config allowlist filter
        if allowlist is not None and name not in allowlist:
            continue
        actions.append(name)
    return actions


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

def _build_schema(
    actions: List[str],
    caps: Optional[Dict[str, Any]] = None,
    tool_name: str = "discord",
) -> Optional[Dict[str, Any]]:
    """Build the tool schema for the given filtered action list.

    Returns ``None`` when *actions* is empty — callers should drop the
    tool from registration in that case.
    """
    caps = caps or {}
    if not actions:
        return None

    # Action manifest lines (action-first, parameter-scoped).
    manifest_lines = [
        f"  {name}{sig}  — {desc}"
        for name, sig, desc in _ACTION_MANIFEST
        if name in actions
    ]
    manifest_block = "\n".join(manifest_lines)

    content_note = ""
    affected_actions = {"fetch_messages", "list_pins"} & set(actions)
    if affected_actions and caps.get("detected") and caps.get("has_message_content") is False:
        names = " and ".join(sorted(affected_actions))
        content_note = (
            f"\n\nNOTE: Bot does NOT have the MESSAGE_CONTENT privileged intent. "
            f"{names} will return message metadata (author, "
            "timestamps, attachments, reactions, pin state) but `content` will be "
            "empty for messages not sent as a direct mention to the bot or in DMs. "
            "Enable the intent in the Discord Developer Portal to see all content."
        )

    if tool_name == "discord_admin":
        description = (
            "Manage a Discord server via the REST API.\n\n"
            "Available actions:\n"
            f"{manifest_block}\n\n"
            "Call list_guilds first to discover guild_ids, then list_channels for "
            "channel_ids. Runtime errors will tell you if the bot lacks a specific "
            "per-guild permission (e.g. MANAGE_ROLES for add_role)."
            f"{content_note}"
        )
    else:
        description = (
            "Read and participate in a Discord server.\n\n"
            "Available actions:\n"
            f"{manifest_block}\n\n"
            "Use the channel_id from the current conversation context. "
            "Use search_members to look up user IDs by name prefix."
            f"{content_note}"
        )

    properties: Dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": actions,
        },
        "guild_id": {
            "type": "string",
            "description": "Discord server (guild) ID.",
        },
        "channel_id": {
            "type": "string",
            "description": "Discord channel ID.",
        },
        "user_id": {
            "type": "string",
            "description": "Discord user ID.",
        },
        "role_id": {
            "type": "string",
            "description": "Discord role ID.",
        },
        "message_id": {
            "type": "string",
            "description": "Discord message ID.",
        },
        "query": {
            "type": "string",
            "description": "Member name prefix to search for (search_members).",
        },
        "name": {
            "type": "string",
            "description": "New thread name (create_thread).",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Max results (default 50). Applies to fetch_messages, search_members.",
        },
        "before": {
            "type": "string",
            "description": "Snowflake ID for reverse pagination (fetch_messages).",
        },
        "after": {
            "type": "string",
            "description": "Snowflake ID for forward pagination (fetch_messages).",
        },
        "auto_archive_duration": {
            "type": "integer",
            "enum": [60, 1440, 4320, 10080],
            "description": "Thread archive duration in minutes (create_thread, default 1440).",
        },
    }

    return {
        "name": tool_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["action"],
        },
    }


def _get_dynamic_schema(
    action_subset: Dict[str, Any],
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    """Build a dynamic schema for *action_subset* filtered by intents + config."""
    token = _get_bot_token()
    if not token:
        return None
    caps = _detect_capabilities(token)
    allowlist = _load_allowed_actions_config()
    actions = [a for a in _available_actions(caps, allowlist) if a in action_subset]
    if not actions:
        return None
    return _build_schema(actions, caps, tool_name=tool_name)


def get_dynamic_schema_core() -> Optional[Dict[str, Any]]:
    return _get_dynamic_schema(_CORE_ACTIONS, "discord")


def get_dynamic_schema_admin() -> Optional[Dict[str, Any]]:
    return _get_dynamic_schema(_ADMIN_ACTIONS, "discord_admin")


def get_dynamic_schema() -> Optional[Dict[str, Any]]:
    """Backward-compat wrapper — returns core schema."""
    return get_dynamic_schema_core()


# ---------------------------------------------------------------------------
# 403 error enrichment
# ---------------------------------------------------------------------------

_ACTION_403_HINT = {
    "pin_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel. "
        "Ask the server admin to grant the bot a role that has MANAGE_MESSAGES, "
        "or a per-channel overwrite."
    ),
    "unpin_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel."
    ),
    "delete_message": (
        "Bot lacks MANAGE_MESSAGES permission in this channel, or cannot view the channel/message."
    ),
    "create_thread": (
        "Bot lacks CREATE_PUBLIC_THREADS in this channel, or cannot view it."
    ),
    "add_role": (
        "Either the bot lacks MANAGE_ROLES, or the target role sits higher "
        "than the bot's highest role. Roles can only be assigned below the "
        "bot's own position in the role hierarchy."
    ),
    "remove_role": (
        "Either the bot lacks MANAGE_ROLES, or the target role sits higher "
        "than the bot's highest role."
    ),
    "fetch_messages": (
        "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY)."
    ),
    "list_pins": (
        "Bot cannot view this channel (missing VIEW_CHANNEL or READ_MESSAGE_HISTORY)."
    ),
    "channel_info": (
        "Bot cannot view this channel (missing VIEW_CHANNEL)."
    ),
    "search_members": (
        "Likely missing the Server Members privileged intent — enable it in the "
        "Discord Developer Portal under your bot's settings."
    ),
    "member_info": (
        "Bot cannot see this guild member (missing Server Members intent or "
        "insufficient permissions)."
    ),
}


def _enrich_403(action: str, body: str) -> str:
    """Return a user-friendly guidance string for a 403 on ``action``."""
    hint = _ACTION_403_HINT.get(action)
    base = f"Discord API 403 (forbidden) on '{action}'."
    if hint:
        return f"{base} {hint} (Raw: {body})"
    return f"{base} (Raw: {body})"


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

def check_discord_tool_requirements() -> bool:
    """Tool is available only when a Discord bot token is configured."""
    return bool(_get_bot_token())


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _run_discord_action(
    action: str,
    valid_actions: Dict[str, Any],
    tool_label: str,
    guild_id: str = "",
    channel_id: str = "",
    user_id: str = "",
    role_id: str = "",
    message_id: str = "",
    query: str = "",
    name: str = "",
    limit: int = 50,
    before: str = "",
    after: str = "",
    auto_archive_duration: int = 1440,
) -> str:
    """Shared handler logic for both discord tools."""
    token = _get_bot_token()
    if not token:
        return json.dumps({"error": "DISCORD_BOT_TOKEN not configured."})

    action_fn = valid_actions.get(action)
    if not action_fn:
        return json.dumps({
            "error": f"Unknown action: {action}",
            "available_actions": list(valid_actions.keys()),
        })

    # Config-level allowlist gate (defense in depth — schema already filtered,
    # but a stale cached schema from a prior config should not let denied
    # actions through).
    allowlist = _load_allowed_actions_config()
    if allowlist is not None and action not in allowlist:
        return json.dumps({
            "error": (
                f"Action '{action}' is disabled by config (discord.server_actions). "
                f"Allowed: {', '.join(allowlist) if allowlist else '<none>'}"
            ),
        })

    local_vars = {
        "guild_id": guild_id,
        "channel_id": channel_id,
        "user_id": user_id,
        "role_id": role_id,
        "message_id": message_id,
        "query": query,
        "name": name,
    }

    missing = [p for p in _REQUIRED_PARAMS.get(action, []) if not local_vars.get(p)]
    if missing:
        return json.dumps({
            "error": f"Missing required parameters for '{action}': {', '.join(missing)}",
        })

    try:
        return action_fn(
            token=token,
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            role_id=role_id,
            message_id=message_id,
            query=query,
            name=name,
            limit=limit,
            before=before,
            after=after,
            auto_archive_duration=auto_archive_duration,
        )
    except DiscordAPIError as e:
        logger.warning("Discord API error in %s action '%s': %s", tool_label, action, e)
        if e.status == 403:
            return json.dumps({"error": _enrich_403(action, e.body)})
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.exception("Unexpected error in %s action '%s'", tool_label, action)
        return json.dumps({"error": f"Unexpected error: {e}"})


def discord_core(action: str, **kwargs) -> str:
    """Execute a core Discord action (fetch_messages, search_members, create_thread)."""
    return _run_discord_action(action, _CORE_ACTIONS, "discord", **kwargs)


def discord_admin_handler(action: str, **kwargs) -> str:
    """Execute a Discord admin action (server management)."""
    return _run_discord_action(action, _ADMIN_ACTIONS, "discord_admin", **kwargs)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

_HANDLER_DEFAULTS = {
    "action": "", "guild_id": "", "channel_id": "", "user_id": "",
    "role_id": "", "message_id": "", "query": "", "name": "",
    "limit": 50, "before": "", "after": "", "auto_archive_duration": 1440,
}


def _make_handler(handler_fn):
    """Create a registry-compatible handler that enforces Discord session scope."""
    def _handler(args, **kw):
        call_args = {k: args.get(k, v) for k, v in _HANDLER_DEFAULTS.items()}
        token = _get_bot_token()
        if not token:
            return json.dumps({"error": "DISCORD_BOT_TOKEN not configured."})
        auth_error = _authorize_discord_tool_call(call_args, token)
        if auth_error:
            return json.dumps({"error": auth_error})
        return handler_fn(**call_args)
    return _handler


_STATIC_CORE_SCHEMA = _build_schema(
    list(_CORE_ACTIONS.keys()), caps={"detected": False}, tool_name="discord",
)
_STATIC_ADMIN_SCHEMA = _build_schema(
    list(_ADMIN_ACTIONS.keys()), caps={"detected": False}, tool_name="discord_admin",
)

registry.register(
    name="discord",
    toolset="discord",
    schema=_STATIC_CORE_SCHEMA,
    handler=_make_handler(discord_core),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
)

registry.register(
    name="discord_admin",
    toolset="discord_admin",
    schema=_STATIC_ADMIN_SCHEMA,
    handler=_make_handler(discord_admin_handler),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
)
