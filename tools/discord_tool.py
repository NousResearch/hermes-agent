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
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
_SESSION_KICKOFF_RECEIPT_ACTION = "post_session_kickoff_receipt_v1"
_SESSION_KICKOFF_RECEIPT_INTENT = "session_kickoff_receipt_v1"
_DISCORD_MESSAGE_LIMIT = 2000
_SESSION_KICKOFF_CONTENT_LIMIT = 20_000
_SESSION_KICKOFF_CHUNK_LIMIT = 10
_SESSION_KICKOFF_POST_429_ATTEMPT_LIMIT = 5
_SUPPRESS_EMBEDS_FLAG = 4
_BOUNDARY_PHRASES = (
    "Boundary: reply before mutation.",
    "Boundary: reply in this thread before mutation.",
    "awaiting explicit mutation approval",
)
_ACTIVATION_MARKERS = (
    "Activation prompt template",
    "Adoptable only by the user; user must adopt this prompt in their own message.",
    "Instruction after adoption: Use the kickoff seed it references. Proceed only within those boundaries.",
    "Class B actions without exact approval",
)
_INVISIBLE_LINE_CHARS = {
    ord("\ufeff"): None,
    ord("\u200b"): None,
    ord("\u200c"): None,
    ord("\u200d"): None,
    ord("\u2060"): None,
}
_ROLE_PING_RE = re.compile(r"<@&\d{1,20}>")

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


def _load_discord_config() -> Dict[str, Any]:
    """Load the ``discord`` config block, returning an empty dict on failure."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception as exc:
        logger.debug("discord: could not load config (%s); using defaults.", exc)
        return {}
    raw = cfg.get("discord") if isinstance(cfg, dict) else None
    return raw if isinstance(raw, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _session_kickoff_receipt_enabled() -> bool:
    """Return whether the opt-in receipt action should be exposed/run."""
    return _as_bool(_load_discord_config().get("session_kickoff_receipt_enabled", False))


def _is_discord_snowflake(value: Any) -> bool:
    """Discord IDs used in REST paths must be ASCII decimal snowflake strings."""
    return (
        isinstance(value, str)
        and value.isascii()
        and value.isdigit()
        and 1 <= len(value) <= 20
    )


def _first_visible_line(text: str) -> str:
    """Return the first non-blank visible line after minimal Discord normalization."""
    for line in str(text).splitlines():
        normalized = line.translate(_INVISIBLE_LINE_CHARS)
        if normalized.strip():
            return normalized.lstrip()
    return ""


def _has_forbidden_ping(text: str) -> bool:
    value = str(text)
    return "@everyone" in value or "@here" in value or bool(_ROLE_PING_RE.search(value))


def _split_discord_text(text: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Split text into deterministic Discord message chunks or return an error code."""
    value = str(text)
    if len(value) > _SESSION_KICKOFF_CONTENT_LIMIT:
        return None, "content_too_large"
    chunks = [
        value[index:index + _DISCORD_MESSAGE_LIMIT]
        for index in range(0, len(value), _DISCORD_MESSAGE_LIMIT)
    ] or [""]
    if len(chunks) > _SESSION_KICKOFF_CHUNK_LIMIT:
        return None, "too_many_discord_chunks"
    return chunks, None


def _receipt_result(
    status: str,
    *,
    channel_id: str = "",
    trusted_requester_user_id: str = "",
    message_ids: Optional[List[str]] = None,
    seed_message_ids: Optional[List[str]] = None,
    activation_message_ids: Optional[List[str]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "receipt_pass": status == "receipt_pass",
        "channel_id": channel_id,
        "trusted_requester_user_id": trusted_requester_user_id,
        "message_ids": message_ids or [],
        "seed_message_ids": seed_message_ids or [],
        "activation_message_ids": activation_message_ids or [],
        "errors": errors or [],
        "warnings": warnings or [],
        "evidence": evidence or {},
    }


def _receipt_error(code: str, message: str = "", **details: Any) -> Dict[str, Any]:
    error = {"code": code, "message": message or code}
    error.update(details)
    return error


def _receipt_json(status: str, code: str, **kwargs: Any) -> str:
    return json.dumps(_receipt_result(status, errors=[_receipt_error(code)], **kwargs))


def _parse_receipt_fetch_options(
    max_fetch_attempts: Any,
    fetch_delay_seconds: Any,
) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    if not isinstance(max_fetch_attempts, int) or isinstance(max_fetch_attempts, bool):
        return None, None, "invalid_max_fetch_attempts"
    if max_fetch_attempts < 1 or max_fetch_attempts > 5:
        return None, None, "invalid_max_fetch_attempts"
    if not isinstance(fetch_delay_seconds, (int, float)) or isinstance(fetch_delay_seconds, bool):
        return None, None, "invalid_fetch_delay_seconds"
    delay = float(fetch_delay_seconds)
    if delay < 0 or delay > 2.0 or (max_fetch_attempts - 1) * delay >= 8.0:
        return None, None, "invalid_fetch_delay_seconds"
    return max_fetch_attempts, delay, None


def _trusted_discord_requester_id() -> Optional[str]:
    try:
        from gateway.session_context import get_session_env_strict
    except Exception:
        return None
    if get_session_env_strict("HERMES_SESSION_PLATFORM") != "discord":
        return None
    user_id = get_session_env_strict("HERMES_SESSION_USER_ID")
    if not _is_discord_snowflake(user_id):
        return None
    return user_id


def _receipt_allowed_mentions(trusted_requester_user_id: str) -> Dict[str, Any]:
    return {
        "parse": [],
        "users": [trusted_requester_user_id],
        "roles": [],
        "replied_user": False,
    }


def _post_receipt_message(
    token: str,
    channel_id: str,
    content: str,
    trusted_requester_user_id: str,
    retry_sleep_used: float,
) -> Tuple[Dict[str, Any], float]:
    body = {
        "content": content,
        "flags": _SUPPRESS_EMBEDS_FLAG,
        "allowed_mentions": _receipt_allowed_mentions(trusted_requester_user_id),
    }
    rate_limit_attempts = 0
    while True:
        try:
            return _discord_request("POST", f"/channels/{channel_id}/messages", token, body=body), retry_sleep_used
        except DiscordAPIError as exc:
            if exc.status != 429:
                raise
            rate_limit_attempts += 1
            delay = _retry_after_seconds(exc)
            if (
                rate_limit_attempts >= _SESSION_KICKOFF_POST_429_ATTEMPT_LIMIT
                or retry_sleep_used + delay > 8.0
            ):
                raise
            if delay:
                time.sleep(delay)
            retry_sleep_used += delay


def _fetch_receipt_message(token: str, channel_id: str, message_id: str) -> Dict[str, Any]:
    return _discord_request("GET", f"/channels/{channel_id}/messages/{message_id}", token)


def _message_author_id(message: Dict[str, Any]) -> str:
    author = message.get("author") if isinstance(message, dict) else None
    if isinstance(author, dict):
        value = author.get("id")
        return value if isinstance(value, str) else ""
    return ""


def _receipt_failure(
    status: str,
    code: str,
    *,
    channel_id: str,
    trusted_requester_user_id: str,
    message_ids: Optional[List[str]] = None,
    seed_message_ids: Optional[List[str]] = None,
    activation_message_ids: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> str:
    return json.dumps(_receipt_result(
        status,
        channel_id=channel_id,
        trusted_requester_user_id=trusted_requester_user_id,
        message_ids=message_ids,
        seed_message_ids=seed_message_ids,
        activation_message_ids=activation_message_ids,
        errors=[error or _receipt_error(code)],
        evidence=evidence,
    ))


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
        headers = dict(e.headers.items()) if e.headers else {}
        raise DiscordAPIError(e.code, error_body, headers=headers) from e


class DiscordAPIError(Exception):
    """Raised when a Discord API call fails."""
    def __init__(self, status: int, body: str, headers: Optional[Dict[str, str]] = None):
        self.status = status
        self.body = body
        self.headers = headers or {}
        super().__init__(f"Discord API error {status}: {body}")


def _sanitize_discord_error_body(body: str) -> str:
    value = str(body or "")
    try:
        from agent.redact import redact_sensitive_text
        value = redact_sensitive_text(value, force=True)
    except Exception:
        pass
    token = _get_bot_token()
    if token:
        value = value.replace(token, "<redacted>")
    value = re.sub(
        r"([?&](?:token|access_token|api_key|signature|sig)=)[^&\s]+",
        r"\1<redacted>",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"/home/[A-Za-z0-9._~+/@%-]+", "<local-path-redacted>", value)
    return value[:1000]


def _error_from_discord_exception(code: str, exc: DiscordAPIError) -> Dict[str, Any]:
    return _receipt_error(
        code,
        http_status=exc.status,
        body=_sanitize_discord_error_body(exc.body),
    )


def _retry_after_seconds(exc: DiscordAPIError) -> float:
    retry_after: Any = None
    try:
        payload = json.loads(exc.body or "{}")
        retry_after = payload.get("retry_after")
    except Exception:
        retry_after = None
    if retry_after is None:
        retry_after = (exc.headers or {}).get("Retry-After") or (exc.headers or {}).get("retry-after")
    try:
        return min(2.0, max(0.0, float(retry_after)))
    except (TypeError, ValueError):
        return 0.5


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

def _list_guilds(token: str, **_kwargs: Any) -> str:
    """List all guilds the bot is a member of."""
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


def _post_session_kickoff_receipt(
    token: str,
    channel_id: str = "",
    seed_content: str = "",
    post_intent: str = "",
    expected_requester_user_id: str = "",
    activation_content: str = "",
    max_fetch_attempts: int = 3,
    fetch_delay_seconds: float = 0.5,
    **_kwargs: Any,
) -> str:
    """Opt-in Discord session-kickoff receipt action."""
    if not _session_kickoff_receipt_enabled():
        return _receipt_json("precheck_failed", "opt_in_disabled", channel_id=channel_id)

    requester_id = _trusted_discord_requester_id()
    if requester_id is None:
        return _receipt_json("precheck_failed", "unsupported_session_context", channel_id=channel_id)

    if post_intent != _SESSION_KICKOFF_RECEIPT_INTENT:
        return _receipt_json(
            "precheck_failed", "invalid_post_intent",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
        )

    if not _is_discord_snowflake(channel_id):
        return _receipt_json(
            "precheck_failed", "invalid_channel_id",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
        )

    if expected_requester_user_id:
        if not _is_discord_snowflake(expected_requester_user_id):
            return _receipt_json(
                "precheck_failed", "invalid_expected_requester_user_id",
                channel_id=channel_id, trusted_requester_user_id=requester_id,
            )
        if expected_requester_user_id != requester_id:
            return _receipt_json(
                "precheck_failed", "requester_user_id_mismatch",
                channel_id=channel_id, trusted_requester_user_id=requester_id,
            )

    first_line = _first_visible_line(seed_content)
    if not first_line.startswith(f"<@{requester_id}>"):
        return _receipt_json(
            "precheck_failed", "missing_requester_mention_first_line",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            evidence={"first_seed_line": first_line},
        )

    if _has_forbidden_ping(seed_content) or _has_forbidden_ping(activation_content):
        return _receipt_json(
            "precheck_failed", "forbidden_mass_or_role_ping",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            evidence={"first_seed_line": first_line},
        )

    if "PRE-START VERIFICATION GATE" not in seed_content:
        return _receipt_json(
            "precheck_failed", "missing_pre_start_gate",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            evidence={"first_seed_line": first_line},
        )

    if not any(phrase in seed_content for phrase in _BOUNDARY_PHRASES):
        return _receipt_json(
            "precheck_failed", "missing_boundary_phrase",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            evidence={"first_seed_line": first_line},
        )

    if activation_content and any(marker not in activation_content for marker in _ACTIVATION_MARKERS):
        return _receipt_json(
            "precheck_failed", "missing_activation_marker",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            evidence={"first_seed_line": first_line},
        )

    attempts, delay, option_error = _parse_receipt_fetch_options(max_fetch_attempts, fetch_delay_seconds)
    if option_error:
        return _receipt_json(
            "precheck_failed", option_error,
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            evidence={"first_seed_line": first_line},
        )

    expected_seed_chunks, seed_error = _split_discord_text(seed_content)
    if seed_error:
        return _receipt_json(
            "precheck_failed", seed_error,
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            evidence={"first_seed_line": first_line},
        )
    expected_activation_chunks: List[str] = []
    if activation_content:
        activation_chunks, activation_error = _split_discord_text(activation_content)
        if activation_error:
            return _receipt_json(
                "precheck_failed", activation_error,
                channel_id=channel_id, trusted_requester_user_id=requester_id,
                evidence={"first_seed_line": first_line},
            )
        expected_activation_chunks = activation_chunks or []

    expected_seed_chunks = expected_seed_chunks or []
    seed_message_ids: List[str] = []
    activation_message_ids: List[str] = []
    message_ids: List[str] = []
    bot_author_id = ""
    post_retry_sleep_used = 0.0

    try:
        for content in expected_seed_chunks:
            posted, post_retry_sleep_used = _post_receipt_message(
                token, channel_id, content, requester_id, post_retry_sleep_used,
            )
            posted_message_id = posted.get("id")
            message_id = posted_message_id if isinstance(posted_message_id, str) else ""
            author_id = _message_author_id(posted) if isinstance(posted, dict) else ""
            if not _is_discord_snowflake(message_id) or not _is_discord_snowflake(author_id):
                return _receipt_failure(
                    "post_failed", "post_failed",
                    channel_id=channel_id, trusted_requester_user_id=requester_id,
                    message_ids=message_ids, seed_message_ids=seed_message_ids,
                    activation_message_ids=activation_message_ids,
                    evidence={"first_seed_line": first_line},
                )
            if not bot_author_id:
                bot_author_id = author_id
            seed_message_ids.append(message_id)
            message_ids.append(message_id)

        for content in expected_activation_chunks:
            posted, post_retry_sleep_used = _post_receipt_message(
                token, channel_id, content, requester_id, post_retry_sleep_used,
            )
            posted_message_id = posted.get("id")
            message_id = posted_message_id if isinstance(posted_message_id, str) else ""
            author_id = _message_author_id(posted) if isinstance(posted, dict) else ""
            if not _is_discord_snowflake(message_id) or not _is_discord_snowflake(author_id):
                return _receipt_failure(
                    "partial_post", "partial_post",
                    channel_id=channel_id, trusted_requester_user_id=requester_id,
                    message_ids=message_ids, seed_message_ids=seed_message_ids,
                    activation_message_ids=activation_message_ids,
                    evidence={"first_seed_line": first_line},
                )
            activation_message_ids.append(message_id)
            message_ids.append(message_id)
    except DiscordAPIError as exc:
        status = "partial_post" if message_ids else "post_failed"
        code = "post_rate_limited" if exc.status == 429 else ("partial_post" if message_ids else "sanitized_discord_error")
        return _receipt_failure(
            status, code,
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            message_ids=message_ids, seed_message_ids=seed_message_ids,
            activation_message_ids=activation_message_ids,
            error=_error_from_discord_exception(code, exc),
            evidence={"first_seed_line": first_line},
        )

    fetched_messages: List[Dict[str, Any]] = []
    try:
        for message_id in message_ids:
            last_fetch_error: Optional[DiscordAPIError] = None
            for attempt_index in range(attempts or 1):
                try:
                    fetched = _fetch_receipt_message(token, channel_id, message_id)
                    if not isinstance(fetched, dict):
                        raise DiscordAPIError(0, "non-object Discord message response")
                    fetched_messages.append(fetched)
                    last_fetch_error = None
                    break
                except DiscordAPIError as exc:
                    last_fetch_error = exc
                    if attempt_index < (attempts or 1) - 1 and delay:
                        time.sleep(delay)
            if last_fetch_error is not None:
                raise last_fetch_error
    except DiscordAPIError as exc:
        return _receipt_failure(
            "fetch_failed", "message_fetch_failed",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            message_ids=message_ids, seed_message_ids=seed_message_ids,
            activation_message_ids=activation_message_ids,
            error=_error_from_discord_exception("message_fetch_failed", exc),
            evidence={"first_seed_line": first_line},
        )

    expected_chunks = expected_seed_chunks + expected_activation_chunks
    fetched_ids = [msg.get("id") if isinstance(msg.get("id"), str) else "" for msg in fetched_messages]
    fetched_contents = [str(msg.get("content", "")) for msg in fetched_messages]
    fetched_seed_content = "".join(fetched_contents[:len(expected_seed_chunks)])
    fetched_activation_content = "".join(fetched_contents[len(expected_seed_chunks):])
    fetched_first_line = _first_visible_line(fetched_seed_content)
    evidence = {
        "first_seed_line": fetched_first_line,
        "fetched_message_count": len(fetched_messages),
        "expected_seed_chunk_count": len(expected_seed_chunks),
        "expected_activation_chunk_count": len(expected_activation_chunks),
        "bot_author_id": bot_author_id,
        "embed_count": sum(len(msg.get("embeds") or []) for msg in fetched_messages),
    }

    if _has_forbidden_ping(fetched_seed_content) or _has_forbidden_ping(fetched_activation_content):
        return _receipt_failure(
            "validation_failed", "forbidden_mass_or_role_ping",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            message_ids=message_ids, seed_message_ids=seed_message_ids,
            activation_message_ids=activation_message_ids,
            evidence=evidence,
        )
    if fetched_ids != message_ids or fetched_contents != expected_chunks:
        return _receipt_failure(
            "validation_failed", "content_mismatch",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            message_ids=message_ids, seed_message_ids=seed_message_ids,
            activation_message_ids=activation_message_ids,
            evidence=evidence,
        )
    if not fetched_first_line.startswith(f"<@{requester_id}>"):
        return _receipt_failure(
            "validation_failed", "missing_requester_mention_first_line",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            message_ids=message_ids, seed_message_ids=seed_message_ids,
            activation_message_ids=activation_message_ids,
            evidence=evidence,
        )
    if "PRE-START VERIFICATION GATE" not in fetched_seed_content:
        return _receipt_failure(
            "validation_failed", "missing_pre_start_gate",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            message_ids=message_ids, seed_message_ids=seed_message_ids,
            activation_message_ids=activation_message_ids,
            evidence=evidence,
        )
    if not any(phrase in fetched_seed_content for phrase in _BOUNDARY_PHRASES):
        return _receipt_failure(
            "validation_failed", "missing_boundary_phrase",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            message_ids=message_ids, seed_message_ids=seed_message_ids,
            activation_message_ids=activation_message_ids,
            evidence=evidence,
        )
    if expected_activation_chunks and any(marker not in fetched_activation_content for marker in _ACTIVATION_MARKERS):
        return _receipt_failure(
            "validation_failed", "missing_activation_marker",
            channel_id=channel_id, trusted_requester_user_id=requester_id,
            message_ids=message_ids, seed_message_ids=seed_message_ids,
            activation_message_ids=activation_message_ids,
            evidence=evidence,
        )
    for msg in fetched_messages:
        if _message_author_id(msg) != bot_author_id:
            return _receipt_failure(
                "validation_failed", "author_mismatch",
                channel_id=channel_id, trusted_requester_user_id=requester_id,
                message_ids=message_ids, seed_message_ids=seed_message_ids,
                activation_message_ids=activation_message_ids,
                evidence=evidence,
            )
        if msg.get("attachments"):
            return _receipt_failure(
                "validation_failed", "unexpected_attachment_in_text_only_mvp",
                channel_id=channel_id, trusted_requester_user_id=requester_id,
                message_ids=message_ids, seed_message_ids=seed_message_ids,
                activation_message_ids=activation_message_ids,
                evidence=evidence,
            )
        if msg.get("embeds"):
            return _receipt_failure(
                "validation_failed", "unexpected_embed_in_text_only_mvp",
                channel_id=channel_id, trusted_requester_user_id=requester_id,
                message_ids=message_ids, seed_message_ids=seed_message_ids,
                activation_message_ids=activation_message_ids,
                evidence=evidence,
            )

    return json.dumps(_receipt_result(
        "receipt_pass",
        channel_id=channel_id,
        trusted_requester_user_id=requester_id,
        message_ids=message_ids,
        seed_message_ids=seed_message_ids,
        activation_message_ids=activation_message_ids,
        evidence=evidence,
    ))


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
    _SESSION_KICKOFF_RECEIPT_ACTION: _post_session_kickoff_receipt,
    "list_pins": _list_pins,
    "pin_message": _pin_message,
    "unpin_message": _unpin_message,
    "delete_message": _delete_message,
    "create_thread": _create_thread,
    "add_role": _add_role,
    "remove_role": _remove_role,
}

_CORE_ACTION_NAMES = frozenset({
    "fetch_messages", "search_members", "create_thread", _SESSION_KICKOFF_RECEIPT_ACTION,
})
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
    (
        _SESSION_KICKOFF_RECEIPT_ACTION,
        "(channel_id, seed_content, post_intent)",
        "post text-only session kickoff seed, fetch exact posted IDs, and validate receipt",
    ),
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
    *,
    receipt_enabled: bool = False,
) -> List[str]:
    """Compute the visible action list from intents + config allowlist.

    Preserves the canonical order from :data:`_ACTIONS`.
    """
    actions: List[str] = []
    for name in _ACTIONS:
        if name == _SESSION_KICKOFF_RECEIPT_ACTION and not receipt_enabled:
            continue
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

    if _SESSION_KICKOFF_RECEIPT_ACTION in actions:
        properties.update({
            "seed_content": {
                "type": "string",
                "description": "Text-only session kickoff seed content to post and validate.",
            },
            "post_intent": {
                "type": "string",
                "enum": [_SESSION_KICKOFF_RECEIPT_INTENT],
                "description": "Required literal intent for the kickoff receipt action.",
            },
            "expected_requester_user_id": {
                "type": "string",
                "description": "Optional cross-check against trusted Discord gateway requester ID.",
            },
            "activation_content": {
                "type": "string",
                "description": "Optional text-only activation prompt posted after the seed.",
            },
            "max_fetch_attempts": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Fetch-by-ID retry attempts for posted messages (default 3).",
            },
            "fetch_delay_seconds": {
                "type": "number",
                "minimum": 0,
                "maximum": 2.0,
                "description": "Delay between fetch attempts; total retry sleep must stay under 8s.",
            },
        })

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
    actions = [
        a for a in _available_actions(
            caps, allowlist, receipt_enabled=_session_kickoff_receipt_enabled(),
        )
        if a in action_subset
    ]
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
    seed_content: str = "",
    post_intent: str = "",
    expected_requester_user_id: str = "",
    activation_content: str = "",
    max_fetch_attempts: int = 3,
    fetch_delay_seconds: float = 0.5,
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
            seed_content=seed_content,
            post_intent=post_intent,
            expected_requester_user_id=expected_requester_user_id,
            activation_content=activation_content,
            max_fetch_attempts=max_fetch_attempts,
            fetch_delay_seconds=fetch_delay_seconds,
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
    "seed_content": "", "post_intent": "", "expected_requester_user_id": "",
    "activation_content": "", "max_fetch_attempts": 3, "fetch_delay_seconds": 0.5,
}


def _make_handler(handler_fn):
    """Create a registry-compatible handler lambda for a discord handler."""
    return lambda args, **kw: handler_fn(
        **{k: args.get(k, v) for k, v in _HANDLER_DEFAULTS.items()},
    )


_STATIC_CORE_SCHEMA = _build_schema(
    [name for name in _CORE_ACTIONS if name != _SESSION_KICKOFF_RECEIPT_ACTION],
    caps={"detected": False}, tool_name="discord",
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
