"""Read-only Discord tools backed by the Discord HTTP API v10."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from tools.registry import registry

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"

LIST_CHANNELS_MAX = 100
HISTORY_MAX = 50
SEARCH_MAX = 25
SEARCH_SCAN_MAX = 250
MESSAGE_PAGE_SIZE = 100

GUILD_TEXT_TYPES = {0, 5}
THREAD_TYPES = {10, 11, 12}
DM_TYPES = {1, 3}
READABLE_TYPES = GUILD_TEXT_TYPES | THREAD_TYPES | DM_TYPES


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _coerce_limit(raw: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _parse_csv_env(name: str) -> Set[str]:
    value = os.getenv(name, "")
    return {part.strip() for part in value.split(",") if part.strip()}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_ref(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("\\", "/")
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")
    normalized = normalized.replace(" /", "/").replace("/ ", "/")
    return normalized


@dataclass(frozen=True)
class CurrentDiscordContext:
    chat_id: Optional[str]
    chat_name: Optional[str]
    thread_id: Optional[str]

    @property
    def allowed_ids(self) -> Set[str]:
        return {value for value in (self.chat_id, self.thread_id) if value}


class DiscordReadError(RuntimeError):
    """Raised when a Discord read operation cannot be completed safely."""


def _get_current_discord_context() -> CurrentDiscordContext:
    platform = os.getenv("HERMES_SESSION_PLATFORM", "").strip().lower()
    if platform != "discord":
        return CurrentDiscordContext(chat_id=None, chat_name=None, thread_id=None)
    return CurrentDiscordContext(
        chat_id=os.getenv("HERMES_SESSION_CHAT_ID", "").strip() or None,
        chat_name=os.getenv("HERMES_SESSION_CHAT_NAME", "").strip() or None,
        thread_id=os.getenv("HERMES_SESSION_THREAD_ID", "").strip() or None,
    )


def _load_discord_token() -> Optional[str]:
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if token:
        return token

    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        platform_cfg = config.platforms.get(Platform.DISCORD)
        if platform_cfg and platform_cfg.enabled and platform_cfg.token:
            return platform_cfg.token.strip()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Discord read tool could not load gateway config: %s", exc)

    return None


def _check_discord_read_requirements() -> bool:
    return bool(_load_discord_token())


def _load_dm_session_candidates() -> List[Dict[str, Any]]:
    try:
        from gateway.channel_directory import _build_from_sessions

        entries = _build_from_sessions("discord")
        return [entry for entry in entries if entry.get("type") == "dm"]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Discord read tool could not load DM session candidates: %s", exc)
        return []


def _format_channel_name(
    *,
    kind: str,
    name: str,
    guild_name: Optional[str],
    parent_name: Optional[str],
) -> str:
    if kind == "dm":
        return name
    if kind == "thread":
        if guild_name and parent_name:
            return f"{guild_name} / #{parent_name} / {name}"
        if parent_name:
            return f"{parent_name} / {name}"
        return name
    if guild_name:
        return f"{guild_name} / #{name}"
    return f"#{name}"


def _channel_aliases(entry: Dict[str, Any]) -> Set[str]:
    aliases = {
        _normalize_ref(entry["id"]),
        _normalize_ref(entry["qualified_name"]),
    }

    name = entry.get("name")
    if name:
        aliases.add(_normalize_ref(name))
        if entry.get("type") == "channel":
            aliases.add(_normalize_ref(f"#{name}"))

    guild_name = entry.get("guild_name")
    parent_name = entry.get("parent_name")
    if guild_name and entry.get("type") == "channel" and name:
        aliases.add(_normalize_ref(f"{guild_name}/{name}"))
        aliases.add(_normalize_ref(f"{guild_name}/#{name}"))
    if entry.get("type") == "thread" and name:
        if parent_name:
            aliases.add(_normalize_ref(f"{parent_name}/{name}"))
            aliases.add(_normalize_ref(f"#{parent_name}/{name}"))
        if guild_name and parent_name:
            aliases.add(_normalize_ref(f"{guild_name}/{parent_name}/{name}"))
            aliases.add(_normalize_ref(f"{guild_name}/#{parent_name}/{name}"))

    return {alias for alias in aliases if alias}


def _truncate_text(value: str, *, limit: int = 500) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _message_permalink(message: Dict[str, Any], channel: Dict[str, Any]) -> Optional[str]:
    message_id = str(message.get("id") or "").strip()
    channel_id = str(channel.get("id") or "").strip()
    if not message_id or not channel_id:
        return None

    guild_id = channel.get("guild_id")
    if guild_id:
        return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
    if channel.get("type") == "dm":
        return f"https://discord.com/channels/@me/{channel_id}/{message_id}"
    return None


def _format_message(message: Dict[str, Any], channel: Dict[str, Any]) -> Dict[str, Any]:
    author = message.get("author") or {}
    content = (message.get("content") or "").strip()
    attachments = [
        {
            "filename": attachment.get("filename"),
            "url": attachment.get("url"),
            "content_type": attachment.get("content_type"),
        }
        for attachment in message.get("attachments", [])
        if attachment.get("filename") or attachment.get("url")
    ]
    if attachments:
        attachment_names = ", ".join(
            attachment["filename"] or "attachment" for attachment in attachments
        )
        if content:
            content = f"{content}\nAttachments: {attachment_names}"
        else:
            content = f"[Attachment only] {attachment_names}"

    return {
        "id": str(message.get("id")),
        "timestamp": message.get("timestamp"),
        "author_id": str(author.get("id")) if author.get("id") is not None else None,
        "author_name": author.get("global_name") or author.get("username"),
        "content": _truncate_text(content),
        "permalink": _message_permalink(message, channel),
        "attachments": attachments,
    }


async def _request_json(
    client: Any,
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    import httpx

    url = f"{DISCORD_API_BASE}{path}"
    try:
        response = await client.request(method, url, params=params)
    except httpx.HTTPError as exc:
        raise DiscordReadError(f"Discord API request failed: {exc}") from exc

    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        raise DiscordReadError(f"Discord API error ({response.status_code}) for {path}: {body}")

    try:
        return response.json()
    except ValueError as exc:
        raise DiscordReadError(f"Discord API returned invalid JSON for {path}") from exc


async def _fetch_guild_name(
    client: Any,
    guild_id: Optional[str],
    guild_names: Dict[str, Optional[str]],
) -> Optional[str]:
    if not guild_id:
        return None
    if guild_id in guild_names:
        return guild_names[guild_id]

    data = await _request_json(client, "GET", f"/guilds/{guild_id}")
    guild_name = data.get("name") or guild_id
    guild_names[guild_id] = guild_name
    return guild_name


async def _build_channel_entry(
    client: Any,
    channel_data: Dict[str, Any],
    guild_names: Dict[str, Optional[str]],
    parent_cache: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    channel_type = channel_data.get("type")
    channel_id = str(channel_data.get("id") or "").strip()
    if not channel_id or channel_type not in READABLE_TYPES:
        return None

    guild_id = (
        str(channel_data.get("guild_id")) if channel_data.get("guild_id") is not None else None
    )
    parent_id = (
        str(channel_data.get("parent_id")) if channel_data.get("parent_id") is not None else None
    )
    guild_name = await _fetch_guild_name(client, guild_id, guild_names) if guild_id else None

    parent_name = None
    if parent_id:
        parent = parent_cache.get(parent_id)
        if parent is None:
            try:
                parent = await _request_json(client, "GET", f"/channels/{parent_id}")
                parent_cache[parent_id] = parent
            except DiscordReadError:
                parent = None
        if parent is not None:
            parent_name = parent.get("name")

    if channel_type in DM_TYPES:
        recipients = channel_data.get("recipients") or []
        recipient = recipients[0] if recipients else {}
        name = (
            recipient.get("global_name")
            or recipient.get("username")
            or recipient.get("id")
            or channel_id
        )
        kind = "dm"
    else:
        name = channel_data.get("name") or channel_id
        kind = "thread" if channel_type in THREAD_TYPES else "channel"

    qualified_name = _format_channel_name(
        kind=kind,
        name=name,
        guild_name=guild_name,
        parent_name=parent_name,
    )

    return {
        "id": channel_id,
        "name": name,
        "qualified_name": qualified_name,
        "type": kind,
        "discord_type": channel_type,
        "guild_id": guild_id,
        "guild_name": guild_name,
        "parent_id": parent_id,
        "parent_name": parent_name,
    }


async def _fetch_channel(
    client: Any,
    channel_id: str,
    guild_names: Dict[str, Optional[str]],
    parent_cache: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    data = await _request_json(client, "GET", f"/channels/{channel_id}")
    return await _build_channel_entry(client, data, guild_names, parent_cache)


async def _discover_guild_scope(
    client: Any,
    guild_id: str,
    *,
    guild_names: Dict[str, Optional[str]],
    parent_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    guild_name = await _fetch_guild_name(client, guild_id, guild_names)
    channel_payloads = await _request_json(client, "GET", f"/guilds/{guild_id}/channels")

    for payload in channel_payloads:
        entry = await _build_channel_entry(client, payload, guild_names, parent_cache)
        if entry is None:
            continue
        entry["guild_name"] = guild_name
        parent_cache.setdefault(entry["id"], payload)
        entries[entry["id"]] = entry

    try:
        active_threads = await _request_json(client, "GET", f"/guilds/{guild_id}/threads/active")
    except DiscordReadError:
        active_threads = {}

    for payload in active_threads.get("threads", []) or []:
        entry = await _build_channel_entry(client, payload, guild_names, parent_cache)
        if entry is None:
            continue
        entry["guild_name"] = guild_name
        entries[entry["id"]] = entry

    return entries


async def _discover_accessible_channels() -> List[Dict[str, Any]]:
    import httpx

    token = _load_discord_token()
    if not token:
        raise DiscordReadError("DISCORD_BOT_TOKEN is not configured.")

    allowed_guilds = _parse_csv_env("DISCORD_READ_ALLOWED_GUILDS")
    allowed_channels = _parse_csv_env("DISCORD_READ_ALLOWED_CHANNELS")
    include_dms = _env_bool("DISCORD_READ_INCLUDE_DMS", default=False)
    current = _get_current_discord_context()

    dm_candidates: Set[str] = set()
    if include_dms:
        for entry in _load_dm_session_candidates():
            channel_id = str(entry.get("id") or "").split(":", 1)[0].strip()
            if channel_id:
                dm_candidates.add(channel_id)
    if current.chat_id:
        dm_candidates.add(current.chat_id)

    allowed_direct_ids = set(allowed_channels) | current.allowed_ids

    if not allowed_guilds and not allowed_channels and not current.allowed_ids:
        raise DiscordReadError(
            "Discord read scope is empty. Set DISCORD_READ_ALLOWED_GUILDS or "
            "DISCORD_READ_ALLOWED_CHANNELS, or use the tool from an active Discord session."
        )

    headers = {"Authorization": f"Bot {token}"}
    timeout = httpx.Timeout(30.0)

    entries: Dict[str, Dict[str, Any]] = {}
    guild_names: Dict[str, Optional[str]] = {}
    parent_cache: Dict[str, Dict[str, Any]] = {}
    guilds_with_thread_scan: Set[str] = set()

    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        for guild_id in sorted(allowed_guilds):
            for channel_id, entry in (
                await _discover_guild_scope(
                    client,
                    guild_id,
                    guild_names=guild_names,
                    parent_cache=parent_cache,
                )
            ).items():
                entries[channel_id] = entry
            guilds_with_thread_scan.add(guild_id)

        direct_ids_to_fetch = set(allowed_direct_ids)
        if include_dms or (current.chat_id and current.chat_id in current.allowed_ids):
            direct_ids_to_fetch.update(dm_candidates)

        channel_entries_from_ids: List[Dict[str, Any]] = []
        for channel_id in sorted(direct_ids_to_fetch):
            try:
                entry = await _fetch_channel(client, channel_id, guild_names, parent_cache)
            except DiscordReadError as exc:
                logger.debug("Discord read tool could not fetch channel %s: %s", channel_id, exc)
                continue
            if entry is None:
                continue
            channel_entries_from_ids.append(entry)
            entries[entry["id"]] = entry

        for entry in channel_entries_from_ids:
            guild_id = entry.get("guild_id")
            if not guild_id or guild_id in guilds_with_thread_scan:
                continue
            if entry["id"] not in allowed_channels:
                continue
            try:
                active_threads = await _request_json(client, "GET", f"/guilds/{guild_id}/threads/active")
            except DiscordReadError:
                continue
            guilds_with_thread_scan.add(guild_id)
            for payload in active_threads.get("threads", []) or []:
                if str(payload.get("parent_id")) != entry["id"]:
                    continue
                thread_entry = await _build_channel_entry(client, payload, guild_names, parent_cache)
                if thread_entry is None:
                    continue
                entries[thread_entry["id"]] = thread_entry

    accessible: List[Dict[str, Any]] = []
    current_ids = current.allowed_ids
    for entry in entries.values():
        channel_id = entry["id"]
        guild_id = entry.get("guild_id")
        parent_id = entry.get("parent_id")
        is_dm = entry["type"] == "dm"

        if channel_id in current_ids:
            allow_reason = "current_session"
        elif guild_id and guild_id in allowed_guilds:
            allow_reason = "allowed_guild"
        elif channel_id in allowed_channels:
            allow_reason = "allowed_channel"
        elif parent_id and parent_id in allowed_channels and entry["type"] == "thread":
            allow_reason = "allowed_parent_channel"
        elif is_dm and include_dms:
            allow_reason = "allowed_dm"
        else:
            continue

        accessible.append(
            {
                **entry,
                "allow_reason": allow_reason,
                "is_current": channel_id in current_ids,
            }
        )

    accessible.sort(
        key=lambda entry: (
            0 if entry.get("is_current") else 1,
            entry.get("guild_name") or "",
            entry.get("parent_name") or "",
            entry.get("name") or "",
            entry["id"],
        )
    )
    return accessible


def _resolve_channel_ref(
    ref: Optional[str],
    channels: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    current = _get_current_discord_context()
    if not ref:
        if not current.chat_id:
            raise DiscordReadError(
                "No Discord channel specified and there is no active Discord session to infer one from."
            )
        ref = current.chat_id

    query = _normalize_ref(ref)
    if not query:
        raise DiscordReadError("Channel reference cannot be empty.")

    channels = list(channels)
    by_id = {entry["id"]: entry for entry in channels}
    if query in by_id:
        return by_id[query]

    matches = [entry for entry in channels if query in _channel_aliases(entry)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        candidates = ", ".join(entry["qualified_name"] for entry in matches[:5])
        raise DiscordReadError(f"Channel reference '{ref}' is ambiguous. Use one of: {candidates}")

    raise DiscordReadError(
        f"Channel reference '{ref}' is not accessible. Use discord_list_channels to inspect the scoped channels."
    )


async def _fetch_recent_messages(
    channel_id: str,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    import httpx

    token = _load_discord_token()
    if not token:
        raise DiscordReadError("DISCORD_BOT_TOKEN is not configured.")

    headers = {"Authorization": f"Bot {token}"}
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        return await _request_json(
            client,
            "GET",
            f"/channels/{channel_id}/messages",
            params={"limit": limit},
        )


async def _search_recent_messages(
    channel_id: str,
    *,
    query: str,
    result_limit: int,
    scan_limit: int,
) -> Dict[str, Any]:
    import httpx

    token = _load_discord_token()
    if not token:
        raise DiscordReadError("DISCORD_BOT_TOKEN is not configured.")

    headers = {"Authorization": f"Bot {token}"}
    timeout = httpx.Timeout(30.0)
    normalized_query = query.lower()
    matches: List[Dict[str, Any]] = []
    scanned = 0
    before: Optional[str] = None

    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        while scanned < scan_limit and len(matches) < result_limit:
            batch_size = min(MESSAGE_PAGE_SIZE, scan_limit - scanned)
            params: Dict[str, Any] = {"limit": batch_size}
            if before:
                params["before"] = before
            page = await _request_json(
                client,
                "GET",
                f"/channels/{channel_id}/messages",
                params=params,
            )
            if not page:
                break

            scanned += len(page)
            for message in page:
                author = message.get("author") or {}
                haystack_parts = [
                    message.get("content") or "",
                    author.get("global_name") or "",
                    author.get("username") or "",
                ]
                for attachment in message.get("attachments", []) or []:
                    if attachment.get("filename"):
                        haystack_parts.append(attachment["filename"])
                haystack = "\n".join(haystack_parts).lower()
                if normalized_query in haystack:
                    matches.append(message)
                    if len(matches) >= result_limit:
                        break

            before = str(page[-1].get("id")) if page else None
            if len(page) < batch_size or not before:
                break

    return {"matches": matches, "scanned": scanned}


async def _discord_list_channels_impl(args: Dict[str, Any], **_kwargs) -> str:
    limit = _coerce_limit(args.get("limit"), default=25, minimum=1, maximum=LIST_CHANNELS_MAX)
    query = str(args.get("query") or "").strip().lower()

    channels = await _discover_accessible_channels()
    if query:
        channels = [
            entry
            for entry in channels
            if query in entry["qualified_name"].lower() or query in (entry.get("name") or "").lower()
        ]

    visible = channels[:limit]
    return _json(
        {
            "channels": [
                {
                    "id": entry["id"],
                    "name": entry["name"],
                    "qualified_name": entry["qualified_name"],
                    "type": entry["type"],
                    "guild_id": entry.get("guild_id"),
                    "guild_name": entry.get("guild_name"),
                    "parent_id": entry.get("parent_id"),
                    "parent_name": entry.get("parent_name"),
                    "is_current": entry.get("is_current", False),
                    "allow_reason": entry.get("allow_reason"),
                }
                for entry in visible
            ],
            "returned": len(visible),
            "total_accessible": len(channels),
            "limit": limit,
        }
    )


async def _discord_read_history_impl(args: Dict[str, Any], **_kwargs) -> str:
    limit = _coerce_limit(args.get("limit"), default=20, minimum=1, maximum=HISTORY_MAX)
    channels = await _discover_accessible_channels()
    channel = _resolve_channel_ref(args.get("channel"), channels)
    messages = await _fetch_recent_messages(channel["id"], limit=limit)

    return _json(
        {
            "channel": {
                "id": channel["id"],
                "qualified_name": channel["qualified_name"],
                "type": channel["type"],
                "guild_id": channel.get("guild_id"),
                "guild_name": channel.get("guild_name"),
            },
            "messages": [_format_message(message, channel) for message in messages[:limit]],
            "returned": min(len(messages), limit),
            "limit": limit,
        }
    )


async def _discord_search_messages_impl(args: Dict[str, Any], **_kwargs) -> str:
    query = str(args.get("query") or "").strip()
    if len(query) < 2:
        return _json({"error": "Query must be at least 2 characters long."})

    result_limit = _coerce_limit(args.get("limit"), default=10, minimum=1, maximum=SEARCH_MAX)
    scan_limit = _coerce_limit(
        args.get("scan_limit"),
        default=SEARCH_SCAN_MAX,
        minimum=1,
        maximum=SEARCH_SCAN_MAX,
    )

    channels = await _discover_accessible_channels()
    channel = _resolve_channel_ref(args.get("channel"), channels)
    search_result = await _search_recent_messages(
        channel["id"],
        query=query,
        result_limit=result_limit,
        scan_limit=scan_limit,
    )

    return _json(
        {
            "channel": {
                "id": channel["id"],
                "qualified_name": channel["qualified_name"],
                "type": channel["type"],
                "guild_id": channel.get("guild_id"),
                "guild_name": channel.get("guild_name"),
            },
            "query": query,
            "matches": [_format_message(message, channel) for message in search_result["matches"]],
            "returned": len(search_result["matches"]),
            "limit": result_limit,
            "scanned_messages": search_result["scanned"],
            "scan_limit": scan_limit,
        }
    )


async def _wrap_tool(handler, args: Dict[str, Any]) -> str:
    try:
        return await handler(args)
    except DiscordReadError as exc:
        return _json({"error": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Discord read tool failed: %s", exc)
        return _json({"error": f"Discord read tool failed: {type(exc).__name__}: {exc}"})


DISCORD_LIST_CHANNELS_SCHEMA = {
    "name": "discord_list_channels",
    "description": (
        "List the Discord channels and threads that are readable within the configured allowlist scope. "
        "Results are bounded and include the current Discord session target when one is active."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional case-insensitive substring filter applied to readable channel names.",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of channels to return (1-{LIST_CHANNELS_MAX}).",
            },
        },
        "required": [],
    },
}


DISCORD_READ_HISTORY_SCHEMA = {
    "name": "discord_read_history",
    "description": (
        "Read the most recent messages from one readable Discord channel or thread. "
        "If no channel is provided, the current Discord session target is used."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Readable Discord channel reference or numeric channel ID within the configured scope.",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of recent messages to return (1-{HISTORY_MAX}).",
            },
        },
        "required": [],
    },
}


DISCORD_SEARCH_MESSAGES_SCHEMA = {
    "name": "discord_search_messages",
    "description": (
        "Search a bounded recent window of messages in one readable Discord channel or thread. "
        "If no channel is provided, the current Discord session target is used."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Case-insensitive text query to search for.",
            },
            "channel": {
                "type": "string",
                "description": "Readable Discord channel reference or numeric channel ID within the configured scope.",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of matching messages to return (1-{SEARCH_MAX}).",
            },
            "scan_limit": {
                "type": "integer",
                "description": (
                    f"Maximum number of recent messages to scan while searching (1-{SEARCH_SCAN_MAX}). "
                    "Higher values increase coverage but stay within a hard safety bound."
                ),
            },
        },
        "required": ["query"],
    },
}


registry.register(
    name="discord_list_channels",
    toolset="discord_read",
    schema=DISCORD_LIST_CHANNELS_SCHEMA,
    handler=lambda args, **kwargs: _wrap_tool(_discord_list_channels_impl, args),
    check_fn=_check_discord_read_requirements,
    is_async=True,
    emoji="💬",
)

registry.register(
    name="discord_read_history",
    toolset="discord_read",
    schema=DISCORD_READ_HISTORY_SCHEMA,
    handler=lambda args, **kwargs: _wrap_tool(_discord_read_history_impl, args),
    check_fn=_check_discord_read_requirements,
    is_async=True,
    emoji="📜",
)

registry.register(
    name="discord_search_messages",
    toolset="discord_read",
    schema=DISCORD_SEARCH_MESSAGES_SCHEMA,
    handler=lambda args, **kwargs: _wrap_tool(_discord_search_messages_impl, args),
    check_fn=_check_discord_read_requirements,
    is_async=True,
    emoji="🔎",
)
