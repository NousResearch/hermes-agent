"""Slack channel history search tool.

Exposes Slack Web API reads to gateway agents running in Slack. The adapter
already receives live events and can fetch thread context, but agents also need
an explicit tool for "look back in this channel and find the latest X link".
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tools.registry import registry

SLACK_API_BASE = "https://slack.com/api"
_URL_RE = re.compile(r"https?://[^\s<>|)]+")
_SLACK_LINK_RE = re.compile(r"<([^>|]+)(?:\|[^>]+)?>")
_CHANNEL_ID_RE = re.compile(r"^[CGD][A-Z0-9]{8,}$")
_MAX_MESSAGE_TEXT_CHARS = 2000


class SlackAPIError(Exception):
    """Raised when Slack Web API returns a transport or platform error."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Slack API error {status}: {body}")


def _configured_bot_tokens() -> list[str]:
    raw = os.getenv("SLACK_BOT_TOKEN", "").strip()
    if not raw:
        try:
            from gateway.config import Platform, load_gateway_config

            raw = (load_gateway_config().platforms[Platform.SLACK].token or "").strip()
        except Exception:
            raw = ""
    return [token.strip() for token in raw.split(",") if token.strip()]


def _get_bot_token() -> str | None:
    tokens = _configured_bot_tokens()
    if len(tokens) != 1:
        return None
    return tokens[0]


def _token_configuration_error() -> str:
    tokens = _configured_bot_tokens()
    if not tokens:
        return "SLACK_BOT_TOKEN or platforms.slack.token not configured."
    return "Slack history tool requires exactly one bot token; multi-workspace token selection is not supported yet."


def _split_configured_ids(raw: object) -> set[str]:
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    if raw is None:
        return set()
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _configured_allowed_channels() -> set[str]:
    raw: object = os.getenv("SLACK_ALLOWED_CHANNELS", "").strip()
    if not raw:
        try:
            from gateway.config import Platform, load_gateway_config

            config = load_gateway_config()
            raw = os.getenv("SLACK_ALLOWED_CHANNELS", "").strip()
            if not raw:
                raw = config.platforms[Platform.SLACK].extra.get("allowed_channels", "")
        except Exception:
            raw = ""
    return _split_configured_ids(raw)


def _channel_allowed(channel_id: str) -> bool:
    allowed = _configured_allowed_channels()
    if not allowed:
        return True
    # Match live Slack adapter policy: DMs are exempt from allowed_channels.
    if channel_id.startswith("D"):
        return True
    return channel_id in allowed


def _channel_not_allowed_error(channel_id: str) -> str:
    return json.dumps({
        "error": "Slack channel is not in configured allowed_channels.",
        "channel": channel_id,
    })


def _clamp_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        if isinstance(value, bool):
            number = int(value)
        elif isinstance(value, int | float | str):
            number = int(value)
        else:
            number = default
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
        return default
    if isinstance(value, int | float):
        return bool(value)
    return default


def _truncate_text(text: str) -> str:
    if len(text) <= _MAX_MESSAGE_TEXT_CHARS:
        return text
    return text[: _MAX_MESSAGE_TEXT_CHARS - 1] + "…"


def _request(
    method: str,
    endpoint: str,
    token: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    url = f"{SLACK_API_BASE}/{endpoint.lstrip('/')}"
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Hermes-Agent (https://github.com/NousResearch/hermes-agent)",
    }

    if method.upper() == "GET":
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v not in {None, ""}})
    else:
        data = json.dumps(body or {}).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SlackAPIError(exc.code, error_body) from exc

    if not payload.get("ok", False):
        raise SlackAPIError(200, json.dumps(payload))
    return payload


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for link in _SLACK_LINK_RE.findall(text or ""):
        urls.extend(_URL_RE.findall(link))
    urls.extend(_URL_RE.findall(text or ""))
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _message_summary(message: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(message.get("text") or "")
    text = _truncate_text(raw_text)
    return {
        "ts": message.get("ts", ""),
        "thread_ts": message.get("thread_ts", ""),
        "user": message.get("user", ""),
        "bot_id": message.get("bot_id", ""),
        "subtype": message.get("subtype", ""),
        "text": text,
        "text_truncated": len(raw_text) > len(text),
        "urls": _extract_urls(raw_text),
    }


def _resolve_channel(token: str, channel: str) -> str:
    value = (channel or "").strip()
    if not value:
        return ""
    if _CHANNEL_ID_RE.match(value):
        return value

    wanted = value[1:] if value.startswith("#") else value
    cursor = ""
    for _ in range(20):
        payload = _request(
            "GET",
            "conversations.list",
            token,
            params={
                "types": "public_channel,private_channel,im,mpim",
                "exclude_archived": "true",
                "limit": 200,
                "cursor": cursor,
            },
        )
        for channel_obj in payload.get("channels", []):
            if channel_obj.get("name") == wanted:
                return str(channel_obj.get("id", ""))
        cursor = (payload.get("response_metadata") or {}).get("next_cursor") or ""
        if not cursor:
            break
    return value


def _list_channels(token: str, types: str = "public_channel,private_channel", limit: object = 200, cursor: str = "") -> str:
    payload = _request(
        "GET",
        "conversations.list",
        token,
        params={
            "types": types or "public_channel,private_channel",
            "exclude_archived": "true",
            "limit": _clamp_int(limit, 200, 1, 1000),
            "cursor": cursor,
        },
    )
    channels = [
        {
            "id": c.get("id", ""),
            "name": c.get("name", ""),
            "is_channel": c.get("is_channel", False),
            "is_private": c.get("is_private", False),
            "is_im": c.get("is_im", False),
            "is_member": c.get("is_member", False),
        }
        for c in payload.get("channels", [])
        if _channel_allowed(str(c.get("id", "")))
    ]
    allowed_channels = _configured_allowed_channels()
    return json.dumps({
        "channels": channels,
        "count": len(channels),
        "allowed_channels_applied": bool(allowed_channels),
        "next_cursor": (payload.get("response_metadata") or {}).get("next_cursor", ""),
    })


def _fetch_history(
    token: str,
    channel: str,
    limit: object = 50,
    latest: str = "",
    oldest: str = "",
    cursor: str = "",
    inclusive: object = False,
) -> str:
    channel_id = _resolve_channel(token, channel)
    if not channel_id:
        return json.dumps({"error": "Missing channel or channel ID."})
    if not _channel_allowed(channel_id):
        return _channel_not_allowed_error(channel_id)
    payload = _request(
        "GET",
        "conversations.history",
        token,
        params={
            "channel": channel_id,
            "limit": _clamp_int(limit, 50, 1, 200),
            "latest": latest,
            "oldest": oldest,
            "cursor": cursor,
            "inclusive": str(_coerce_bool(inclusive)).lower(),
        },
    )
    messages = [_message_summary(m) for m in payload.get("messages", [])]
    return json.dumps({
        "channel": channel_id,
        "messages": messages,
        "count": len(messages),
        "has_more": payload.get("has_more", False),
        "next_cursor": (payload.get("response_metadata") or {}).get("next_cursor", ""),
    })


def _find_messages(
    token: str,
    channel: str,
    query: str = "",
    link_domains: str = "x.com,twitter.com",
    limit: object = 50,
    max_pages: object = 5,
    latest: str = "",
    oldest: str = "",
) -> str:
    channel_id = _resolve_channel(token, channel)
    if not channel_id:
        return json.dumps({"error": "Missing channel or channel ID."})
    if not _channel_allowed(channel_id):
        return _channel_not_allowed_error(channel_id)

    wanted_domains = tuple(d.strip().lower() for d in (link_domains or "").split(",") if d.strip())
    query_l = (query or "").lower().strip()
    match_limit = _clamp_int(limit, 50, 1, 200)
    page_limit = _clamp_int(max_pages, 5, 1, 20)
    matches: list[dict[str, Any]] = []
    cursor = ""
    pages = 0
    scanned = 0

    while pages < page_limit and len(matches) < match_limit:
        pages += 1
        payload = _request(
            "GET",
            "conversations.history",
            token,
            params={
                "channel": channel_id,
                "limit": 200,
                "latest": latest,
                "oldest": oldest,
                "cursor": cursor,
            },
        )
        for raw in payload.get("messages", []):
            scanned += 1
            raw_text = str(raw.get("text") or "")
            msg = _message_summary(raw)
            text_l = raw_text.lower()
            urls = [
                u for u in msg["urls"]
                if not wanted_domains or urllib.parse.urlparse(u).netloc.lower().removeprefix("www.") in wanted_domains
            ]
            if query_l and query_l not in text_l:
                continue
            if wanted_domains and not urls:
                continue
            if urls:
                msg["urls"] = urls
            matches.append(msg)
            if len(matches) >= match_limit:
                break
        cursor = (payload.get("response_metadata") or {}).get("next_cursor") or ""
        if not cursor:
            break

    return json.dumps({
        "channel": channel_id,
        "matches": matches,
        "count": len(matches),
        "scanned": scanned,
        "pages": pages,
        "next_cursor": cursor,
        "note": "Messages are returned newest-first by Slack conversations.history.",
    })


def slack(
    action: str,
    channel: str = "",
    query: str = "",
    link_domains: str = "x.com,twitter.com",
    types: str = "public_channel,private_channel",
    limit: object = 50,
    max_pages: object = 5,
    latest: str = "",
    oldest: str = "",
    cursor: str = "",
    inclusive: object = False,
) -> str:
    token = _get_bot_token()
    if not token:
        return json.dumps({"error": _token_configuration_error()})

    try:
        if action == "list_channels":
            return _list_channels(token, types=types, limit=limit, cursor=cursor)
        if action == "fetch_history":
            return _fetch_history(token, channel, limit=limit, latest=latest, oldest=oldest, cursor=cursor, inclusive=inclusive)
        if action == "find_messages":
            return _find_messages(token, channel, query=query, link_domains=link_domains, limit=limit, max_pages=max_pages, latest=latest, oldest=oldest)
        return json.dumps({
            "error": f"Unknown action: {action}",
            "available_actions": ["list_channels", "fetch_history", "find_messages"],
        })
    except SlackAPIError as exc:
        return json.dumps({"error": str(exc), "body": exc.body})
    except Exception as exc:
        return json.dumps({"error": f"Unexpected Slack tool error: {exc}"})


def check_slack_tool_requirements() -> bool:
    return bool(_get_bot_token())


_SLACK_SCHEMA = {
    "name": "slack",
    "description": (
        "Read Slack channel history visible to the bot. "
        "Use find_messages to retrieve recent X/Twitter links from channels visible to the configured token. "
        "If Slack allowed_channels is configured, list_channels is filtered and history/search reject other channels. "
        "This read-only tool requires exactly one configured Slack bot token and does not delete or mutate Slack messages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_channels", "fetch_history", "find_messages"],
                "description": "Slack operation to perform.",
            },
            "channel": {
                "type": "string",
                "description": "Slack conversation ID (C..., G..., D...) or #channel name for history/search actions.",
            },
            "query": {
                "type": "string",
                "description": "Optional case-insensitive text filter for find_messages.",
            },
            "link_domains": {
                "type": "string",
                "description": "Comma-separated domains to require in find_messages; default x.com,twitter.com. Empty means no URL-domain filter.",
            },
            "types": {
                "type": "string",
                "description": "Conversation types for list_channels, e.g. public_channel,private_channel,im,mpim.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum channels/messages to return. Channel lists are capped at 1000; history/search results are capped at 200.",
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum 200-message history pages to scan for find_messages, capped at 20.",
            },
            "latest": {
                "type": "string",
                "description": "Optional Slack timestamp upper bound for history/find.",
            },
            "oldest": {
                "type": "string",
                "description": "Optional Slack timestamp lower bound for history/find.",
            },
            "cursor": {
                "type": "string",
                "description": "Pagination cursor for list_channels/fetch_history.",
            },
            "inclusive": {
                "type": "boolean",
                "description": "Whether latest/oldest bounds are inclusive for fetch_history.",
            },
        },
        "required": ["action"],
    },
}


_HANDLER_DEFAULTS = {
    "action": "",
    "channel": "",
    "query": "",
    "link_domains": "x.com,twitter.com",
    "types": "public_channel,private_channel",
    "limit": 50,
    "max_pages": 5,
    "latest": "",
    "oldest": "",
    "cursor": "",
    "inclusive": False,
}


registry.register(
    name="slack",
    toolset="slack",
    schema=_SLACK_SCHEMA,
    handler=lambda args, **_kw: slack(**{k: args.get(k, v) for k, v in _HANDLER_DEFAULTS.items()}),
    check_fn=check_slack_tool_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
)
