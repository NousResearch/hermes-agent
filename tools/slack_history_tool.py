"""Read-only Slack history tool.

Provides callable Slack history access to agents without exposing raw user names,
full transcripts, secrets, or obvious PII in tool output. The Slack app/token must
already have the relevant history scopes (channels/groups/im/mpim history).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, Optional

from agent.redact import redact_sensitive_text
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_SLACK_CONVERSATION_RE = re.compile(r"^\s*([CGD][A-Z0-9]{2,})\s*$")
_URL_SECRET_QUERY_RE = re.compile(
    r"([?&](?:access_token|api[_-]?key|auth[_-]?token|token|signature|sig)=)([^&#\s]+)",
    re.IGNORECASE,
)
_GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"\b(access_token|api[_-]?key|auth[_-]?token|token|signature|sig|secret)\s*[:=]\s*([^\s,;&]+)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.+-])", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<![\w.])(?:\+?\d[\d\s().-]{7,}\d)(?![\w.])")
_ALLOWED_ACTIONS = {"history", "thread"}
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_MAX_MESSAGE_CHARS = 200

SLACK_HISTORY_SCHEMA = {
    "name": "slack_history",
    "description": (
        "Read recent Slack channel, private channel, DM, MPIM, or thread history via the Slack API. "
        "Use this when the user asks to inspect Slack alerts or prior Slack context. "
        "Requires the Slack bot token to have the matching history scopes. "
        "Output is redacted: actors are hashed, emails/phones/secrets are masked, "
        "and each message text is capped at 200 characters."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["history", "thread"],
                "description": "history = conversations.history; thread = conversations.replies.",
            },
            "channel": {
                "type": "string",
                "description": "Slack conversation ID (C..., G..., D...) or a cached channel name like #alerts.",
            },
            "thread_ts": {
                "type": "string",
                "description": "Thread root timestamp. Required when action='thread'.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_LIMIT,
                "description": "Maximum messages to return. Default 20, hard max 100.",
            },
            "oldest": {
                "type": "string",
                "description": "Optional Slack timestamp lower bound.",
            },
            "latest": {
                "type": "string",
                "description": "Optional Slack timestamp upper bound.",
            },
            "inclusive": {
                "type": "boolean",
                "description": "Whether oldest/latest bounds are inclusive.",
            },
        },
        "required": ["channel"],
    },
}


def check_slack_history_requirements() -> bool:
    """Return True when a Slack bot token and SDK are available."""
    if not _get_slack_token():
        return False
    try:
        from slack_sdk.web.async_client import AsyncWebClient  # noqa: F401
    except Exception:
        return False
    return True


def _get_slack_token() -> str:
    """Load a Slack bot token from env or gateway config without logging it."""
    token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    if not token:
        try:
            from gateway.config import Platform, load_gateway_config

            config = load_gateway_config()
            pconfig = config.platforms.get(Platform.SLACK)
            token = (getattr(pconfig, "token", "") or "").strip() if pconfig else ""
        except Exception:
            token = ""

    # Multi-workspace configs can be comma-separated; use the first token here.
    # A future extension can add explicit team_id → token routing.
    return token.split(",", 1)[0].strip()


def _make_slack_client(token: str):
    from slack_sdk.web.async_client import AsyncWebClient

    proxy = None
    try:
        from gateway.platforms.slack import _resolve_slack_proxy_url

        proxy = _resolve_slack_proxy_url()
    except Exception:
        proxy = None

    kwargs = {"token": token}
    if proxy:
        kwargs["proxy"] = proxy
    return AsyncWebClient(**kwargs)


def _limit_from_args(args: Dict[str, Any]) -> int:
    raw = args.get("limit", _DEFAULT_LIMIT)
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    return max(1, min(limit, _MAX_LIMIT))


def _sanitize_message_text(text: Any, *, max_chars: int = _MAX_MESSAGE_CHARS) -> str:
    """Redact secrets/PII and cap a Slack message snippet."""
    if text is None:
        return ""
    safe = str(text).replace("\x00", "").strip()
    if not safe:
        return ""

    # Mask obvious PII before the generic secret redactor can partially mask
    # phone numbers into a shape that no longer matches the phone regex.
    safe = _EMAIL_RE.sub("[email]", safe)
    safe = _PHONE_RE.sub("[phone]", safe)
    safe = redact_sensitive_text(safe)
    safe = _URL_SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}***", safe)
    safe = _GENERIC_SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=***", safe)
    safe = re.sub(r"\s+", " ", safe).strip()

    if max_chars > 0 and len(safe) > max_chars:
        safe = safe[: max_chars - 1].rstrip() + "…"
    return safe


def _hash_identifier(identifier: str) -> str:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:10]


def _safe_actor_label(message: Dict[str, Any]) -> str:
    """Return a stable, non-PII actor label for a Slack message."""
    bot_id = str(message.get("bot_id") or "").strip()
    is_bot = bool(bot_id) or message.get("subtype") == "bot_message"
    if is_bot:
        identifier = bot_id or str(message.get("app_id") or message.get("username") or "bot")
        return f"bot:{_hash_identifier(identifier)}"

    user_id = str(message.get("user") or "").strip()
    if user_id:
        return f"user:{_hash_identifier(user_id)}"
    return "unknown"


def _resolve_channel(channel: str) -> Optional[str]:
    """Resolve a Slack channel ID or cached human label to a conversation ID."""
    raw = (channel or "").strip()
    if raw.lower().startswith("slack:"):
        raw = raw.split(":", 1)[1].strip()
    if not raw:
        return None

    direct = _SLACK_CONVERSATION_RE.fullmatch(raw)
    if direct:
        return direct.group(1)

    label = raw[1:] if raw.startswith("#") else raw
    try:
        from gateway.channel_directory import resolve_channel_name

        resolved = resolve_channel_name("slack", label)
        if resolved:
            direct = _SLACK_CONVERSATION_RE.fullmatch(str(resolved).strip())
            return direct.group(1) if direct else str(resolved).strip()
    except Exception:
        logger.debug("Slack channel directory lookup failed", exc_info=True)
    return None


def _message_to_result(message: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "ts": str(message.get("ts") or ""),
        "actor": _safe_actor_label(message),
        "kind": str(message.get("subtype") or "message"),
        "text": _sanitize_message_text(message.get("text", "")),
    }
    thread_ts = message.get("thread_ts")
    if thread_ts:
        result["thread_ts"] = str(thread_ts)
    if message.get("reply_count") is not None:
        result["reply_count"] = message.get("reply_count")
    return result


def _format_messages(messages: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [_message_to_result(message) for message in messages if isinstance(message, dict)]


def _api_error(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    error = ""
    needed = ""
    provided = ""
    if hasattr(response, "get"):
        error = str(response.get("error", "") or "")
        needed = str(response.get("needed", "") or "")
        provided = str(response.get("provided", "") or "")
    parts = ["Slack history read failed"]
    if error:
        parts.append(f"error={error}")
    else:
        parts.append(f"error={type(exc).__name__}")
    if needed:
        parts.append(f"needed_scope={needed}")
    if provided:
        parts.append(f"provided_scopes={provided}")
    return "; ".join(parts)


def _request_kwargs(args: Dict[str, Any], channel_id: str, limit: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"channel": channel_id, "limit": limit}
    for key in ("oldest", "latest"):
        value = args.get(key)
        if value:
            kwargs[key] = str(value)
    if "inclusive" in args and args.get("inclusive") is not None:
        kwargs["inclusive"] = bool(args.get("inclusive"))
    return kwargs


async def slack_history_tool(args: Dict[str, Any], **_kw) -> str:
    """Handle read-only Slack history tool calls."""
    action = str(args.get("action") or "history").strip().lower()
    if action not in _ALLOWED_ACTIONS:
        return tool_error(f"Unknown action: {action}")

    channel_id = _resolve_channel(str(args.get("channel") or ""))
    if not channel_id:
        return tool_error("Could not resolve Slack channel. Pass a C/G/D conversation ID or a cached channel name.")

    if action == "thread" and not args.get("thread_ts"):
        return tool_error("thread_ts is required when action='thread'")

    token = _get_slack_token()
    if not token:
        return tool_error("SLACK_BOT_TOKEN is not configured")

    client = _make_slack_client(token)
    limit = _limit_from_args(args)

    try:
        if action == "thread":
            kwargs = _request_kwargs(args, channel_id, limit)
            kwargs["ts"] = str(args.get("thread_ts"))
            response = await client.conversations_replies(**kwargs)
        else:
            kwargs = _request_kwargs(args, channel_id, limit)
            response = await client.conversations_history(**kwargs)
    except Exception as exc:
        return tool_error(_api_error(exc))

    messages = response.get("messages", []) if hasattr(response, "get") else []
    metadata = response.get("response_metadata", {}) if hasattr(response, "get") else {}
    next_cursor = metadata.get("next_cursor") if isinstance(metadata, dict) else ""

    return tool_result(
        success=True,
        action=action,
        channel=channel_id,
        count=len(messages),
        messages=_format_messages(messages),
        has_more=bool(response.get("has_more", False)) if hasattr(response, "get") else False,
        next_cursor=next_cursor or "",
        redaction={
            "actors": "hashed",
            "message_text_max_chars": _MAX_MESSAGE_CHARS,
            "pii": "emails/phones masked",
            "secrets": "masked",
        },
    )


registry.register(
    name="slack_history",
    toolset="slack",
    schema=SLACK_HISTORY_SCHEMA,
    handler=slack_history_tool,
    check_fn=check_slack_history_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    is_async=True,
    emoji="💬",
    max_result_size_chars=50000,
)
