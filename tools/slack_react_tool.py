"""
slack_react_tool.py — Add an emoji reaction to a Slack message without sending a text reply.

Useful when the agent wants to acknowledge a message (e.g. 👍, 👎, ✅, 🔍) without
generating any visible response text in the channel.
"""

import json
import logging
from typing import Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_SCHEMA = {
    "name": "slack_react",
    "description": (
        "Add an emoji reaction to a Slack message without sending any text reply. "
        "Use this to silently acknowledge messages — e.g. react with 👍 instead of "
        "typing 'Got it'. Only works when the Slack gateway is configured. "
        "The target defaults to the current conversation message; supply channel and "
        "timestamp explicitly to react to a different message.\n\n"
        "Examples:\n"
        "  Acknowledge an emoji-only message: slack_react(emoji='thumbsup')\n"
        "  React to a specific message: slack_react(emoji='white_check_mark', "
        "channel='C0B3XC4TZ1S', timestamp='1716201336.943429')"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": (
                    "Slack emoji name without colons, e.g. 'thumbsup', 'eyes', "
                    "'white_check_mark', 'x', 'wave', 'rocket'. "
                    "Standard Unicode emoji names work too."
                ),
            },
            "channel": {
                "type": "string",
                "description": (
                    "Slack channel ID (e.g. 'C0B3XC4TZ1S'). "
                    "Defaults to the current gateway session's channel when omitted."
                ),
            },
            "timestamp": {
                "type": "string",
                "description": (
                    "Message timestamp (ts) to react to, e.g. '1716201336.943429'. "
                    "Defaults to the triggering message's timestamp when omitted."
                ),
            },
        },
        "required": ["emoji"],
    },
}


def _get_slack_token() -> Optional[str]:
    """Retrieve the Slack bot token from gateway config."""
    try:
        from gateway.config import load_gateway_config, Platform

        config = load_gateway_config()
        for pconfig in config.platforms:
            if pconfig.platform == Platform.SLACK and pconfig.token:
                return pconfig.token
    except Exception:
        pass
    return None


def _get_current_context() -> tuple[Optional[str], Optional[str]]:
    """Return (channel_id, message_ts) from the active gateway session context, if available."""
    try:
        from gateway.session_context import get_current_session_context

        ctx = get_current_session_context()
        if ctx:
            return ctx.get("chat_id"), ctx.get("message_ts")
    except Exception:
        pass
    return None, None


def slack_react_tool(args: dict, **kwargs) -> str:
    emoji: str = args.get("emoji", "").strip().strip(":")
    if not emoji:
        return json.dumps({"error": "emoji is required"})

    channel: Optional[str] = args.get("channel") or None
    timestamp: Optional[str] = args.get("timestamp") or None

    # Fall back to current session context when channel/ts are not provided
    if not channel or not timestamp:
        ctx_channel, ctx_ts = _get_current_context()
        channel = channel or ctx_channel
        timestamp = timestamp or ctx_ts

    if not channel:
        return json.dumps({"error": "channel could not be determined — pass channel= explicitly"})
    if not timestamp:
        return json.dumps(
            {"error": "message timestamp could not be determined — pass timestamp= explicitly"}
        )

    token = _get_slack_token()
    if not token:
        return json.dumps({"error": "Slack gateway not configured or no bot token found"})

    import asyncio

    async def _add_reaction() -> dict:
        try:
            import aiohttp
        except ImportError:
            return {"error": "aiohttp not installed — run: pip install aiohttp"}
        try:
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

            _proxy = resolve_proxy_url()
            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        except Exception:
            _sess_kw, _req_kw = {}, {}

        url = "https://slack.com/api/reactions.add"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"channel": channel, "timestamp": timestamp, "name": emoji}

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10), **_sess_kw
            ) as session:
                async with session.post(
                    url, headers=headers, json=payload, **_req_kw
                ) as resp:
                    data = await resp.json()
                    if data.get("ok"):
                        return {
                            "success": True,
                            "emoji": emoji,
                            "channel": channel,
                            "timestamp": timestamp,
                        }
                    err = data.get("error", "unknown")
                    # "already_reacted" is not a failure — treat as success
                    if err == "already_reacted":
                        return {
                            "success": True,
                            "emoji": emoji,
                            "note": "already reacted",
                        }
                    return {"error": f"Slack API error: {err}"}
        except Exception as exc:
            return {"error": f"Request failed: {exc}"}

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _add_reaction())
                result = future.result(timeout=15)
        else:
            result = loop.run_until_complete(_add_reaction())
    except Exception as exc:
        result = {"error": f"Async execution failed: {exc}"}

    return json.dumps(result)


def _check_slack_react() -> bool:
    """Tool is available when a Slack gateway config with a token exists."""
    return _get_slack_token() is not None


registry.register(
    name="slack_react",
    toolset="messaging",
    schema=_SCHEMA,
    handler=lambda args, **kw: slack_react_tool(args, **kw),
    check_fn=_check_slack_react,
    emoji="👍",
)
