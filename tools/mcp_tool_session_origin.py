"""Opt-in session-origin injection for MCP tools.

A stdio MCP server cannot see the gateway's session ContextVars, so a tool that opens a durable watch has no way
to record "wake this session later". ``mcp_servers.<name>.session_origin.tools`` lists the tools that receive the
calling session's return address as one reserved argument (default ``_hermesOrigin``). No other tool gets it."""

import logging
from typing import Any, Optional

logger = logging.getLogger("tools.mcp_tool")

DEFAULT_ORIGIN_ARGUMENT = "_hermesOrigin"

# Origin key -> gateway session variable. The single source of truth for what an MCP server is told.
SESSION_ORIGIN_FIELDS = {
    "platform": "HERMES_SESSION_PLATFORM",
    "chatId": "HERMES_SESSION_CHAT_ID",
    "chatType": "HERMES_SESSION_CHAT_TYPE",
    "threadId": "HERMES_SESSION_THREAD_ID",
    "userId": "HERMES_SESSION_USER_ID",
    "userIdAlt": "HERMES_SESSION_USER_ID_ALT",
    "profile": "HERMES_SESSION_PROFILE",
    "scopeId": "HERMES_SESSION_SCOPE_ID",
    "sessionId": "HERMES_SESSION_ID",
    "sessionKey": "HERMES_SESSION_KEY",
}
_REQUIRED_FIELDS = ("platform", "chatId", "chatType")


def parse_session_origin_config(server_name: str, config: Any) -> Optional[tuple[frozenset, str]]:
    """``(tool names, argument name)`` for a server config, or None when the server has not opted in.
    A malformed block opts nothing in and warns, so a typo can never widen what gets injected."""
    block = config.get("session_origin") if isinstance(config, dict) else None
    if block is None:
        return None
    tools = block.get("tools") if isinstance(block, dict) else None
    argument = block.get("argument", DEFAULT_ORIGIN_ARGUMENT) if isinstance(block, dict) else None
    if (not isinstance(tools, list) or not tools or not all(isinstance(t, str) and t.strip() for t in tools)
            or "*" in tools or not isinstance(argument, str) or not argument.strip()):
        logger.warning(
            "MCP server '%s': ignoring session_origin; expected {tools: [explicit tool names], argument: <name>}",
            server_name)
        return None
    return frozenset(t.strip() for t in tools), argument.strip()


def attach_session_origin(server: Any, server_name: str, tool_name: str, args: dict) -> dict:
    """``args`` plus the session origin when this tool opted in and a real chat session is calling; otherwise
    ``args`` itself. A model-supplied value is always overwritten: the origin is a return address the gateway
    vouches for, never something the model may choose. The caller's dict is never mutated."""
    parsed = parse_session_origin_config(server_name, getattr(server, "_config", None))
    if parsed is None or tool_name not in parsed[0]:
        return args
    argument = parsed[1]
    try:
        from gateway.session_context import get_session_env

        origin = {key: get_session_env(var, "") for key, var in SESSION_ORIGIN_FIELDS.items()}
    except Exception:
        logger.debug("Could not read Hermes session origin for MCP tool %s/%s", server_name, tool_name, exc_info=True)
        return args
    if not all(origin[key] for key in _REQUIRED_FIELDS):
        return args
    if argument in args:
        logger.warning("MCP tool %s/%s: overwriting model-supplied '%s' with the gateway session origin",
                       server_name, tool_name, argument)
    return {**args, argument: {key: value for key, value in origin.items() if value}}
