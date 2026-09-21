"""config_set tool — lets the agent switch modes via gateway RPC.

When the agent calls config_set(key="mode", value="auto"), it reaches
the gateway's config.set handler which updates the session state,
persists to DB, and emits session.info so the TUI badge updates.

This closes the gap where mode prompts instructed the agent to call
config.set but no tool existed for it.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from tools.registry import registry, tool_error


CONFIG_SET_SCHEMA = {
    "name": "config_set",
    "description": (
        "Change a runtime configuration value via the gateway. "
        "Currently supports key='mode' to switch agent mode "
        "(auto, plan, gods_plan, recon). The gateway updates the "
        "session state, persists to DB, and notifies the TUI."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The config key to set (e.g. 'mode').",
            },
            "value": {
                "type": "string",
                "description": "The value to set (e.g. 'auto', 'plan', 'gods_plan', 'recon').",
            },
        },
        "required": ["key", "value"],
    },
}


def config_set_tool(*, key: str, value: str, **kwargs: Any) -> str:
    """Execute a config.set RPC call via the gateway.

    The gateway handler does the heavy lifting: validates the mode,
    updates the in-memory session dict, sets ephemeral_system_prompt,
    persists to DB, and emits session.info for TUI sync.
    """
    if not key:
        return json.dumps({"error": "key is required"})
    if not value:
        return json.dumps({"error": "value is required"})

    local_result: Dict[str, Any] | None = None
    if key == "mode":
        from hermes_cli.mode_prompts import get_mode_prompt, validate_mode

        try:
            mode = validate_mode(value)
        except ValueError as exc:
            return tool_error(str(exc))
        agent = kwargs.get("agent")
        if agent is not None:
            previous = getattr(agent, "agent_mode", "auto") or "auto"
            agent.agent_mode = mode
            agent.ephemeral_system_prompt = get_mode_prompt(mode)
            local_result = {
                "key": "mode",
                "value": mode,
                "prompt_cache_reset": previous != mode,
            }
        value = mode

    # Import here to avoid circular imports at module load time
    from tui_gateway.server import handle_request

    session_id = kwargs.get("session_id", "")
    result = handle_request({
        "method": "config.set",
        "params": {"key": key, "value": value, "session_id": session_id},
    })
    if local_result is not None:
        if not isinstance(result, dict):
            result = {}
        result.setdefault("result", {}).update(local_result)
    return json.dumps(result or {}, ensure_ascii=False)


def check_config_set_requirements() -> str | None:
    """Always available — no special requirements."""
    return None


# --- Registry ---

registry.register(
    name="config_set",
    toolset="terminal",
    schema=CONFIG_SET_SCHEMA,
    handler=lambda args, **kw: config_set_tool(
        key=args.get("key", ""),
        value=args.get("value", ""),
        session_id=kw.get("session_id", ""),
    ),
    check_fn=check_config_set_requirements,
    emoji="⚙️",
)
