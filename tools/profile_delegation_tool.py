"""Model-facing cross-profile delegation tool."""

from __future__ import annotations

import json
import os
from typing import Optional

from hermes_cli.delegation_policy import ToolActionRequest
from hermes_cli.executive_bus_gate import executive_bus_enabled_for_current_context
from hermes_cli.profile_delegation import ProfileDelegationRequest, delegate_to_profile
from tools.registry import registry


def _current_session_key() -> Optional[str]:
    try:
        from tools.approval import get_current_session_key
        return get_current_session_key()
    except Exception:
        return os.environ.get("HERMES_GATEWAY_SESSION_KEY")


def delegate_to_profile_tool(
    task: str,
    required_capability: str,
    profile: Optional[str] = None,
    risk: str = "READ",
    return_to: str = "current_session",
    timeout_seconds: int = 300,
    max_runtime_seconds: int = 300,
    board: Optional[str] = None,
    tool_name: Optional[str] = None,
    action_name: Optional[str] = None,
    operation: Optional[str] = None,
    max_concurrency: Optional[int] = None,
) -> str:
    requester_profile = os.environ.get("HERMES_PROFILE") or "default"
    tool_action = None
    if tool_name or action_name or operation:
        tool_action = ToolActionRequest(
            capability=required_capability,
            tool_name=tool_name,
            action_name=action_name,
            operation=operation,
        )
    req = ProfileDelegationRequest(
        profile=profile,
        task=task,
        required_capability=required_capability,
        risk=risk,
        requester_profile=requester_profile,
        requester_session_key=_current_session_key(),
        requester_session_id=os.environ.get("HERMES_SESSION_ID"),
        return_to=return_to,
        timeout_seconds=int(timeout_seconds),
        max_runtime_seconds=int(max_runtime_seconds),
        board=board,
        tool_action=tool_action,
        max_concurrency=max_concurrency,
    )
    result = delegate_to_profile(req)
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


registry.register(
    name="delegate_to_profile",
    toolset="executive_bus",
    schema={
        "name": "delegate_to_profile",
        "description": (
            "Delegate a bounded subtask to another Hermes profile, executed under that "
            "profile's own HERMES_HOME/config/tools/credentials. If profile is omitted, "
            "the executor is selected automatically using capability and workload ranking. "
            "Uses Kanban internally; credentials are never exposed to the requester."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile": {"type": "string", "description": "Optional executor profile, e.g. cto. Omit for auto-selection."},
                "task": {"type": "string", "description": "Exact bounded subtask for the executor."},
                "required_capability": {"type": "string", "description": "Capability required, e.g. mcp:vercel."},
                "risk": {"type": "string", "enum": ["READ", "PREPARE", "CONSEQUENTIAL_WRITE"], "default": "READ"},
                "return_to": {"type": "string", "enum": ["current_session"], "default": "current_session"},
                "timeout_seconds": {"type": "integer", "default": 300, "minimum": 0, "maximum": 900},
                "max_runtime_seconds": {"type": "integer", "default": 300, "minimum": 5, "maximum": 1800},
                "board": {"type": "string", "description": "Optional Kanban board slug."},
                "tool_name": {"type": "string", "description": "Future policy atom: concrete tool name, if known."},
                "action_name": {"type": "string", "description": "Future policy atom: concrete tool action, if known."},
                "operation": {"type": "string", "description": "Future policy atom: read/write/delete/etc., if known."},
                "max_concurrency": {"type": "integer", "description": "Optional simple per-profile concurrency cap for ranking/dispatch."},
            },
            "required": ["task", "required_capability"],
        },
    },
    handler=lambda args, **kw: delegate_to_profile_tool(
        profile=args.get("profile"),
        task=args.get("task", ""),
        required_capability=args.get("required_capability", ""),
        risk=args.get("risk", "READ"),
        return_to=args.get("return_to", "current_session"),
        timeout_seconds=args.get("timeout_seconds", 300),
        max_runtime_seconds=args.get("max_runtime_seconds", 300),
        board=args.get("board"),
        tool_name=args.get("tool_name"),
        action_name=args.get("action_name"),
        operation=args.get("operation"),
        max_concurrency=args.get("max_concurrency"),
    ),
    check_fn=executive_bus_enabled_for_current_context,
)
