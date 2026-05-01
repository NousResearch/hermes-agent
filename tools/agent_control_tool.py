"""Agent-control tools for orchestrating peer Hermes profiles."""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any

from agent.orchestration import AgentController
from tools.registry import registry, tool_error


_controller: AgentController | None = None


def _get_controller() -> AgentController:
    global _controller
    if _controller is None:
        _controller = AgentController()
    return _controller


def check_agent_control_requirements() -> bool:
    return importlib.util.find_spec("acp") is not None


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


_ADMIN_APPROVAL_POLICY_ENV = "HERMES_AGENT_CONTROL_APPROVAL_POLICY"


def _approval_policy_from_env() -> str:
    """Resolve dangerous-permission policy from process/admin configuration only."""
    value = os.environ.get(_ADMIN_APPROVAL_POLICY_ENV, "deny").strip().lower()
    if value in {"allow_once", "once"}:
        return "allow_once"
    return "deny"


def _handle_start(args: dict, **kw) -> str:
    profile = str(args.get("profile") or "").strip()
    if not profile:
        return tool_error("profile is required.")
    result = _get_controller().start_agent(
        profile=profile,
        cwd=args.get("cwd"),
        session_id=args.get("session_id"),
        idempotency_key=args.get("idempotency_key"),
        approval_policy=_approval_policy_from_env(),
    )
    return _json_result(result)


def _handle_prompt(args: dict, **kw) -> str:
    agent_id = str(args.get("agent_id") or "").strip()
    prompt = str(args.get("prompt") or "")
    if not agent_id:
        return tool_error("agent_id is required.")
    if not prompt.strip():
        return tool_error("prompt is required.")
    result = _get_controller().prompt_agent(
        agent_id=agent_id,
        prompt=prompt,
        timeout_seconds=float(args.get("timeout_seconds") or 600.0),
        lease_wait_seconds=float(args.get("lease_wait_seconds") or 5.0),
        approval_policy=_approval_policy_from_env(),
    )
    return _json_result(result)


def _handle_status(args: dict, **kw) -> str:
    agent_id = str(args.get("agent_id") or "").strip()
    if not agent_id:
        return tool_error("agent_id is required.")
    return _json_result(_get_controller().status(agent_id=agent_id))


def _handle_list(args: dict, **kw) -> str:
    return _json_result(
        _get_controller().list_agents(
            profile=args.get("profile"),
            limit=int(args.get("limit") or 50),
        )
    )


def _handle_fork(args: dict, **kw) -> str:
    agent_id = str(args.get("agent_id") or "").strip()
    if not agent_id:
        return tool_error("agent_id is required.")
    result = _get_controller().fork_agent(
        agent_id=agent_id,
        cwd=args.get("cwd"),
        idempotency_key=args.get("idempotency_key"),
        approval_policy=_approval_policy_from_env(),
    )
    return _json_result(result)


AGENT_START_SCHEMA = {
    "name": "agent_start",
    "description": (
        "Create or attach to a persistent Hermes ACP session for another "
        "profile. This controls a real profile agent with its own config, "
        "memory, skills, session history, and tool surface; it is not an "
        "ephemeral delegate_task subagent. Returns an agent_id handle."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": "Hermes profile name to control, e.g. 'researcher' or 'reviewer'.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the controlled session. Defaults to the current process cwd.",
            },
            "session_id": {
                "type": "string",
                "description": "Existing ACP session id to load instead of creating a new one.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Optional key to safely reuse an existing handle across retries.",
            },
        },
        "required": ["profile"],
    },
}


AGENT_PROMPT_SCHEMA = {
    "name": "agent_prompt",
    "description": (
        "Send a prompt to a controlled profile agent and wait for its reply. "
        "Calls are serialized per agent_id with a durable lease so two "
        "orchestrators do not race the same profile session. Ask for "
        "structured handoffs with files changed, tests run, artifacts, and "
        "blockers when reliability matters."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "Handle returned by agent_start or agent_fork.",
            },
            "prompt": {
                "type": "string",
                "description": "Instruction to send to the controlled profile agent.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Maximum seconds to wait for the controlled agent response. Default 600.",
            },
            "lease_wait_seconds": {
                "type": "number",
                "description": "Seconds to wait if another run currently owns this profile session. Default 5.",
            },
        },
        "required": ["agent_id", "prompt"],
    },
}


AGENT_STATUS_SCHEMA = {
    "name": "agent_status",
    "description": "Return the durable state and latest run for a controlled profile agent.",
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string", "description": "Agent handle id."},
        },
        "required": ["agent_id"],
    },
}


AGENT_LIST_SCHEMA = {
    "name": "agent_list",
    "description": "List controlled profile-agent handles known to this Hermes profile.",
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": "Optional profile filter.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum handles to return, capped at 200. Default 50.",
            },
        },
        "required": [],
    },
}


AGENT_FORK_SCHEMA = {
    "name": "agent_fork",
    "description": (
        "Fork a controlled profile agent's ACP session and return a new "
        "agent_id. Use this for speculative work where the same profile "
        "identity should branch without mutating the original session history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string", "description": "Source agent handle id."},
            "cwd": {
                "type": "string",
                "description": "Working directory for the fork. Defaults to the source handle cwd.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Optional key to safely reuse an existing fork handle across retries.",
            },
        },
        "required": ["agent_id"],
    },
}


registry.register(
    name="agent_start",
    toolset="agent_control",
    schema=AGENT_START_SCHEMA,
    handler=_handle_start,
    check_fn=check_agent_control_requirements,
    emoji="",
)

registry.register(
    name="agent_prompt",
    toolset="agent_control",
    schema=AGENT_PROMPT_SCHEMA,
    handler=_handle_prompt,
    check_fn=check_agent_control_requirements,
    emoji="",
)

registry.register(
    name="agent_status",
    toolset="agent_control",
    schema=AGENT_STATUS_SCHEMA,
    handler=_handle_status,
    check_fn=check_agent_control_requirements,
    emoji="",
)

registry.register(
    name="agent_list",
    toolset="agent_control",
    schema=AGENT_LIST_SCHEMA,
    handler=_handle_list,
    check_fn=check_agent_control_requirements,
    emoji="",
)

registry.register(
    name="agent_fork",
    toolset="agent_control",
    schema=AGENT_FORK_SCHEMA,
    handler=_handle_fork,
    check_fn=check_agent_control_requirements,
    emoji="",
)
