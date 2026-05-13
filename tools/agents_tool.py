#!/usr/bin/env python3
"""
Agents Tool Module — Native Agent Registry Tools

Provides tools for discovering and delegating to named agents:
  - agents_list: discover available named agents (read-only)
  - agent_view: inspect a single agent's full definition including prompt
  - assign_agent: delegate a task to a named agent via delegate_task

Named agents are defined in:
  - Global: $HERMES_HOME/agents/*.md and $HERMES_HOME/agents/**/AGENT.md
  - Project: <project>/.hermes/agents/*.md and <project>/.hermes/agents/**/AGENT.md
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Literal, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependency at module level
_AgentDefinition = None


def _get_agent_definition():
    global _AgentDefinition
    if _AgentDefinition is None:
        from agent.agent_registry import AgentDefinition
        _AgentDefinition = AgentDefinition
    return _AgentDefinition


# ─────────────────────────────────────────────────────────────────────────────
# agents_list
# ─────────────────────────────────────────────────────────────────────────────

def agents_list(
    category: Optional[str] = None,
    include_disabled: bool = False,
    include_shadowed: bool = False,
    workdir: Optional[str] = None,
) -> str:
    """
    List all discovered named agents with compact metadata (no prompts).

    Agents are discovered from:
      - Global: $HERMES_HOME/agents/
      - Project: <project>/.hermes/agents/ (when workdir is inside a project)

    Args:
        category: Optional tag to filter agents by (e.g. "web", "code").
                  If provided, only agents with this tag are returned.
                  If no agent matches, an empty list is returned.
        include_disabled: Include agents with enabled=False (default: False).
        include_shadowed: Include shadowed agents hidden by higher-priority
                          entries of the same name (default: False).
        workdir: Optional working directory used to resolve the project root
                 for project-local agent discovery.

    Returns:
        JSON string with success, count, agents (list_summary), and hint.
    """
    try:
        from agent.agent_registry import list_agents

        agents = list_agents(
            workdir=workdir,
            include_disabled=include_disabled,
            include_shadowed=include_shadowed,
        )

        # Filter by category tag if provided
        if category:
            agents = [a for a in agents if category in a.tags]

        summaries = [a.list_summary() for a in agents]

        return json.dumps(
            {
                "success": True,
                "count": len(summaries),
                "agents": summaries,
                "hint": "Use agent_view(name='...') to see full details including the prompt.",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.warning("agents_list failed: %s", exc)
        return json.dumps({"success": False, "error": str(exc), "count": 0, "agents": []})


# ─────────────────────────────────────────────────────────────────────────────
# agent_view
# ─────────────────────────────────────────────────────────────────────────────

def agent_view(
    name: str,
    source: Optional[Literal["global", "project"]] = None,
    workdir: Optional[str] = None,
) -> str:
    """
    Get the full definition of a single named agent, including its prompt.

    Args:
        name: The agent's unique name (lowercase alphanumeric + hyphens).
        source: Optional source filter — "global" or "project".
                If omitted, returns the effective (non-shadowed) agent.
        workdir: Optional working directory for project-local agent discovery.

    Returns:
        JSON string with success and full agent.to_dict() on success,
        or success=False with error on failure.
    """
    try:
        from agent.agent_registry import get_agent

        agent = get_agent(name=name, workdir=workdir, source=source)
        if agent is None:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Agent '{name}' not found.",
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "success": True,
                "agent": agent.to_dict(),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.warning("agent_view(%s) failed: %s", name, exc)
        return json.dumps({"success": False, "error": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Workdir helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_workdir(workdir: str | None, parent_agent: Any | None) -> str | None:
    """Resolve effective workdir: explicit > TERMINAL_CWD > parent_agent hints."""
    if workdir:
        return workdir

    # Check TERMINAL_CWD env var
    env_cwd = os.environ.get("TERMINAL_CWD")
    if env_cwd:
        return env_cwd

    # Best-effort from parent_agent attributes
    if parent_agent is not None:
        # _subdirectory_hints is a common pattern in AIAgent
        hints = getattr(parent_agent, "_subdirectory_hints", None)
        if hints is not None:
            wd = getattr(hints, "working_dir", None)
            if wd:
                return wd
        # terminal_cwd attribute
        wd = getattr(parent_agent, "terminal_cwd", None)
        if wd:
            return wd
        # plain cwd attribute
        wd = getattr(parent_agent, "cwd", None)
        if wd:
            return wd

    return None


def _parent_session_id(parent_agent: Any | None) -> str:
    """Best-effort stable session id for runtime trace correlation."""
    if parent_agent is not None:
        for attr in ("session_id", "task_id"):
            value = getattr(parent_agent, attr, None)
            if isinstance(value, str) and value:
                return value
    return os.environ.get("HERMES_SESSION_ID") or "default"


def _emit_runtime_trace(event: str, *, parent_session_id: str, task_id: str = "", **data: Any) -> None:
    try:
        from agent.runtime_trace import emit_runtime_event

        emit_runtime_event(event, session_id=parent_session_id, task_id=task_id, **data)
    except Exception:
        pass


def _agent_trace_payload(agent: Any) -> dict[str, Any]:
    return {
        "name": agent.name,
        "source": agent.source,
        "path": str(agent.path),
        "enabled": agent.enabled,
    }


# ─────────────────────────────────────────────────────────────────────────────
# assign_agent
# ─────────────────────────────────────────────────────────────────────────────

def assign_agent(
    agent_name: str,
    task: str,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    role: Optional[str] = None,
    workdir: Optional[str] = None,
    parent_agent: Optional[Any] = None,
) -> str:
    """
    Delegate a task to a named agent, compiling its prompt as the child context.

    The named agent's prompt is embedded in a structured context block that
    informs the child agent of its identity and instructions.  The child's
    toolsets and role are derived from the agent definition (respecting
    tools.mode == "restrict" for allow_toolsets) and can be overridden at
    call time.

    Args:
        agent_name: Name of the agent to assign the task to.
        task: The user's task/goal for the named agent.
        context: Optional additional user context to include in the child's context.
        toolsets: Runtime override for the agent's effective toolsets.
                  If provided, overrides agent.tools.allow_toolsets.
        role: Runtime override for the agent's delegation role.
              If provided, overrides agent.delegation_role.
        workdir: Optional working directory for agent discovery.
                 If not provided, falls back to TERMINAL_CWD env var or
                 parent_agent terminal_cwd/cwd attributes.
        parent_agent: Optional. The calling agent instance (dispatcher injects this).

    Returns:
        JSON string with success, agent metadata, and the delegate_task result
        (parsed if JSON, raw string otherwise).
    """
    # ── Require parent_agent ──────────────────────────────────────────────
    if parent_agent is None:
        return json.dumps(
            {
                "success": False,
                "error": "assign_agent requires a parent_agent context. "
                         "Omitting parent_agent is not allowed.",
            },
            ensure_ascii=False,
        )

    started_at = time.monotonic()
    parent_session_id = _parent_session_id(parent_agent)

    # ── Resolve effective workdir ──────────────────────────────────────────
    effective_workdir = _resolve_workdir(workdir, parent_agent)
    _emit_runtime_trace(
        "assign_agent.requested",
        parent_session_id=parent_session_id,
        agent_name=agent_name,
        task_preview=(task or "")[:120],
        workdir_requested=workdir,
        workdir_effective=effective_workdir,
    )

    # ── Resolve the agent ─────────────────────────────────────────────────
    try:
        from agent.agent_registry import get_agent

        agent = get_agent(name=agent_name, workdir=effective_workdir)
        if agent is None:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Agent '{agent_name}' not found.",
                },
                ensure_ascii=False,
            )

        if not agent.enabled:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Agent '{agent_name}' is disabled.",
                },
                ensure_ascii=False,
            )
    except Exception as exc:
        logger.warning("assign_agent resolution failed for '%s': %s", agent_name, exc)
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to load agent '{agent_name}': {exc}",
            },
            ensure_ascii=False,
        )

    # ── Validate routing mode ───────────────────────────────────────────────
    # assign_agent supports inherited routing and native Hermes provider/model
    # overrides.  ACP transport remains unsupported here; it belongs either in
    # delegate_task's explicit ACP override path or trusted CLI runners.
    routing_mode = agent.routing.mode
    if routing_mode not in (None, "inherit", "hermes"):
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"Agent '{agent_name}' has routing.mode='{routing_mode}' which is "
                    "not supported. Supported routing modes: inherit, hermes. "
                    "Use trusted CLI runners for external transports."
                ),
            },
            ensure_ascii=False,
        )

    # ── Reject routing.acp_command defensively ─────────────────────────────
    if agent.routing.acp_command:
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"Agent '{agent_name}' has routing.acp_command set, which is "
                    "not supported in the PR1 native registry. "
                    "Use delegate_task for ACP-based routing."
                ),
            },
            ensure_ascii=False,
        )

    # ── Compute effective toolsets ─────────────────────────────────────────
    # Runtime toolsets override agent defaults; otherwise derive from
    # agent.tools.allow_toolsets when mode == "restrict".
    if toolsets is not None:
        effective_toolsets: Optional[List[str]] = list(toolsets)
    elif agent.tools.mode == "restrict" and agent.tools.allow_toolsets:
        effective_toolsets = list(agent.tools.allow_toolsets)
    else:
        effective_toolsets = None

    # ── Compute effective role ─────────────────────────────────────────────
    # Runtime role overrides agent delegation_role.
    effective_role: str = role if role else (agent.delegation_role or "leaf")

    # ── Handle runner execution modes ─────────────────────────────────────
    # delegate_task remains the default in-process runner.  CLI-backed agents
    # are executed via trusted agent_runners config; agent files only name the
    # runner and continuity preference.
    runner_mode = agent.routing.runner_mode or "delegate_task"
    _emit_runtime_trace(
        "assign_agent.resolved",
        parent_session_id=parent_session_id,
        agent=_agent_trace_payload(agent),
        routing={
            "mode": agent.routing.mode,
            "provider": agent.routing.provider,
            "model": agent.routing.model,
            "runner_mode": runner_mode,
            "runner_name": agent.routing.runner_name,
            "runner_continue": agent.routing.runner_continue,
        },
        toolsets_effective=effective_toolsets,
        role_effective=effective_role,
        workdir_effective=effective_workdir,
    )
    if runner_mode == "cli":
        try:
            from agent.agent_runner import run_cli_agent

            _emit_runtime_trace(
                "assign_agent.dispatched",
                parent_session_id=parent_session_id,
                agent=_agent_trace_payload(agent),
                dispatch="cli_runner",
                routing={"runner_mode": runner_mode, "runner_name": agent.routing.runner_name},
            )
            cli_result = run_cli_agent(
                agent=agent,
                task=task,
                context=context,
                workdir=effective_workdir,
                parent_session_id=parent_session_id,
            )
            payload = cli_result.to_dict()
            payload["agent"] = agent.list_summary()
            _emit_runtime_trace(
                "assign_agent.completed",
                parent_session_id=parent_session_id,
                agent=_agent_trace_payload(agent),
                dispatch="cli_runner",
                success=cli_result.success,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                runner={
                    "mode": "cli",
                    "name": cli_result.runner_name,
                    "resumed": cli_result.resumed,
                },
                external_session_id=cli_result.external_session_id,
                returncode=cli_result.returncode,
                error=cli_result.error,
                warning=cli_result.warning,
            )
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            logger.warning("assign_agent cli runner failed: %s", exc)
            _emit_runtime_trace(
                "assign_agent.completed",
                parent_session_id=parent_session_id,
                agent=_agent_trace_payload(agent),
                dispatch="cli_runner",
                success=False,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                error=str(exc),
            )
            return json.dumps(
                {
                    "success": False,
                    "error": f"CLI runner failed: {exc}",
                    "agent": agent.list_summary(),
                },
                ensure_ascii=False,
            )

    if runner_mode != "delegate_task":
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"Agent '{agent_name}' has runner.mode='{runner_mode}' which is "
                    "not supported. Supported runner modes: delegate_task, cli."
                ),
            },
            ensure_ascii=False,
        )

    # ── Compile context block ──────────────────────────────────────────────
    # Build a structured context string from the agent definition:
    #   ## Named agent: <name>
    #   Source/path
    #   ## Agent instructions
    #   <prompt>
    #   ## Assignment context
    #   <user context>
    context_parts: List[str] = [
        f"## Named agent: {agent.name}",
        f"Source: {agent.source} — {agent.path}",
        "",
        "## Agent instructions",
        agent.prompt,
    ]
    if context and context.strip():
        context_parts.extend(["", "## Assignment context", context.strip()])

    compiled_context = "\n".join(context_parts)

    # ── Delegate via delegate_task ─────────────────────────────────────────
    try:
        from tools.delegate_tool import delegate_task

        _emit_runtime_trace(
            "assign_agent.dispatched",
            parent_session_id=parent_session_id,
            agent=_agent_trace_payload(agent),
            dispatch="delegate_task",
            routing={"runner_mode": runner_mode},
            toolsets_effective=effective_toolsets,
            role_effective=effective_role,
        )
        raw_result = delegate_task(
            goal=task,
            context=compiled_context,
            toolsets=effective_toolsets,
            role=effective_role,
            parent_agent=parent_agent,
            model=agent.routing.model if routing_mode == "hermes" else None,
            provider=agent.routing.provider if routing_mode == "hermes" else None,
            base_url=agent.routing.base_url if routing_mode == "hermes" else None,
            api_mode=agent.routing.api_mode if routing_mode == "hermes" else None,
        )
    except Exception as exc:
        logger.warning("assign_agent delegate_task failed: %s", exc)
        _emit_runtime_trace(
            "assign_agent.completed",
            parent_session_id=parent_session_id,
            agent=_agent_trace_payload(agent),
            dispatch="delegate_task",
            success=False,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            error=str(exc),
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Delegation failed: {exc}",
                "agent": agent.list_summary(),
            },
            ensure_ascii=False,
        )

    # ── Parse and wrap the result ─────────────────────────────────────────
    try:
        parsed_result = json.loads(raw_result)
        success = parsed_result.get("success", True)
        _emit_runtime_trace(
            "assign_agent.completed",
            parent_session_id=parent_session_id,
            agent=_agent_trace_payload(agent),
            dispatch="delegate_task",
            success=success,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )
        return json.dumps(
            {
                "success": success,
                "agent": agent.list_summary(),
                "result": parsed_result,
            },
            ensure_ascii=False,
        )
    except (json.JSONDecodeError, TypeError):
        _emit_runtime_trace(
            "assign_agent.completed",
            parent_session_id=parent_session_id,
            agent=_agent_trace_payload(agent),
            dispatch="delegate_task",
            success=True,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )
        return json.dumps(
            {
                "success": True,
                "agent": agent.list_summary(),
                "result": raw_result,
            },
            ensure_ascii=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Schema definitions
# ─────────────────────────────────────────────────────────────────────────────

AGENTS_LIST_SCHEMA = {
    "name": "agents_list",
    "description": (
        "List all discovered named agents with compact metadata (no prompts). "
        "Agents are defined in $HERMES_HOME/agents/ (global) or "
        "<project>/.hermes/agents/ (project-local). "
        "Use this to discover available named agents before calling agent_view "
        "or assign_agent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": (
                    "Optional tag to filter agents by (e.g. 'web', 'code'). "
                    "If provided, only agents with this tag are returned. "
                    "If no agent matches, an empty list is returned."
                ),
            },
            "include_disabled": {
                "type": "boolean",
                "description": "Include disabled agents in the listing (default: false).",
                "default": False,
            },
            "include_shadowed": {
                "type": "boolean",
                "description": (
                    "Include agents shadowed by higher-priority entries of the same name "
                    "(default: false)."
                ),
                "default": False,
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Optional working directory used to resolve the project root "
                    "for project-local agent discovery."
                ),
            },
        },
        "required": [],
    },
}

AGENT_VIEW_SCHEMA = {
    "name": "agent_view",
    "description": (
        "Get the full definition of a single named agent, including its prompt. "
        "Returns all metadata (routing, tools, skills, limits, delegation role, etc.). "
        "Use this before assign_agent to inspect an agent's instructions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "The agent's unique name (lowercase alphanumeric + hyphens, "
                    "1-64 chars, must start with a letter). "
                    "Path traversal attempts are rejected."
                ),
            },
            "source": {
                "type": "string",
                "enum": ["global", "project"],
                "description": (
                    "Optional source filter. "
                    "If omitted, returns the effective (non-shadowed) agent."
                ),
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Optional working directory for project-local agent discovery."
                ),
            },
        },
        "required": ["name"],
    },
}

ASSIGN_AGENT_SCHEMA = {
    "name": "assign_agent",
    "description": (
        "Delegate a task to a named agent by composing its prompt into the child's "
        "context.  The named agent's instructions become the child's system prompt "
        "fragment, allowing reusable agent definitions to be invoked by name. "
        "The child inherits toolsets and delegation role from the agent definition "
        "(or runtime overrides), and the parent's context is compiled as a structured "
        "block."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": (
                    "Name of the agent to assign the task to. "
                    "Must be a valid agent definition in "
                    "$HERMES_HOME/agents/ or <project>/.hermes/agents/."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "The user's task or goal to delegate to the named agent. "
                    "This is passed as the child's 'goal' parameter."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional additional context from the parent to include "
                    "in the child's context block (e.g. relevant files, user preferences)."
                ),
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Runtime override for the agent's effective toolsets. "
                    "If provided, overrides agent.tools.allow_toolsets. "
                    "Example: ['web', 'terminal']"
                ),
            },
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "description": (
                    "Runtime override for the agent's delegation role. "
                    "'leaf' (default): cannot delegate further. "
                    "'orchestrator': retains the delegation toolset and can spawn workers."
                ),
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Optional working directory for agent discovery. "
                    "Used to resolve the project root for project-local agents."
                ),
            },
        },
        "required": ["agent_name", "task"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

registry.register(
    name="agents_list",
    toolset="agents",
    schema=AGENTS_LIST_SCHEMA,
    handler=lambda args, **kw: agents_list(
        category=args.get("category"),
        include_disabled=args.get("include_disabled", False),
        include_shadowed=args.get("include_shadowed", False),
        workdir=args.get("workdir"),
    ),
    description="List all discovered named agents with compact metadata",
    emoji="🤖",
)

registry.register(
    name="agent_view",
    toolset="agents",
    schema=AGENT_VIEW_SCHEMA,
    handler=lambda args, **kw: agent_view(
        name=args.get("name"),
        source=args.get("source"),
        workdir=args.get("workdir"),
    ),
    description="Get the full definition of a single named agent including its prompt",
    emoji="🔍",
)

registry.register(
    name="assign_agent",
    toolset="delegation",
    schema=ASSIGN_AGENT_SCHEMA,
    handler=lambda args, **kw: assign_agent(
        agent_name=args.get("agent_name"),
        task=args.get("task"),
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        role=args.get("role"),
        workdir=args.get("workdir"),
        parent_agent=kw.get("parent_agent"),
    ),
    description="Delegate a task to a named agent by composing its prompt into child context",
    emoji="📋",
)
