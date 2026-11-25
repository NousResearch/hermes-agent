#!/usr/bin/env python3
"""Delegate coding work to the local Codex CLI via ACP."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.delegate_tool import delegate_task
from tools.registry import registry, tool_error

DEFAULT_CODEX_ACP_ARGS = ["--acp", "--stdio"]


def check_codex_task_requirements() -> bool:
    """Codex delegation itself has no static requirements.

    The actual command availability is validated when delegate_task spawns the
    child process, which yields a more precise runtime error than a coarse PATH
    check here.
    """

    return True


CODEX_TASK_SCHEMA: Dict[str, Any] = {
    "name": "codex_task",
    "description": (
        "Launch the local Codex CLI as a subagent and delegate a coding task to it. "
        "Use this when you want Hermes to hand implementation/debugging work to Codex "
        "without manually configuring ACP arguments each time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What Codex should accomplish. Be specific and self-contained.",
            },
            "context": {
                "type": "string",
                "description": "Relevant background: file paths, errors, repo structure, constraints.",
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional toolsets for the Codex subagent. Common choice: ['terminal', 'file'].",
            },
            "max_iterations": {
                "type": "integer",
                "description": "Optional max tool-calling turns for the Codex subagent.",
            },
        },
        "required": ["goal"],
    },
}


def codex_task(
    goal: str,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    max_iterations: Optional[int] = None,
    *,
    parent_agent=None,
) -> str:
    if not goal or not str(goal).strip():
        return tool_error("goal is required for codex_task")

    return delegate_task(
        goal=str(goal).strip(),
        context=context,
        toolsets=toolsets,
        max_iterations=max_iterations,
        acp_command="codex",
        acp_args=list(DEFAULT_CODEX_ACP_ARGS),
        parent_agent=parent_agent,
    )


def _handle_codex_task(args: Dict[str, Any], **kw) -> str:
    goal = str(args.get("goal") or "").strip()
    if not goal:
        return tool_error("goal is required for codex_task")
    return codex_task(
        goal=goal,
        context=args.get("context"),
        toolsets=args.get("toolsets"),
        max_iterations=args.get("max_iterations"),
        parent_agent=kw.get("parent_agent"),
    )


registry.register(
    name="codex_task",
    toolset="codex",
    schema=CODEX_TASK_SCHEMA,
    handler=_handle_codex_task,
    check_fn=check_codex_task_requirements,
    emoji="🧑‍💻",
)
