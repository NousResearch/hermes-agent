"""Hermes-owned Codex executor tool.

``codex_session`` is a bounded execution lane called by Hermes. Codex returns
structured evidence; Hermes stays responsible for final response and verification.
"""
from __future__ import annotations

import json
import shutil
from typing import Any

from agent.executors.codex_session import run_codex_session
from agent.transports.codex_app_server_session import CodexAppServerSession
from tools.registry import registry


def check_codex_session_requirements() -> bool:
    """Return True when the bounded Codex executor prerequisites are present.

    The executor is still opt-in at the toolset/config level; this check keeps
    the tool hidden unless the Codex CLI app-server prerequisite exists on PATH.
    """
    return shutil.which("codex") is not None


CODEX_SESSION_SCHEMA = {
    "name": "codex_session",
    "description": (
        "Run one bounded Codex app-server executor session and return structured "
        "execution evidence. This is not a user-facing final answer; Hermes must "
        "verify the evidence before reporting completion."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Concrete bounded execution task for Codex to attempt.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the Codex app-server session. Defaults to the current process cwd.",
            },
            "turn_timeout": {
                "type": "number",
                "description": "Maximum seconds to wait for the Codex turn. Defaults to 600.",
            },
            "permission_profile": {
                "type": "string",
                "description": "Optional Codex permission profile override such as workspace-write, read-only-with-approval, or hermes-worktree-write.",
            },
        },
        "required": ["task"],
    },
}


def codex_session(
    task: str,
    cwd: str | None = None,
    turn_timeout: float = 600.0,
    permission_profile: str | None = None,
    task_id: str | None = None,
) -> str:
    """Tool handler for the Hermes-owned Codex executor session."""
    del task_id  # reserved for future run correlation; avoid unused lint noise
    evidence = run_codex_session(
        task=task,
        cwd=cwd,
        turn_timeout=float(turn_timeout or 600.0),
        permission_profile=permission_profile,
        session_factory=CodexAppServerSession,
    )
    return json.dumps(evidence, ensure_ascii=False)


def _handler(args: dict[str, Any], **kwargs: Any) -> str:
    return codex_session(
        task=args.get("task") or "",
        cwd=args.get("cwd"),
        turn_timeout=args.get("turn_timeout") or 600.0,
        permission_profile=args.get("permission_profile"),
        task_id=kwargs.get("task_id"),
    )


registry.register(
    name="codex_session",
    toolset="codex",
    schema=CODEX_SESSION_SCHEMA,
    handler=_handler,
    check_fn=check_codex_session_requirements,
    description=CODEX_SESSION_SCHEMA["description"],
    emoji="🧭",
    max_result_size_chars=20000,
)
