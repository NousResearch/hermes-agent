"""Durable background Hermes agent launcher.

This complements ``delegate_task``: delegation is synchronous inside the parent
turn, while this tool starts an independent Hermes process tracked by the
terminal/process manager so long work can continue after the current chat turn.
"""

from __future__ import annotations

import json
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error
from tools.terminal_tool import check_terminal_requirements, terminal_tool


def _as_csv(value: Optional[str | Iterable[str]]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]
    return ",".join(part for part in parts if part)


def _write_prompt_file(prompt: str, name: str = "") -> Path:
    root = get_hermes_home() / "background_agents" / "prompts"
    root.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in (name or "agent"))
    safe_name = safe_name.strip("-")[:48] or "agent"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}-{safe_name}-{uuid.uuid4().hex[:8]}.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


def spawn_background_agent(
    prompt: str,
    toolsets: Optional[list[str] | str] = None,
    skills: Optional[list[str] | str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    workdir: Optional[str] = None,
    name: Optional[str] = None,
    notify_on_complete: bool = True,
    startup_timeout: int = 10,
) -> str:
    """Start a one-shot Hermes agent in a managed background process."""

    if not isinstance(prompt, str) or not prompt.strip():
        return tool_error("spawn_background_agent requires a non-empty prompt.")

    prompt_path = _write_prompt_file(prompt, name or "agent")
    prompt_arg = f'"$(cat {shlex.quote(str(prompt_path))})"'
    cmd_parts = ["hermes", "chat", "-q", prompt_arg, "--quiet"]

    toolsets_csv = _as_csv(toolsets)
    if toolsets_csv:
        cmd_parts.extend(["--toolsets", shlex.quote(toolsets_csv)])

    skills_csv = _as_csv(skills)
    if skills_csv:
        cmd_parts.extend(["--skills", shlex.quote(skills_csv)])

    if model:
        cmd_parts.extend(["--model", shlex.quote(str(model))])
    if provider:
        cmd_parts.extend(["--provider", shlex.quote(str(provider))])

    command = " ".join(cmd_parts)
    raw = terminal_tool(
        command=command,
        background=True,
        timeout=int(startup_timeout or 10),
        workdir=workdir or None,
        notify_on_complete=bool(notify_on_complete),
    )

    try:
        result = json.loads(raw)
    except Exception:
        result = {"output": raw}

    error = result.get("error") if isinstance(result, dict) else None
    if error:
        return json.dumps({
            "success": False,
            "error": error,
            "prompt_path": str(prompt_path),
            "terminal_result": result,
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "session_id": result.get("session_id") if isinstance(result, dict) else None,
        "pid": result.get("pid") if isinstance(result, dict) else None,
        "prompt_path": str(prompt_path),
        "command": command,
        "notify_on_complete": bool(notify_on_complete),
        "terminal_result": result,
    }, ensure_ascii=False)


SPAWN_BACKGROUND_AGENT_SCHEMA = {
    "name": "spawn_background_agent",
    "description": (
        "Spawn a durable one-shot Hermes agent in a managed background process. "
        "Use this when work should continue after the current chat turn, when "
        "the user says to spawn a mini/background agent, or when the user needs "
        "to keep chatting while the worker runs. Unlike delegate_task, this does "
        "not run synchronously inside the parent turn; manage it with the process "
        "tool using the returned session_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Self-contained task instruction for the background Hermes agent.",
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional toolsets for the background agent, e.g. ['terminal','file'].",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional skills to preload in the background agent.",
            },
            "model": {
                "type": "string",
                "description": "Optional model override passed to hermes chat --model.",
            },
            "provider": {
                "type": "string",
                "description": "Optional provider override passed to hermes chat --provider.",
            },
            "workdir": {
                "type": "string",
                "description": "Optional working directory for the background process.",
            },
            "name": {
                "type": "string",
                "description": "Optional short label used only for the saved prompt filename.",
            },
            "notify_on_complete": {
                "type": "boolean",
                "description": "If true, deliver one process notification when the worker exits.",
            },
            "startup_timeout": {
                "type": "integer",
                "description": "Seconds to wait for the background process to start (default 10).",
            },
        },
        "required": ["prompt"],
    },
}


registry.register(
    name="spawn_background_agent",
    toolset="delegation",
    schema=SPAWN_BACKGROUND_AGENT_SCHEMA,
    handler=lambda args, **kw: spawn_background_agent(
        prompt=args.get("prompt", ""),
        toolsets=args.get("toolsets"),
        skills=args.get("skills"),
        model=args.get("model"),
        provider=args.get("provider"),
        workdir=args.get("workdir"),
        name=args.get("name"),
        notify_on_complete=args.get("notify_on_complete", True),
        startup_timeout=args.get("startup_timeout", 10),
    ),
    check_fn=check_terminal_requirements,
    emoji="🧵",
)
