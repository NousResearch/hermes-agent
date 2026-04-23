#!/usr/bin/env python3
"""Hermes tool for delegating one-shot work to the local Claude Code CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.registry import registry, tool_error


BRIDGE_SCRIPT_PATH = Path.home() / ".hermes" / "hermes-agent" / "scripts" / "run_claude_cli_json.py"
WRAPPER_PATH = Path.home() / ".local" / "bin" / "hermes-call-claude"
DEFAULT_MODEL = "opus"
DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_SECONDS = 300


def _path_exists(path: Path) -> bool:
    return path.exists()


def check_claude_cli_requirements() -> bool:
    return shutil.which("claude") is not None and _path_exists(BRIDGE_SCRIPT_PATH)


def _coerce_allowed_tools(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        joined = ",".join(str(item).strip() for item in value if str(item).strip())
        return joined or None
    text = str(value).strip()
    return text or None


def claude_cli_run_tool(
    *,
    prompt: str,
    workdir: str | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = DEFAULT_MAX_TURNS,
    allowed_tools: Any = None,
    append_system_prompt: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    if not prompt or not prompt.strip():
        return tool_error("prompt is required")
    if not _path_exists(BRIDGE_SCRIPT_PATH):
        return tool_error(f"Claude CLI bridge script not found: {BRIDGE_SCRIPT_PATH}")
    claude_path = shutil.which("claude")
    if not claude_path:
        return tool_error("claude CLI not found on PATH")

    launcher = str(WRAPPER_PATH if _path_exists(WRAPPER_PATH) else Path(sys.executable))
    command = [launcher]
    if launcher == sys.executable:
        command.append(str(BRIDGE_SCRIPT_PATH))
    command.append(prompt)
    if workdir:
        command.extend(["--workdir", str(Path(workdir).expanduser())])
    if model:
        command.extend(["--model", model])
    command.extend(["--max-turns", str(max_turns)])
    normalized_allowed = _coerce_allowed_tools(allowed_tools)
    if normalized_allowed:
        command.extend(["--allowed-tools", normalized_allowed])
    if append_system_prompt:
        command.extend(["--append-system-prompt", append_system_prompt])
    command.extend(["--timeout", str(timeout_seconds)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 5,
        )
    except subprocess.TimeoutExpired:
        return tool_error(f"Claude CLI bridge timed out after {timeout_seconds + 5}s")

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if not stdout:
        return tool_error(stderr or "Claude CLI bridge produced no output")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return tool_error(f"Claude CLI bridge returned non-JSON output: {stdout[:400]}")

    if stderr and not payload.get("stderr"):
        payload["stderr"] = stderr
    return json.dumps(payload, ensure_ascii=False)


CLAUDE_CLI_RUN_SCHEMA = {
    "name": "claude_cli_run",
    "description": (
        "Run the local Claude Code CLI in print mode and return structured JSON. "
        "Use this when you specifically want the first-party Claude Code execution path "
        "instead of Hermes's native Anthropic provider, for example when local Claude Code "
        "auth works but native Anthropic provider billing/auth semantics do not."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Prompt to send to Claude Code CLI.",
            },
            "workdir": {
                "type": "string",
                "description": "Optional working directory for the claude process.",
            },
            "model": {
                "type": "string",
                "description": "Claude model alias/name (default: opus).",
                "default": DEFAULT_MODEL,
            },
            "max_turns": {
                "type": "integer",
                "description": "Max turns for claude -p (default: 10).",
                "default": DEFAULT_MAX_TURNS,
                "minimum": 1,
            },
            "allowed_tools": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": "Optional pass-through for Claude Code --allowedTools.",
            },
            "append_system_prompt": {
                "type": "string",
                "description": "Optional additional system prompt passed to Claude Code.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Subprocess timeout in seconds (default: 300).",
                "default": DEFAULT_TIMEOUT_SECONDS,
                "minimum": 1,
            },
        },
        "required": ["prompt"],
    },
}


registry.register(
    name="claude_cli_run",
    toolset="terminal",
    schema=CLAUDE_CLI_RUN_SCHEMA,
    handler=lambda args, **kw: claude_cli_run_tool(
        prompt=args.get("prompt", ""),
        workdir=args.get("workdir"),
        model=args.get("model", DEFAULT_MODEL),
        max_turns=args.get("max_turns", DEFAULT_MAX_TURNS),
        allowed_tools=args.get("allowed_tools"),
        append_system_prompt=args.get("append_system_prompt"),
        timeout_seconds=args.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
    ),
    check_fn=check_claude_cli_requirements,
    emoji="🧠",
)
