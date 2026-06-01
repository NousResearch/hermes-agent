#!/usr/bin/env python3
"""Codex CLI delegation tool.

This tool is intentionally small: Hermes constructs a bounded, non-interactive
`codex exec` subprocess and returns structured evidence about the invocation.
It does not parse or rewrite Codex output into success claims.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.registry import registry, tool_error

DEFAULT_TIMEOUT_SECONDS = 1800
MAX_TIMEOUT_SECONDS = 7200


def _profile_local_codex() -> str | None:
    candidate = Path.home() / ".local" / "bin" / "codex"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _codex_binary() -> str:
    return shutil.which("codex") or _profile_local_codex() or "codex"


def _bounded_timeout(timeout: int | None) -> int:
    if timeout is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(timeout)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return max(1, min(value, MAX_TIMEOUT_SECONDS))


def _codex_env() -> dict[str, str]:
    env = os.environ.copy()
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("CODEX_HOME", str(Path.home() / ".codex"))
    return env


def delegate_to_codex_tool(
    task: str,
    working_directory: str,
    context_files: list[str] | None = None,
    timeout: int | None = None,
    sandbox: str = "workspace-write",
    model: str | None = None,
) -> str:
    """Run `codex exec` non-interactively and return structured evidence."""
    if not task or not isinstance(task, str):
        return json.dumps({"ok": False, "error": "missing_task"}, ensure_ascii=False)

    workdir = Path(working_directory or "").expanduser().resolve()
    if not workdir.exists() or not workdir.is_dir():
        return json.dumps({
            "ok": False,
            "error": "invalid_working_directory",
            "working_directory": str(workdir),
        }, ensure_ascii=False)

    timeout_seconds = _bounded_timeout(timeout)
    prompt = task
    if context_files:
        prompt += "\n\nContext files to inspect first:\n" + "\n".join(f"- {p}" for p in context_files)

    cmd = [
        _codex_binary(),
        "exec",
        "--json",
        "--cd",
        str(workdir),
        "--skip-git-repo-check",
        "--sandbox",
        sandbox or "workspace-write",
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=_codex_env(),
            stdin=None,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return json.dumps({
            "ok": False,
            "error": "timeout",
            "execution_mode": "non_pty",
            "timeout_seconds": timeout_seconds,
            "command": cmd[:6] + ["<prompt>"],
            "stdout": exc.output or "",
            "stderr": exc.stderr or "",
        }, ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({
            "ok": False,
            "error": "codex_not_found",
            "execution_mode": "non_pty",
            "command": cmd[:1],
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "error": "subprocess_error",
            "execution_mode": "non_pty",
            "message": str(exc),
        }, ensure_ascii=False)

    return json.dumps({
        "ok": completed.returncode == 0,
        "error": None if completed.returncode == 0 else "codex_exec_failed",
        "execution_mode": "non_pty",
        "returncode": completed.returncode,
        "timeout_seconds": timeout_seconds,
        "working_directory": str(workdir),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }, ensure_ascii=False)


DELEGATE_TO_CODEX_SCHEMA: dict[str, Any] = {
    "name": "delegate_to_codex",
    "description": (
        "Delegate code-writing work to the local Codex CLI via non-interactive `codex exec`. "
        "Use this for product-code edits when Build Zeller has code_writes_require_codex enabled. "
        "Returns structured subprocess evidence; verify changed files and tests separately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Self-contained Codex task brief."},
            "working_directory": {"type": "string", "description": "Repository/workspace root for Codex."},
            "context_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional files Codex should inspect first.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout seconds, capped at {MAX_TIMEOUT_SECONDS}.",
                "default": DEFAULT_TIMEOUT_SECONDS,
            },
            "sandbox": {
                "type": "string",
                "enum": ["read-only", "workspace-write", "danger-full-access"],
                "default": "workspace-write",
                "description": "Codex sandbox mode.",
            },
            "model": {"type": "string", "description": "Optional Codex model override."},
        },
        "required": ["task", "working_directory"],
    },
}


def _handle_delegate_to_codex(args, **kw):
    return delegate_to_codex_tool(
        task=args.get("task", ""),
        working_directory=args.get("working_directory", ""),
        context_files=args.get("context_files"),
        timeout=args.get("timeout"),
        sandbox=args.get("sandbox", "workspace-write"),
        model=args.get("model"),
    )


registry.register(
    name="delegate_to_codex",
    toolset="codex",
    schema=DELEGATE_TO_CODEX_SCHEMA,
    handler=_handle_delegate_to_codex,
    emoji="🤖",
    max_result_size_chars=100_000,
)
