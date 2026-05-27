#!/usr/bin/env python3
"""Aider subagent tool for code-writing delegation."""

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_OPENROUTER_PREFIX = "openrouter/"
_DEFAULT_TIMEOUT_SECONDS = 1800
_MAX_OUTPUT_CHARS = 12000


def _normalize_model(model: Any) -> Optional[str]:
    """Return an OpenRouter-qualified model name or None."""
    if model is None:
        return None
    model_name = str(model).strip()
    if not model_name:
        return None
    if model_name.startswith(_OPENROUTER_PREFIX):
        return model_name
    return f"{_OPENROUTER_PREFIX}{model_name}"


def _build_aider_command(
    instruction: str,
    model: Any = None,
    *,
    executable: str = "aider",
) -> list[str]:
    """Build the non-interactive Aider command for a delegated instruction."""
    instruction_text = str(instruction or "").strip()
    if not instruction_text:
        raise ValueError("instruction is required")

    command = [executable, "--message", instruction_text, "--yes"]
    normalized_model = _normalize_model(model)
    if normalized_model:
        command.extend(["--model", normalized_model])
    return command


def _command_preview(command: list[str]) -> str:
    """Return a shell-style command string without echoing the full prompt."""
    preview: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            preview.append("<instruction>")
            redact_next = False
            continue
        preview.append(part)
        if part == "--message":
            redact_next = True
    return shlex.join(preview)


def _resolve_aider_executable() -> Optional[str]:
    """Resolve the Aider executable from AIDER_BIN or PATH."""
    configured = os.getenv("AIDER_BIN", "").strip()
    if configured:
        return shutil.which(configured) or configured
    return shutil.which("aider")


def _resolve_workdir(workdir: Any = None) -> Path:
    """Resolve the directory where Aider should run."""
    base = Path(os.environ.get("TERMINAL_CWD") or os.getcwd()).expanduser()
    raw = str(workdir or "").strip()
    path = Path(raw).expanduser() if raw else base
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"workdir does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"workdir is not a directory: {resolved}")
    return resolved


def _coerce_timeout(timeout_seconds: Any = None) -> int:
    """Return a bounded timeout for the Aider subprocess."""
    if timeout_seconds in (None, ""):
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = int(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be an integer") from exc
    return max(30, timeout)


def _tail_text(value: Any, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Return a bounded text tail suitable for a tool result."""
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def aider_subagent(
    instruction: Any,
    model: Any = None,
    workdir: Any = None,
    timeout_seconds: Any = None,
) -> str:
    """Run Aider as a local code-writing subagent and return a JSON result."""
    started = time.monotonic()
    try:
        cwd = _resolve_workdir(workdir)
        timeout = _coerce_timeout(timeout_seconds)
        executable = _resolve_aider_executable()
        if not executable:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Aider executable not found. Install Aider or set AIDER_BIN to its executable path.",
                    "command": _command_preview(_build_aider_command(instruction, model)),
                    "workdir": str(cwd),
                },
                ensure_ascii=False,
            )

        command = _build_aider_command(instruction, model, executable=executable)
        env = os.environ.copy()
        env.setdefault("AIDER_CHECK_UPDATE", "false")
        env.setdefault("AIDER_SHOW_RELEASE_NOTES", "false")

        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        status = "completed" if completed.returncode == 0 else "error"
        return json.dumps(
            {
                "status": status,
                "exit_code": completed.returncode,
                "duration_seconds": round(time.monotonic() - started, 2),
                "command": _command_preview(command),
                "model": _normalize_model(model),
                "workdir": str(cwd),
                "stdout_tail": _tail_text(completed.stdout),
                "stderr_tail": _tail_text(completed.stderr),
            },
            ensure_ascii=False,
        )
    except subprocess.TimeoutExpired as exc:
        return json.dumps(
            {
                "status": "timeout",
                "error": f"Aider timed out after {_coerce_timeout(timeout_seconds)} seconds",
                "duration_seconds": round(time.monotonic() - started, 2),
                "stdout_tail": _tail_text(exc.stdout),
                "stderr_tail": _tail_text(exc.stderr),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("Aider subagent failed: %s", exc)
        return json.dumps(
            {
                "status": "error",
                "error": str(exc),
                "duration_seconds": round(time.monotonic() - started, 2),
            },
            ensure_ascii=False,
        )


AIDER_SUBAGENT_SCHEMA = {
    "name": "aider_subagent",
    "description": (
        "Delegate code-writing work to a local Aider subprocess. Hermes stays "
        "the orchestrator; Aider performs repository edits. The default command "
        "is `aider --message <instruction> --yes`. If model is provided, it is "
        "passed as `--model openrouter/<model>`."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "Self-contained coding instruction for Aider to execute.",
            },
            "model": {
                "type": "string",
                "description": "Optional OpenRouter model name, with or without the openrouter/ prefix.",
            },
            "workdir": {
                "type": "string",
                "description": "Optional repository directory for Aider. Defaults to TERMINAL_CWD or the current process directory.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional subprocess timeout in seconds. Defaults to 1800.",
                "minimum": 30,
            },
        },
        "required": ["instruction"],
    },
}


def _handle_aider_subagent(args: dict, **_kwargs) -> str:
    """Registry handler for the Aider subagent tool."""
    return aider_subagent(
        instruction=args.get("instruction"),
        model=args.get("model"),
        workdir=args.get("workdir"),
        timeout_seconds=args.get("timeout_seconds"),
    )


registry.register(
    name="aider_subagent",
    toolset="aider",
    schema=AIDER_SUBAGENT_SCHEMA,
    handler=_handle_aider_subagent,
    max_result_size_chars=100_000,
)