#!/usr/bin/env python3
"""Aider subagent tool for code-writing delegation."""

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
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
    state_dir: Path | None = None,
) -> list[str]:
    """Build the non-interactive Aider command for a delegated instruction."""
    instruction_text = str(instruction or "").strip()
    if not instruction_text:
        raise ValueError("instruction is required")

    command = [
        executable,
        "--message",
        instruction_text,
        "--yes",
        "--no-auto-commits",
        "--no-gitignore",
        "--no-restore-chat-history",
        "--map-tokens",
        "0",
    ]
    if state_dir is not None:
        command.extend(
            [
                "--input-history-file",
                str(state_dir / "input.history"),
                "--chat-history-file",
                str(state_dir / "chat.history.md"),
                "--llm-history-file",
                str(state_dir / "llm.history.md"),
            ]
        )
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


def _shell_unquote(value: str) -> str:
    """Remove simple shell quoting from a dotenv value."""
    try:
        parts = shlex.split(value, comments=False, posix=True)
    except ValueError:
        return value.strip().strip("'").strip('"')
    if not parts:
        return ""
    return parts[0]


def _load_env_file(env_file: Any = None) -> dict[str, str]:
    """Load non-comment KEY=VALUE pairs from the Hermes env file."""
    path = Path(str(env_file or "~/.hermes/.env")).expanduser()
    if not path.exists() or not path.is_file():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if key:
            values[key] = _shell_unquote(raw_value.strip())
    return values


def _first_present(*values: Optional[str]) -> Optional[str]:
    """Return the first non-empty string from a list of candidates."""
    for value in values:
        if value:
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _build_aider_env(model: Any = None, env_file: Any = None) -> dict[str, str]:
    """Build Aider's subprocess environment with model credentials bridged."""
    env = os.environ.copy()
    file_values = _load_env_file(env_file)
    normalized_model = _normalize_model(model)
    if normalized_model and normalized_model.startswith(_OPENROUTER_PREFIX):
        openrouter_token = _first_present(
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_KEY"),
            file_values.get("OPENROUTER_API_KEY"),
            file_values.get("OPENROUTER_KEY"),
        )
        if openrouter_token:
            env["OPENROUTER_API_KEY"] = openrouter_token
            env["OPENAI_API_KEY"] = openrouter_token
    return env


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
    env_file: Any = None,
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

        env = _build_aider_env(model, env_file=env_file)
        env.setdefault("AIDER_CHECK_UPDATE", "false")
        env.setdefault("AIDER_SHOW_RELEASE_NOTES", "false")

        with tempfile.TemporaryDirectory(prefix="hermes-aider-") as state_dir:
            command = _build_aider_command(
                instruction,
                model,
                executable=executable,
                state_dir=Path(state_dir),
            )
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
            "env_file": {
                "type": "string",
                "description": "Optional dotenv file used for OpenRouter credentials. Defaults to ~/.hermes/.env.",
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
        env_file=args.get("env_file"),
    )


registry.register(
    name="aider_subagent",
    toolset="aider",
    schema=AIDER_SUBAGENT_SCHEMA,
    handler=_handle_aider_subagent,
    max_result_size_chars=100_000,
)