"""Authenticated Telegram -> Threadwire coding-worker launch boundary.

This tool is intentionally separate from ``terminal``.  It accepts structured
worker inputs, derives the delivery target only from the current gateway
ContextVars, and invokes the fixed Threadwire launcher with an argv list.
Threadwire alone loads Telegram credentials and delivers worker output.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from agent.runtime_cwd import resolve_agent_cwd
from gateway.session_context import get_bound_session_env
from tools.environments.local import hermes_subprocess_env
from tools.interrupt import is_interrupted
from tools.registry import registry


THREADWIRE_EXECUTABLE = "/opt/data/bin/threadwire"
SUPPORTED_PROVIDERS = frozenset({"codex", "claude", "opencode"})
_CHAT_ID_RE = re.compile(r"-?[1-9][0-9]*\Z")
_THREAD_ID_RE = re.compile(r"[1-9][0-9]*\Z")
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,511}\Z")
_FORBIDDEN_PROVIDER_OPTIONS = ("--target",)
_ALLOWED_ARGUMENTS = frozenset({
    "provider",
    "prompt",
    "prompt_file",
    "stdin",
    "cwd",
    "timeout",
    "process_number",
    "max_output_length",
    "resume_session",
    "provider_arguments",
})
_POLL_SECONDS = 0.1
_TERMINATE_GRACE_SECONDS = 3.0
_DEFAULT_TIMEOUT_SECONDS = 3600
_MAX_TIMEOUT_SECONDS = 86400
_STDIN_STAGE_CHARS = 64 * 1024


def check_threadwire_requirements() -> bool:
    """Return whether the fixed launcher is installed and executable."""
    return os.path.isfile(THREADWIRE_EXECUTABLE) and os.access(
        THREADWIRE_EXECUTABLE, os.X_OK
    )


def _error(reason: str, *, status: str = "blocked") -> str:
    # Deliberately generic: never put IDs, targets, argv, prompts, paths,
    # provider output, or exception text into a Telegram-bound tool result.
    return json.dumps({"status": status, "error": reason}, ensure_ascii=False)


def _success(returncode: int) -> str:
    return json.dumps(
        {
            "status": "completed" if returncode == 0 else "failed",
            "exit_code": returncode,
        },
        ensure_ascii=False,
    )


def _positive_integer(value: Any, name: str) -> int:
    # bool is an int subclass; accepting it would silently turn True into 1.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid {name}")
    if value > 2**53 - 1:
        raise ValueError(f"Invalid {name}")
    return value


def _validated_target() -> str:
    platform = get_bound_session_env("HERMES_SESSION_PLATFORM", "")
    if platform != "telegram":
        raise ValueError("This worker launch requires an authenticated Telegram session")

    chat_id = get_bound_session_env("HERMES_SESSION_CHAT_ID", "")
    if not isinstance(chat_id, str) or not _CHAT_ID_RE.fullmatch(chat_id):
        raise ValueError("Authenticated Telegram chat context is invalid")

    thread_id = get_bound_session_env("HERMES_SESSION_THREAD_ID", "")
    if thread_id is None:
        thread_id = ""
    if not isinstance(thread_id, str):
        raise ValueError("Authenticated Telegram topic context is invalid")
    if thread_id and not _THREAD_ID_RE.fullmatch(thread_id):
        raise ValueError("Authenticated Telegram topic context is invalid")
    if thread_id and int(thread_id) > 2**53 - 1:
        # Threadwire converts Telegram topic IDs to JavaScript Number and
        # rejects values outside the exact-integer range. Mirror that check so
        # malformed authenticated context is rejected before process creation.
        raise ValueError("Authenticated Telegram topic context is invalid")

    return f"telegram:{chat_id}" + (f":{thread_id}" if thread_id else "")


def _validated_path(value: Any, *, name: str, directory: bool) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Invalid {name}")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Invalid {name}")
    if directory and not path.is_dir():
        raise ValueError(f"Invalid {name}")
    if not directory and not path.is_file():
        raise ValueError(f"Invalid {name}")
    return str(path)


def _validated_provider_arguments(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Invalid provider arguments")
    result: list[str] = []
    for argument in value:
        if not isinstance(argument, str) or "\x00" in argument:
            raise ValueError("Invalid provider arguments")
        if any(
            argument == option or argument.startswith(f"{option}=")
            for option in _FORBIDDEN_PROVIDER_OPTIONS
        ):
            raise ValueError("Caller-supplied delivery targets are forbidden")
        result.append(argument)
    return result


def _build_invocation(args: dict[str, Any]) -> tuple[list[str], str, str | None, int]:
    if not isinstance(args, dict):
        raise ValueError("Invalid worker request")
    unknown = set(args) - _ALLOWED_ARGUMENTS
    if unknown:
        if unknown & {"target", "chat_id", "thread_id"}:
            raise ValueError("Caller-supplied delivery targets are forbidden")
        raise ValueError("Unsupported worker option")

    # Resolve the authenticated target before doing filesystem work and before
    # process creation.  No tool argument can participate in this value.
    target = _validated_target()

    provider = args.get("provider")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError("Invalid coding-worker provider")

    prompt_fields = [
        name for name in ("prompt", "prompt_file", "stdin") if args.get(name) is not None
    ]
    if len(prompt_fields) != 1:
        raise ValueError("Exactly one prompt source is required")

    stdin_text: str | None = None
    prompt_argv: list[str]
    prompt_field = prompt_fields[0]
    prompt_value = args[prompt_field]
    if prompt_field == "prompt":
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ValueError("Prompt cannot be empty")
        stdin_text = prompt_value
        prompt_argv = []
    elif prompt_field == "prompt_file":
        prompt_path = _validated_path(prompt_value, name="prompt file", directory=False)
        prompt_argv = ["--prompt-file", prompt_path]
    else:
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ValueError("Prompt cannot be empty")
        stdin_text = prompt_value
        prompt_argv = []

    cwd_value = args.get("cwd")
    if cwd_value is None:
        cwd = str(resolve_agent_cwd())
        if not Path(cwd).is_dir():
            raise ValueError("Invalid working directory")
    else:
        cwd = _validated_path(cwd_value, name="working directory", directory=True)

    timeout_value = args.get("timeout", _DEFAULT_TIMEOUT_SECONDS)
    timeout = _positive_integer(timeout_value, "timeout")
    if timeout > _MAX_TIMEOUT_SECONDS:
        raise ValueError("Invalid timeout")

    argv = [
        THREADWIRE_EXECUTABLE,
        "run",
        "--provider",
        provider,
        "--target",
        target,
        "--cwd",
        cwd,
    ]

    for key, option in (
        ("process_number", "--process-number"),
        ("max_output_length", "--max-output-length"),
    ):
        if args.get(key) is not None:
            argv.extend([option, str(_positive_integer(args[key], key.replace("_", " ")))])

    resume_session = args.get("resume_session")
    if resume_session is not None:
        if not isinstance(resume_session, str) or not _SESSION_ID_RE.fullmatch(resume_session):
            raise ValueError("Invalid provider session")
        argv.extend(["--resume-session", resume_session])

    argv.extend(prompt_argv)
    provider_arguments = _validated_provider_arguments(args.get("provider_arguments"))
    if provider_arguments:
        argv.extend(["--", *provider_arguments])
    return argv, cwd, stdin_text, timeout


def _child_environment() -> dict[str, str]:
    # Threadwire is a model-driving CLI, so preserve provider auth according to
    # the existing audited convention.  The shared helper always strips the
    # gateway Telegram token; remove Threadwire's private names defensively too.
    env = hermes_subprocess_env(
        inherit_credentials=True,
        exclude_keys=frozenset({
            "TELEGRAM_BOT_TOKEN",
            "THREADWIRE_TELEGRAM_BOT_TOKEN",
            "THREADWIRE_VALIDATE_ONLY",
        }),
    )
    # Threadwire routing is carried exclusively by the validated --target.
    # The shared helper bridges task-local gateway context for child processes,
    # but none of that metadata belongs across this provider boundary.
    for name in tuple(env):
        if name.startswith("HERMES_SESSION_"):
            del env[name]
    return env


def _reap_process(proc: subprocess.Popen) -> None:
    """Reap the direct child without allowing an unbounded wait."""
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except (subprocess.TimeoutExpired, OSError, ProcessLookupError):
        pass


def _stop_process(proc: subprocess.Popen, initial_signal: signal.Signals) -> None:
    if proc.poll() is None:
        try:
            # start_new_session=True makes the child's PID its process-group
            # ID. Signal the whole worker tree so provider descendants receive
            # the same graceful cancellation/deadline notification.
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, initial_signal)
            else:
                proc.send_signal(initial_signal)
        except (AttributeError, OSError, ProcessLookupError):
            try:
                proc.send_signal(initial_signal)
            except (OSError, ProcessLookupError):
                pass
        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
        if proc.poll() is None:
            try:
                # The child starts a new session. Kill the group only after
                # Threadwire has had a bounded opportunity to handle the first
                # signal and forward it to its provider.
                os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except (AttributeError, OSError, ProcessLookupError):
                try:
                    proc.kill()
                except (OSError, ProcessLookupError):
                    pass
    _reap_process(proc)


def _stage_stdin(stdin_text: str, deadline: float):
    """Stage private stdin without a child pipe or an unbounded copy."""
    staged = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    try:
        for offset in range(0, len(stdin_text), _STDIN_STAGE_CHARS):
            if is_interrupted():
                raise InterruptedError
            if time.monotonic() >= deadline:
                raise TimeoutError
            staged.write(stdin_text[offset : offset + _STDIN_STAGE_CHARS])
        staged.seek(0)
        return staged
    except BaseException:
        staged.close()
        raise


def launch_telegram_coding_worker(args: dict[str, Any]) -> str:
    """Validate and synchronously supervise one Threadwire worker."""
    proc: subprocess.Popen | None = None
    staged_stdin = None
    try:
        # Authenticate from task-local gateway context before probing or
        # starting Threadwire. Process environment values are not identity.
        _validated_target()
        argv, cwd, stdin_text, timeout = _build_invocation(args)
        if not check_threadwire_requirements():
            return _error("Telegram coding-worker service is unavailable", status="error")
        deadline = time.monotonic() + timeout
        if stdin_text is not None:
            staged_stdin = _stage_stdin(stdin_text, deadline)
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=cwd,
            env=_child_environment(),
            stdin=staged_stdin if staged_stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        while True:
            returncode = proc.poll()
            if returncode is not None:
                return _success(returncode)
            if is_interrupted():
                _stop_process(proc, signal.SIGINT)
                return _error("Coding-worker launch was cancelled", status="cancelled")
            if time.monotonic() >= deadline:
                _stop_process(proc, signal.SIGTERM)
                return _error("Coding-worker launch timed out", status="timed_out")
            time.sleep(_POLL_SECONDS)
    except InterruptedError:
        return _error("Coding-worker launch was cancelled", status="cancelled")
    except TimeoutError:
        return _error("Coding-worker launch timed out", status="timed_out")
    except (OSError, ValueError):
        return _error("Telegram coding-worker launch was rejected")
    except Exception:
        return _error("Telegram coding-worker launch failed", status="error")
    finally:
        if proc is not None:
            _stop_process(proc, signal.SIGTERM)
        if staged_stdin is not None:
            staged_stdin.close()


THREADWIRE_SCHEMA = {
    "name": "telegram_coding_worker",
    "description": (
        "Launch a coding worker for the current authenticated Telegram chat/topic. "
        "The delivery target is derived internally and cannot be supplied. Use this "
        "instead of terminal-based Codex, Claude, or OpenCode launch commands when "
        "the current conversation originates from Telegram."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "provider": {"type": "string", "enum": sorted(SUPPORTED_PROVIDERS)},
            "prompt": {"type": "string", "description": "Direct worker prompt."},
            "prompt_file": {"type": "string", "description": "Absolute prompt file path."},
            "stdin": {"type": "string", "description": "Worker prompt supplied on stdin."},
            "cwd": {"type": "string", "description": "Existing absolute worker directory."},
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_TIMEOUT_SECONDS,
                "default": _DEFAULT_TIMEOUT_SECONDS,
            },
            "process_number": {"type": "integer", "minimum": 1},
            "max_output_length": {"type": "integer", "minimum": 1},
            "resume_session": {"type": "string"},
            "provider_arguments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments passed to the selected provider after --.",
            },
        },
        "required": ["provider"],
        "oneOf": [
            {"required": ["prompt"]},
            {"required": ["prompt_file"]},
            {"required": ["stdin"]},
        ],
    },
}


def _handle_threadwire(args: dict[str, Any], **_kwargs: Any) -> str:
    return launch_telegram_coding_worker(args)


registry.register(
    name="telegram_coding_worker",
    toolset="terminal",
    schema=THREADWIRE_SCHEMA,
    handler=_handle_threadwire,
    emoji="🧵",
    max_result_size_chars=2_000,
)
