#!/usr/bin/env python3
"""PR-Agent wrapper tool for GitHub pull-request triage."""

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
_DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-flash"
_DEFAULT_TIMEOUT_SECONDS = 900
_DEFAULT_MAX_TOKENS = 32000
_MAX_OUTPUT_CHARS = 12000
_ALLOWED_ACTIONS = {
    "ask",
    "ask_question",
    "review",
    "review_pr",
    "describe",
    "describe_pr",
    "generate_labels",
}
_BASE_ENV_KEYS = {
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
}


def _normalize_model(model: Any = None) -> str:
    """Return an OpenRouter-qualified model name."""
    model_name = str(model or _DEFAULT_MODEL).strip()
    if not model_name:
        model_name = _DEFAULT_MODEL
    if model_name.startswith(_OPENROUTER_PREFIX):
        return model_name
    return f"{_OPENROUTER_PREFIX}{model_name}"


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Coerce common truthy and falsy strings to bool."""
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("publish_output must be a boolean")


def _coerce_timeout(timeout_seconds: Any = None) -> int:
    """Return a bounded timeout for PR-Agent subprocesses."""
    if timeout_seconds in (None, ""):
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = int(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be an integer") from exc
    return max(30, timeout)


def _coerce_max_tokens(max_tokens: Any = None) -> int:
    """Return a positive token budget for unknown LiteLLM model ids."""
    if max_tokens in (None, ""):
        return _DEFAULT_MAX_TOKENS
    try:
        token_count = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_tokens must be an integer") from exc
    if token_count <= 0:
        raise ValueError("max_tokens must be positive")
    return token_count


def _resolve_workdir(workdir: Any = None) -> Path:
    """Resolve the directory where PR-Agent should run."""
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


def _resolve_pr_agent_executable() -> Optional[str]:
    """Resolve the PR-Agent executable from PR_AGENT_BIN or PATH."""
    configured = os.getenv("PR_AGENT_BIN", "").strip()
    if configured:
        return shutil.which(configured) or configured
    return shutil.which("pr-agent")


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


def _run_gh_auth_token() -> Optional[str]:
    """Return the GitHub CLI token without exposing it in command output."""
    gh = shutil.which("gh")
    if not gh:
        return None
    completed = subprocess.run(
        [gh, "auth", "token"],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        return None
    token = completed.stdout.strip()
    return token or None


def _first_present(*values: Optional[str]) -> Optional[str]:
    """Return the first non-empty string from a list of candidates."""
    for value in values:
        if value:
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _build_pr_agent_env(model: str, env_file: Any = None) -> dict[str, str]:
    """Build a minimal PR-Agent environment with GitHub and model auth."""
    file_values = _load_env_file(env_file)
    env = {key: os.environ[key] for key in _BASE_ENV_KEYS if os.environ.get(key)}

    github_token = _first_present(
        os.getenv("GITHUB__USER_TOKEN"),
        os.getenv("GITHUB_TOKEN"),
        os.getenv("GH_TOKEN"),
        file_values.get("GITHUB__USER_TOKEN"),
        file_values.get("GITHUB_TOKEN"),
        file_values.get("GH_TOKEN"),
    ) or _run_gh_auth_token()
    if github_token:
        env["GITHUB__USER_TOKEN"] = github_token

    if model.startswith(_OPENROUTER_PREFIX):
        openrouter_token = _first_present(
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_KEY"),
            file_values.get("OPENROUTER_API_KEY"),
            file_values.get("OPENROUTER_KEY"),
        )
        if openrouter_token:
            env["OPENROUTER_API_KEY"] = openrouter_token
            env["OPENAI_API_KEY"] = openrouter_token
    else:
        openai_token = _first_present(
            os.getenv("OPENAI_API_KEY"),
            file_values.get("OPENAI_API_KEY"),
        )
        if openai_token:
            env["OPENAI_API_KEY"] = openai_token

    return env


def _validate_pr_agent_env(env: dict[str, str], model: str) -> Optional[str]:
    """Return an actionable error if required credentials are missing."""
    if not env.get("GITHUB__USER_TOKEN"):
        return "GitHub token not found. Configure GITHUB__USER_TOKEN, GITHUB_TOKEN, GH_TOKEN, or gh auth."
    if model.startswith(_OPENROUTER_PREFIX) and not env.get("OPENROUTER_API_KEY"):
        return "OpenRouter token not found. Configure OPENROUTER_API_KEY in the environment or ~/.hermes/.env."
    if not model.startswith(_OPENROUTER_PREFIX) and not env.get("OPENAI_API_KEY"):
        return "OpenAI-compatible token not found. Configure OPENAI_API_KEY or use an OpenRouter model."
    return None


def _build_pr_agent_command(
    pr_url: Any,
    action: Any = "review",
    question: Any = None,
    *,
    publish_output: Any = False,
    model: Any = None,
    max_tokens: Any = None,
    config_path: Any = None,
    executable: str = "pr-agent",
) -> list[str]:
    """Build a non-interactive PR-Agent command."""
    pr_url_text = str(pr_url or "").strip()
    if not pr_url_text:
        raise ValueError("pr_url is required")

    action_name = str(action or "review").strip().lower()
    if action_name not in _ALLOWED_ACTIONS:
        allowed = ", ".join(sorted(_ALLOWED_ACTIONS))
        raise ValueError(f"Unsupported PR-Agent action '{action_name}'. Allowed: {allowed}")

    command = [executable, f"--pr_url={pr_url_text}", action_name]

    if action_name in {"ask", "ask_question"}:
        question_text = str(question or "").strip()
        if not question_text:
            raise ValueError("question is required when action is ask")
        command.append(question_text)

    normalized_model = _normalize_model(model)
    token_budget = _coerce_max_tokens(max_tokens)
    config_path_text = str(config_path or "").strip()
    if config_path_text:
        command.append(f"--config_path={config_path_text}")
    command.extend(
        [
            f"--config.publish_output={str(_coerce_bool(publish_output)).lower()}",
            f"--config.model={normalized_model}",
            "--config.fallback_models=[]",
            f"--config.custom_model_max_tokens={token_budget}",
            f"--config.max_model_tokens={token_budget}",
            "--config.temperature=0.2",
        ]
    )
    return command


def _command_preview(command: list[str]) -> str:
    """Return a shell-style command string without credentials."""
    return shlex.join(command)


def _tail_text(value: Any, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Return a bounded text tail suitable for a tool result."""
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def pr_agent(
    pr_url: Any,
    action: Any = "review",
    question: Any = None,
    publish_output: Any = False,
    model: Any = None,
    workdir: Any = None,
    timeout_seconds: Any = None,
    max_tokens: Any = None,
    config_path: Any = None,
    env_file: Any = None,
) -> str:
    """Run PR-Agent against a GitHub pull request and return a JSON result."""
    started = time.monotonic()
    try:
        cwd = _resolve_workdir(workdir)
        timeout = _coerce_timeout(timeout_seconds)
        executable = _resolve_pr_agent_executable()
        command = _build_pr_agent_command(
            pr_url,
            action,
            question,
            publish_output=publish_output,
            model=model,
            max_tokens=max_tokens,
            config_path=config_path,
            executable=executable or "pr-agent",
        )
        if not executable:
            return json.dumps(
                {
                    "status": "error",
                    "error": "PR-Agent executable not found. Install PR-Agent or set PR_AGENT_BIN.",
                    "command": _command_preview(command),
                    "workdir": str(cwd),
                },
                ensure_ascii=False,
            )

        model_name = _normalize_model(model)
        env = _build_pr_agent_env(model_name, env_file=env_file)
        env_error = _validate_pr_agent_env(env, model_name)
        if env_error:
            return json.dumps(
                {
                    "status": "error",
                    "error": env_error,
                    "command": _command_preview(command),
                    "model": model_name,
                    "workdir": str(cwd),
                },
                ensure_ascii=False,
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
                "model": model_name,
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
                "error": f"PR-Agent timed out after {_coerce_timeout(timeout_seconds)} seconds",
                "duration_seconds": round(time.monotonic() - started, 2),
                "stdout_tail": _tail_text(exc.stdout),
                "stderr_tail": _tail_text(exc.stderr),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("PR-Agent tool failed: %s", exc)
        return json.dumps(
            {
                "status": "error",
                "error": str(exc),
                "duration_seconds": round(time.monotonic() - started, 2),
            },
            ensure_ascii=False,
        )


PR_AGENT_SCHEMA = {
    "name": "pr_agent",
    "description": (
        "Run PR-Agent for GitHub pull-request review, description, labels, or "
        "question answering. Use this before delegating code edits to Aider: "
        "PR-Agent analyzes the pull request; Aider applies follow-up changes "
        "only when the review finds actionable work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pr_url": {
                "type": "string",
                "description": "GitHub pull request URL.",
            },
            "action": {
                "type": "string",
                "description": "PR-Agent action: ask, review, describe, or generate_labels.",
                "default": "review",
            },
            "question": {
                "type": "string",
                "description": "Question to ask about the PR when action is ask.",
            },
            "publish_output": {
                "type": "boolean",
                "description": "Whether PR-Agent should publish output back to GitHub. Defaults to false for dry-run triage.",
                "default": False,
            },
            "model": {
                "type": "string",
                "description": "Optional model name, with or without the openrouter/ prefix.",
            },
            "workdir": {
                "type": "string",
                "description": "Optional repository directory. Defaults to TERMINAL_CWD or current directory.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional subprocess timeout in seconds. Defaults to 900.",
                "minimum": 30,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for custom LiteLLM/OpenRouter model ids. Defaults to 32000.",
                "minimum": 1,
            },
            "config_path": {
                "type": "string",
                "description": "Optional PR-Agent config file path.",
            },
            "env_file": {
                "type": "string",
                "description": "Optional dotenv file used for OPENROUTER_API_KEY or GitHub tokens. Defaults to ~/.hermes/.env.",
            },
        },
        "required": ["pr_url"],
    },
}


def _handle_pr_agent(args: dict, **_kwargs) -> str:
    """Registry handler for the PR-Agent tool."""
    return pr_agent(
        pr_url=args.get("pr_url"),
        action=args.get("action", "review"),
        question=args.get("question"),
        publish_output=args.get("publish_output", False),
        model=args.get("model"),
        workdir=args.get("workdir"),
        timeout_seconds=args.get("timeout_seconds"),
        max_tokens=args.get("max_tokens"),
        config_path=args.get("config_path"),
        env_file=args.get("env_file"),
    )


registry.register(
    name="pr_agent",
    toolset="pr_agent",
    schema=PR_AGENT_SCHEMA,
    handler=_handle_pr_agent,
    max_result_size_chars=100_000,
)