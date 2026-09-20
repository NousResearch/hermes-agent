"""Open Code Review (https://open-codereview.ai) bridge: run the ``ocr`` CLI and keep it pinned
to the model Hermes is currently using.

Two things the CLI can do on its own but not together: review a diff, and know which provider
Hermes has selected. Without a bridge the user re-runs ``ocr config provider ...`` by hand every
time they switch models or rely on auto model selection (#108639). So ``review`` mirrors Hermes's
active provider/model into the CLI first (recorded under ``HERMES_HOME`` so an unchanged
assignment is not re-pushed), and ``sync`` does that on demand.

The OCR CLI's own flags are not Hermes's to guess: ``provider_args`` / ``model_args`` /
``review_args`` are argv templates in config.yaml with ``{provider}`` / ``{model}`` / ``{target}``
placeholders, so a CLI whose surface differs is adapted in config, not in code. The API key is
handed to the child process through ``api_key_env`` only — never on a command line or in OCR's
own config file.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DEFAULT_COMMAND = "ocr"
DEFAULT_API_KEY_ENV = "OPEN_CODEREVIEW_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_PROVIDER_ARGS = ["config", "provider", "{provider}"]
DEFAULT_MODEL_ARGS = ["config", "model", "{model}"]
DEFAULT_REVIEW_ARGS = ["review", "{target}"]
ACTIONS = ("review", "sync", "status")
#: ``auto`` re-syncs whenever the assignment Hermes is using changed; ``manual`` only on ``sync``.
MODES = ("auto", "manual")
STATE_FILENAME = "open_codereview.json"


# ---- config -------------------------------------------------------------------------------


def _load_section() -> Dict[str, Any]:
    """The ``open_codereview:`` section, or ``{}`` when absent/malformed."""
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("open_codereview")
    except Exception as exc:  # a broken config must not take the tool down
        logger.debug("open_codereview: config load failed: %s", exc)
        return {}
    return section if isinstance(section, dict) else {}


def _text(section: Dict[str, Any], key: str, default: str = "") -> str:
    """Trimmed non-empty string value from *section*, else *default*."""
    value = section.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _timeout(section: Dict[str, Any]) -> float:
    try:
        value = float(section.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return float(DEFAULT_TIMEOUT_SECONDS)
    return value if value > 0 else float(DEFAULT_TIMEOUT_SECONDS)


def _args_template(section: Dict[str, Any], key: str, default: Sequence[str]) -> List[str]:
    """Argv template (list of tokens) for *key*; a plain string is split, anything else is ignored."""
    raw = section.get(key)
    if isinstance(raw, (list, tuple)):
        tokens = [str(token) for token in raw]
        return tokens if any(token.strip() for token in tokens) else list(default)
    if isinstance(raw, str) and raw.strip():
        try:
            tokens = shlex.split(raw)
        except ValueError:
            return list(default)
        return tokens or list(default)
    return list(default)


def _render(tokens: Sequence[str], values: Dict[str, str]) -> List[str]:
    """Substitute ``{name}`` placeholders; unknown braces and values stay literal."""
    rendered: List[str] = []
    for token in tokens:
        for name, value in values.items():
            token = token.replace("{" + name + "}", value)
        rendered.append(token)
    return rendered


def _resolve_executable(command: str) -> Optional[str]:
    """Absolute path to *command*, or None when it is neither on PATH nor an existing file."""
    if not command:
        return None
    if os.sep in command or (os.altsep and os.altsep in command):
        path = Path(command).expanduser()
        return str(path) if path.is_file() else None
    return shutil.which(command)


def _credential_env_names(section: Dict[str, Any]) -> List[str]:
    """Parent-env names forwarded to the CLI (its own env is otherwise scrubbed of Hermes secrets)."""
    raw = section.get("env_passthrough")
    names = [str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, (list, tuple)) else []
    key_env = _text(section, "api_key_env", DEFAULT_API_KEY_ENV)
    return names if key_env in names else names + [key_env]


def _child_env(section: Dict[str, Any]) -> Dict[str, str]:
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    for name in _credential_env_names(section):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


# ---- the assignment Hermes is using --------------------------------------------------------


def _scalar(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """``model:`` as a mapping. Older configs store the model name as a bare string."""
    section = config.get("model")
    return section if isinstance(section, dict) else {}


def _resolved_provider(configured: str) -> str:
    """``auto`` (or unset) resolves through the canonical auth resolver. Empty when nothing is
    resolvable, so the literal ``"auto"`` is never pushed into the review tool's configuration."""
    if configured and configured.lower() != "auto":
        return configured
    try:
        from hermes_cli.auth import resolve_provider

        return _scalar(resolve_provider() or "")
    except Exception as exc:
        logger.debug("open_codereview: provider auto-resolution failed: %s", exc)
        return ""


def _configured_model(config: Dict[str, Any], model_cfg: Dict[str, Any]) -> str:
    """``model.default`` (mapping form), the same mapping's ``model``/``name`` aliases, or a legacy
    bare ``model: <name>`` string."""
    for value in (model_cfg.get("default"), model_cfg.get("model"), model_cfg.get("name")):
        if configured := _scalar(value):
            return configured
    return _scalar(config.get("model"))


def hermes_assignment() -> Tuple[str, str]:
    """``(provider, model)`` Hermes is currently configured to use — what the OCR CLI must mirror."""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception as exc:
        logger.debug("open_codereview: config load failed: %s", exc)
        config = {}
    model_cfg = _model_config(config)
    provider = _scalar(model_cfg.get("provider"))
    model = _configured_model(config, model_cfg) or _scalar(os.environ.get("HERMES_MODEL"))
    return _resolved_provider(provider), model


# ---- state ---------------------------------------------------------------------------------


def _state_path() -> Path:
    return get_hermes_home() / STATE_FILENAME


def _read_state() -> Dict[str, Any]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(provider: str, model: str) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"provider": provider, "model": model, "synced_at": time.time()}, indent=2) + "\n",
            encoding="utf-8")
    except OSError as exc:
        # The next call re-syncs instead of trusting the state; never fail a review over this.
        logger.warning("open_codereview: could not record sync state at %s: %s", path, exc)


def _sync_pending(state: Dict[str, Any], provider: str, model: str) -> bool:
    """Whether the recorded assignment differs from the one Hermes is using now."""
    if not provider and not model:
        return False
    return (_scalar(state.get("provider")), _scalar(state.get("model"))) != (provider, model)


# ---- process execution ---------------------------------------------------------------------


def _run(argv: List[str], timeout: float, env: Dict[str, str]) -> subprocess.CompletedProcess:
    """Run *argv* to completion; a stalled CLI is killed with its process tree, not just the child."""
    from tools.tts_command_provider import terminate_command_process_tree

    group = ({"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)} if os.name == "nt"
             else {"start_new_session": True})
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, env=env, **group)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_command_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = "", ""
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, argv, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def _display(argv: Sequence[str]) -> str:
    """Readable command line for a result body — no secret ever reaches argv, so it is safe to show."""
    return subprocess.list2cmdline(list(argv)) if os.name == "nt" else shlex.join(argv)


def _sync_jobs(section: Dict[str, Any], provider: str, model: str) -> List[Tuple[str, str, List[str]]]:
    """``(field, value, argv)`` for each value that has a configured command; commands first."""
    command = _text(section, "command", DEFAULT_COMMAND)
    values = {"provider": provider, "model": model, "target": ""}
    jobs: List[Tuple[str, str, List[str]]] = []
    if provider:
        jobs.append(("provider", provider, [command] + _render(
            _args_template(section, "provider_args", DEFAULT_PROVIDER_ARGS), values)))
    if model:
        jobs.append(("model", model, [command] + _render(
            _args_template(section, "model_args", DEFAULT_MODEL_ARGS), values)))
    return jobs


def _push_assignment(section: Dict[str, Any], provider: str, model: str, env: Dict[str, str]) -> List[Dict[str, str]]:
    """Run the provider/model commands; returns one entry per applied field."""
    applied: List[Dict[str, str]] = []
    for field, value, argv in _sync_jobs(section, provider, model):
        _run(argv, _timeout(section), env)
        applied.append({"field": field, "value": value, "command": _display(argv)})
    if applied:
        _write_state(provider, model)
    return applied


# ---- actions -------------------------------------------------------------------------------


def _status(section: Dict[str, Any], provider: str, model: str) -> str:
    key_env = _text(section, "api_key_env", DEFAULT_API_KEY_ENV)
    state = _read_state()
    return json.dumps({
        "success": True, "action": "status",
        "command": _text(section, "command", DEFAULT_COMMAND),
        "command_path": _resolve_executable(_text(section, "command", DEFAULT_COMMAND)),
        "mode": _mode(section),
        "hermes": {"provider": provider, "model": model},
        "cli": {"provider": _scalar(state.get("provider")), "model": _scalar(state.get("model")),
                "synced_at": state.get("synced_at")},
        "sync_pending": _sync_pending(state, provider, model),
        "api_key_env": key_env, "api_key_present": bool(os.environ.get(key_env)),
    }, ensure_ascii=False)


def _mode(section: Dict[str, Any]) -> str:
    mode = _text(section, "mode", "auto").lower()
    return mode if mode in MODES else "auto"


def _sync(section: Dict[str, Any], provider: str, model: str, env: Dict[str, str]) -> str:
    applied = _push_assignment(section, provider, model, env)
    return json.dumps({
        "success": True, "action": "sync", "synced": bool(applied),
        "hermes": {"provider": provider, "model": model},
        "applied": applied,
        "message": "" if applied else (
            "Nothing to sync: no provider/model resolved from config.yaml, or the matching "
            "open_codereview.provider_args / model_args template is empty."),
    }, ensure_ascii=False)


def _review(section: Dict[str, Any], target: str, provider: str, model: str,
            env: Dict[str, str]) -> str:
    # The point of the integration: a model switch in Hermes propagates here without a terminal.
    synced: List[Dict[str, str]] = []
    if _mode(section) == "auto" and _sync_pending(_read_state(), provider, model):
        synced = _push_assignment(section, provider, model, env)
    command = _text(section, "command", DEFAULT_COMMAND)
    argv = [command] + _render(_args_template(section, "review_args", DEFAULT_REVIEW_ARGS),
                               {"target": target, "provider": provider, "model": model})
    proc = _run(argv, _timeout(section), env)
    return json.dumps({
        "success": True, "action": "review",
        "command": _display(argv), "target": target,
        "provider": provider, "model": model, "synced": synced,
        "output": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip() or None,
    }, ensure_ascii=False)


def open_codereview_tool(action: str = "status", target: str = "", provider: str = "",
                         model: str = "") -> str:
    """Run one ``open_codereview`` action; always a JSON string (repo tool contract)."""
    action = (action or "status").strip().lower()
    if action not in ACTIONS:
        return tool_error(f"Unknown action {action!r}. Use one of: {', '.join(ACTIONS)}.")
    section = _load_section()
    command = _text(section, "command", DEFAULT_COMMAND)
    if _resolve_executable(command) is None:
        return tool_error(
            f"Open Code Review CLI {command!r} is not installed or not on PATH. Install the "
            "open-codereview.ai CLI, or set `open_codereview.command` in config.yaml to its "
            "executable (see https://open-codereview.ai).")

    target, override_provider, override_model = target.strip(), provider.strip(), model.strip()
    if action == "review" and not target:
        return tool_error(
            "`target` is required for action='review' (for example 'HEAD~1..HEAD' or a path).")

    try:
        hermes_provider, hermes_model = hermes_assignment()
        provider, model = override_provider or hermes_provider, override_model or hermes_model
        if action == "status":
            return _status(section, provider, model)
        env = _child_env(section)
        if action == "sync":
            return _sync(section, provider, model, env)
        return _review(section, target, provider, model, env)
    except subprocess.TimeoutExpired as exc:
        return tool_error(
            f"Open Code Review command timed out after {_timeout(section):g}s: {_display(exc.cmd)}")
    except subprocess.CalledProcessError as exc:
        detail = "; ".join(
            f"{stream}: {text.strip()}" for stream, text in (("stderr", exc.stderr), ("stdout", exc.stdout)) if text
        ) or "no output"
        return tool_error(
            f"Open Code Review command {_display(exc.cmd)} exited with code {exc.returncode}: {detail}")
    except Exception as exc:
        logger.error("open_codereview %s failed: %s", action, exc, exc_info=True)
        return tool_error(str(exc))


def check_open_codereview_requirements() -> bool:
    """True when the configured OCR CLI resolves — reachability only, never a network probe."""
    return _resolve_executable(_text(_load_section(), "command", DEFAULT_COMMAND)) is not None


OPEN_CODEREVIEW_SCHEMA = {
    "name": "open_codereview",
    "description": (
        "Run Open Code Review (open-codereview.ai) through its local command line tool and keep "
        "that tool pinned to the model Hermes is currently using. Actions: 'review' reviews "
        "`target` and returns the report into this conversation; 'sync' pushes Hermes's current "
        "provider/model into the Open Code Review configuration; 'status' reports both sides and "
        "whether a sync is pending. Use this instead of configuring the review tool by hand in a "
        "terminal. Available when the Open Code Review command is installed and configured "
        "(config.yaml `open_codereview.command`)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(ACTIONS),
                "default": "status",
                "description": "One of: review, sync, status. Defaults to status.",
            },
            "target": {
                "type": "string",
                "description": (
                    "What the review covers: a revision range such as 'HEAD~1..HEAD', a single "
                    "commit, or a path. Required for action='review'."
                ),
            },
            "provider": {
                "type": "string",
                "description": "Provider to push to the review tool; defaults to Hermes's active provider.",
            },
            "model": {
                "type": "string",
                "description": "Model to push to the review tool; defaults to Hermes's active model.",
            },
        },
        "required": [],
    },
}


registry.register(
    name="open_codereview", toolset="open_codereview", schema=OPEN_CODEREVIEW_SCHEMA,
    handler=lambda args, **kw: open_codereview_tool(
        action=args.get("action", ""), target=args.get("target", ""),
        provider=args.get("provider", ""), model=args.get("model", "")),
    check_fn=check_open_codereview_requirements, emoji="🔎", max_result_size_chars=200_000,
)
