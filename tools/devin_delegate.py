"""Devin CLI delegate backend.

Exposes the local ``devin`` CLI binary as a first-class delegation backend,
peer to ``delegate_task``. Unlike ``delegate_task`` (which spawns an in-process
``AIAgent`` child against a chat-completions inference endpoint), this tool
shells out to the external Devin CLI in non-interactive ``-p`` (print) mode
and returns the finished response. Devin runs its *own* internal agent loop
and hands back a completed answer — there is no chat-completions token stream
for Hermes's conversation loop to consume, so it cannot be a ``ProviderProfile``.

This is the CLI-delegate shape the user asked for: Hermes keeps its own
agent loop; Devin handles a delegated, self-contained subtask end-to-end.

Gating (Footprint Ladder rung 3 — service-gated tool):
  ``check_devin_requirements()`` returns True only when
  ``delegation.devin.enabled`` is truthy in config.yaml AND the ``devin``
  binary is on ``$PATH``. The registry TTL-caches ``check_fn`` results, so
  the tool has ZERO schema footprint for users who have not opted in — it
  never appears in the tool list sent to the model on every API call unless
  Devin is actually configured. Authentication is verified at handler time
  (not in ``check_fn``) so a stale Devin login produces a clear, actionable
  error instead of the tool silently disappearing from the schema.

Config (``delegation.devin.*`` in config.yaml — behavioral settings live in
config, not env vars, per the AGENTS.md ``.env``-is-for-secrets-only rule)::

    delegation:
      devin:
        enabled: true                  # opt-in; default false
        model: swe                      # optional Devin model short name
        permission_mode: dangerous      # auto|accept-edits|smart|dangerous
        timeout_seconds: 1800           # per-call wall-clock cap (>= 60)
        max_result_chars: 20000         # truncate Devin stdout above this

The handler is synchronous (blocking) by design for v1: the model receives
Devin's full result within the same turn. The async re-entry machinery that
``delegate_task`` uses (background subagents whose results re-enter the
conversation as new messages) is a deliberate follow-up, not in scope here —
it would touch the core agent loop (run_agent.py), which the narrow-waist
rule keeps off-limits unless the capability is fundamental.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────
# Floor 60s so a typo can't make Devin unrunnable; ceiling is the user's call.
_DEFAULT_TIMEOUT_SECONDS = 1800.0
_MIN_TIMEOUT_SECONDS = 60.0
# Match delegate_task's sane upper bound for a returned summary so a verbose
# Devin run can't blow out the conversation context.
_DEFAULT_MAX_RESULT_CHARS = 20000
# permission_mode sent to Devin for unattended delegation. "accept-edits" is
# the safe default — Devin can modify files but still prompts for dangerous
# operations (shell commands, etc.). Users who want fully autonomous Devin
# must explicitly set permission_mode: dangerous in config.yaml.
_DEFAULT_PERMISSION_MODE = "accept-edits"
_VALID_PERMISSION_MODES = ("auto", "accept-edits", "smart", "dangerous")
# Ceiling for model-controllable timeout override. The model can request a
# timeout up to this value, but not beyond — prevents a prompt-injected model
# from stalling the turn with an arbitrarily long subprocess.run. The config
# value (timeout_seconds) is the real ceiling for unattended runs; this only
# caps the model-facing override.
_MAX_MODEL_TIMEOUT_SECONDS = 7200.0  # 2 hours


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the child and its entire process tree.

    On POSIX, ``start_new_session=True`` puts the child in a new process
    group, so ``killpg`` reaches all descendants (Devin spawns shell,
    browser, etc.). On Windows, fall back to psutil-based tree kill
    (mirrors ``tools/code_execution_tool.py:_kill_process_group``).
    """
    killpg = getattr(os, "killpg", None)
    if killpg is not None:
        # start_new_session=True makes the child the process-group leader,
        # so its pgid is reliably proc.pid. Using proc.pid directly (instead
        # of os.getpgid(proc.pid)) avoids a ProcessLookupError if the direct
        # child has already exited while descendants still hold the pipe open
        # — which is exactly the case when communicate(timeout=...) timed out.
        try:
            os.killpg(proc.pid, 9)  # SIGKILL
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
    else:
        # Windows: no killpg, use psutil to kill the tree
        try:
            import psutil
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        except ImportError:
            try:
                proc.kill()
            except Exception:
                pass


def _devin_config() -> Dict[str, Any]:
    """Return the ``delegation.devin`` config block (a dict, possibly empty).

    Reuses ``delegate_tool._load_config()`` so the same config priority
    (config.yaml via ``load_config_readonly()`` → legacy ``cli.CLI_CONFIG``)
    and the same ``HERMES_IGNORE_USER_CONFIG`` semantics apply. Never raises.
    """
    try:
        from tools.delegate_tool import _load_config

        delegation = _load_config() or {}
        devin = delegation.get("devin") or {}
        return devin if isinstance(devin, dict) else {}
    except Exception:
        return {}


def _devin_binary() -> Optional[str]:
    """Return the path to the ``devin`` executable on ``$PATH``, or None."""
    return shutil.which("devin")


# Local truthy set — mirrors utils.TRUTHY_STRINGS so ``check_fn`` doesn't have
# to import ``utils`` (which pulls a top-level ``import yaml``) on every schema
# rebuild. ``check_fn`` must stay cheap and dependency-light; the handler path
# still uses the full ``_load_config`` machinery for config priority.
_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    return bool(value)


def check_devin_requirements() -> bool:
    """``check_fn`` gate for the ``delegate_to_devin`` tool.

    True only when the user has opted in via ``delegation.devin.enabled`` AND
    the ``devin`` binary is reachable on ``$PATH``. Kept deliberately cheap
    (no subprocess, no heavy imports) because the registry calls ``check_fn``
    on every ``get_definitions()`` rebuild; authentication is verified at
    handler time so a missing login yields a clear error rather than the tool
    vanishing.
    """
    try:
        cfg = _devin_config()
        if not _truthy(cfg.get("enabled"), default=False):
            return False
        return _devin_binary() is not None
    except Exception:
        return False


def _is_logged_in() -> tuple[bool, str]:
    """Probe ``devin auth status``. Returns (logged_in, detail).

    Best-effort: any failure to run/parse is treated as "not logged in" with
    a detail string suitable for the tool error. Never raises.
    """
    binary = _devin_binary()
    if binary is None:
        return False, "the `devin` binary is not on $PATH"
    try:
        proc = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False, "`devin auth status` timed out"
    except Exception as exc:  # pragma: no cover — environment-dependent
        return False, f"`devin auth status` failed to run: {exc}"
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # `devin auth status` prints "Logged in ..." when authenticated; a non-zero
    # exit or a "not logged in"/"logged out" line means no credentials.
    lowered = out.lower()
    if proc.returncode == 0 and "logged in" in lowered and "not logged in" not in lowered:
        return True, ""
    snippet = (proc.stdout or proc.stderr or "").strip().splitlines()
    detail = snippet[0] if snippet else "not logged in"
    return False, f"Devin is not authenticated ({detail}). Run `devin auth login`."


def _resolve_permission_mode(cfg: Dict[str, Any]) -> str:
    mode = str(cfg.get("permission_mode") or _DEFAULT_PERMISSION_MODE).strip().lower()
    return mode if mode in _VALID_PERMISSION_MODES else _DEFAULT_PERMISSION_MODE


def _resolve_timeout(cfg: Dict[str, Any], override: Any) -> float:
    """Resolve the effective timeout, clamping the model override.

    Config ``timeout_seconds`` is the real ceiling (user-set, not model-
    controllable). When the model passes a ``timeout`` override, it is
    clamped to ``[60, min(config_timeout, _MAX_MODEL_TIMEOUT_SECONDS)]``
    so a prompt-injected model cannot stall the turn with an arbitrarily
    long subprocess, but can still shorten the cap for a quick task.
    """
    config_timeout = cfg.get("timeout_seconds")
    if config_timeout is not None:
        try:
            config_timeout = float(config_timeout)
        except (TypeError, ValueError):
            config_timeout = _DEFAULT_TIMEOUT_SECONDS
        if not math.isfinite(config_timeout):
            config_timeout = _DEFAULT_TIMEOUT_SECONDS
    else:
        config_timeout = _DEFAULT_TIMEOUT_SECONDS
    config_timeout = max(_MIN_TIMEOUT_SECONDS, config_timeout)

    if override is None:
        return config_timeout

    try:
        parsed = float(override)
    except (TypeError, ValueError):
        logger.warning(
            "delegate_to_devin timeout=%r is not a valid number; "
            "using config default %.0f",
            override, config_timeout,
        )
        return config_timeout
    if not math.isfinite(parsed):
        logger.warning(
            "delegate_to_devin timeout=%r is not finite; "
            "using config default %.0f",
            override, config_timeout,
        )
        return config_timeout
    # Clamp model override: floor 60s, ceiling = min(config, hard cap).
    # The model can shorten the timeout but cannot extend it beyond what
    # the user configured (or the hard safety cap).
    ceiling = min(config_timeout, _MAX_MODEL_TIMEOUT_SECONDS)
    return max(_MIN_TIMEOUT_SECONDS, min(parsed, ceiling))


def _resolve_max_result_chars(cfg: Dict[str, Any]) -> int:
    raw = cfg.get("max_result_chars", _DEFAULT_MAX_RESULT_CHARS)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_RESULT_CHARS
    return val if val >= 256 else _DEFAULT_MAX_RESULT_CHARS


def _build_prompt(goal: str, context: Optional[str]) -> str:
    goal = goal.strip()
    context = (context or "").strip()
    if not context:
        return goal
    return f"{goal}\n\n--- Context ---\n{context}"


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def delegate_to_devin(
    goal: Optional[str] = None,
    context: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Any = None,
    **_kw: Any,
) -> str:
    """Delegate a self-contained subtask to the local Devin CLI.

    Spawns ``devin -p --permission-mode <mode> --respect-workspace-trust false
    [--model <m>] -- <prompt>`` in the current working directory, waits for it
    to finish, and returns a JSON result shaped like ``delegate_task``'s
    (``{"results": [entry]}``) so the model can consume either backend
    uniformly.

    Synchronous: blocks the caller's turn until Devin returns. A
    ``timeout_seconds`` cap (default 1800, floor 60) kills the run and reports
    ``status="timeout"``.
    """
    goal = (goal or "").strip()
    if not goal:
        return tool_error("delegate_to_devin requires a non-empty 'goal'.")

    cfg = _devin_config()

    # Re-check the enabled gate at handler time. The registry TTL-caches
    # check_fn results, so if an operator disables delegation.devin.enabled
    # while a cached schema still exposes the tool, the handler must still
    # refuse the call rather than launching Devin with no authorization.
    if not _truthy(cfg.get("enabled"), default=False):
        return tool_error(
            "Devin delegation is disabled. Enable delegation.devin.enabled "
            "in config.yaml to use this tool."
        )

    binary = _devin_binary()
    if binary is None:
        return tool_error(
            "The `devin` CLI is not on $PATH. Install Devin for Terminal and "
            "retry, or disable delegation.devin in config.yaml."
        )

    # Verify auth at handler time so a stale login is a clear, actionable
    # error rather than the tool silently absent from the schema.
    logged_in, auth_detail = _is_logged_in()
    if not logged_in:
        return tool_error(auth_detail)

    permission_mode = _resolve_permission_mode(cfg)
    timeout_seconds = _resolve_timeout(cfg, timeout)
    max_chars = _resolve_max_result_chars(cfg)

    # Model: per-call override beats config. Devin accepts short names
    # (opus, swe, codex, ...) or full model ids.
    resolved_model = (model or cfg.get("model") or "").strip() or None

    prompt = _build_prompt(goal, context)

    argv = [binary, "-p", "--permission-mode", permission_mode,
            "--respect-workspace-trust", "false"]
    if resolved_model:
        argv += ["--model", resolved_model]
    argv += ["--", prompt]

    # Resolve cwd the same way the terminal tool does — TERMINAL_CWD is
    # force-exported from config.yaml's terminal.cwd by cli.py, so messaging
    # gateway sessions delegate to the user's configured workspace, not the
    # gateway launch directory. Validate and fall back to os.getcwd() if the
    # configured directory is relative, deleted, or inaccessible.
    workdir = os.environ.get("TERMINAL_CWD") or os.getcwd()
    try:
        workdir = str(Path(workdir).resolve())
        if not Path(workdir).is_dir():
            workdir = os.getcwd()
    except (OSError, ValueError, RuntimeError):
        workdir = os.getcwd()

    start = time.monotonic()
    try:
        # Use Popen with start_new_session so we can kill the entire process
        # group on timeout — Devin spawns child processes (shell, browser,
        # etc.) and subprocess.run only kills the direct child, leaving
        # descendants running. Mirrors tools/code_execution_tool.py's pattern.
        # stdin=DEVNULL prevents interactive prompts from blocking the
        # synchronous tool call until timeout expires.
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workdir,
            start_new_session=True,  # creates a new process group (POSIX)
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            # Kill the entire process group so Devin's children don't outlive it.
            _kill_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except Exception:
                stdout, stderr = "", ""
            duration = time.monotonic() - start
            # Tailor the error hint: only show the "override" message when the
            # model's timeout actually shortened the effective timeout (i.e.,
            # the resolved timeout is less than the config default). If the
            # override was invalid or at/above the ceiling, _resolve_timeout
            # returns the config default and the "override" hint is misleading.
            config_default = _resolve_timeout(cfg, None)
            if timeout is not None and timeout_seconds < config_default:
                timeout_hint = (
                    "The model supplied a shorter timeout override; ask it to "
                    "retry without the timeout parameter or raise "
                    "delegation.devin.timeout_seconds in config.yaml."
                )
            else:
                timeout_hint = (
                    "Raise delegation.devin.timeout_seconds if the task needs "
                    "more time."
                )
            return json.dumps(
                {"results": [{
                    "task_index": 0,
                    "status": "timeout",
                    "summary": "",
                    "error": (
                        f"Devin did not finish within {timeout_seconds:.0f}s. "
                        f"{timeout_hint}"
                    ),
                    "duration_seconds": round(duration, 3),
                    "model": resolved_model,
                    "exit_reason": "timeout",
                    "truncated": False,
                    "backend": "devin",
                }]},
                ensure_ascii=False,
            )
    except Exception as exc:  # pragma: no cover — environment-dependent
        return tool_error(f"Failed to spawn Devin: {exc}")

    duration = time.monotonic() - start
    stdout = stdout or ""
    stderr = stderr or ""

    if proc.returncode != 0:
        err, err_truncated = _truncate(stderr.strip() or stdout.strip(), max_chars)
        return json.dumps(
            {"results": [{
                "task_index": 0,
                "status": "error",
                "summary": "",
                "error": f"Devin exited with code {proc.returncode}: {err}",
                "duration_seconds": round(duration, 3),
                "model": resolved_model,
                "exit_reason": "error",
                "truncated": err_truncated,
                "backend": "devin",
            }]},
            ensure_ascii=False,
        )

    summary, truncated = _truncate(stdout.strip(), max_chars)
    return json.dumps(
        {"results": [{
            "task_index": 0,
            "status": "completed",
            "summary": summary,
            "duration_seconds": round(duration, 3),
            "model": resolved_model,
            "exit_reason": "completed",
            "truncated": truncated,
            "backend": "devin",
        }]},
        ensure_ascii=False,
    )


DELEGATE_TO_DEVIN_SCHEMA = {
    "name": "delegate_to_devin",
    "description": (
        "Delegate a self-contained subtask to the local Devin CLI (an external "
        "agent) and return its finished response. Devin runs its own complete "
        "agent loop — terminal, file edits, browser — and hands back a done "
        "answer, so use this for well-scoped, end-to-end subtasks you do not "
        "need to observe step-by-step. Unlike `delegate_task` (which spawns a "
        "Hermes subagent you can steer and stream), this is a single blocking "
        "call: you receive Devin's full result in this turn. Only available "
        "when `delegation.devin.enabled` is set in config.yaml and the Devin "
        "CLI is installed and authenticated. Provide a fully self-contained "
        "goal — Devin knows nothing about your current conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What Devin should accomplish. Be specific and "
                    "self-contained — Devin has none of your conversation "
                    "context. Include file paths, constraints, and the "
                    "definition of done."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional background: project structure, error messages, "
                    "relevant code snippets. Appended to the goal under a "
                    "Context header."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional Devin model short name or id (e.g. 'opus', "
                    "'swe', 'codex'). Overrides delegation.devin.model in "
                    "config.yaml for this call."
                ),
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Optional wall-clock cap in seconds (min 60, default "
                    "1800). If Devin does not finish in time it is killed and "
                    "a timeout result is returned."
                ),
            },
        },
        "required": ["goal"],
    },
}


registry.register(
    name="delegate_to_devin",
    toolset="delegation",
    schema=DELEGATE_TO_DEVIN_SCHEMA,
    handler=lambda args, **kw: delegate_to_devin(
        goal=args.get("goal"),
        context=args.get("context"),
        model=args.get("model"),
        timeout=args.get("timeout"),
        **kw,
    ),
    check_fn=check_devin_requirements,
    emoji="🤖",
)
