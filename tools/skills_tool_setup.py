"""Skill readiness: required env vars, secret capture, and setup notes. Split out of
``tools.skills_tool`` (names re-imported there); module state (``_secret_capture_callback``,
``load_env``) stays there and is read lazily at call time so origin-module patches are honored."""

import contextlib
import logging
import os
import re
import shlex
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List

from hermes_constants import display_hermes_home
from utils import env_var_enabled

logger = logging.getLogger("tools.skills_tool")

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMMAND_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]*$")
_SNAPSHOT_PATH_RE = re.compile(r'^(?:declare -x|export) PATH=(?:"([^"]*)"|\'([^\']*)\'|(\S+))$', re.MULTILINE)
_LOGIN_SHELL_PROBE_MARKER = "__HERMES_SKILL_CMD_PROBE__"
_LOGIN_SHELL_PROBE_TIMEOUT_SECONDS = 5
_REMOTE_ENV_BACKENDS = frozenset({"docker", "modal", "ssh", "daytona", "vercel_sandbox"})


class SkillReadinessStatus(str, Enum):
    AVAILABLE = "available"
    SETUP_NEEDED = "setup_needed"
    UNSUPPORTED = "unsupported"


def _is_remote_env_backend(backend: str) -> bool:
    """Built-in remote backends plus plugin backends declaring is_remote."""
    if backend in _REMOTE_ENV_BACKENDS or not backend or backend == "local":
        return backend in _REMOTE_ENV_BACKENDS
    try:
        from agent.terminal_env_registry import provider_flag
        return bool(provider_flag(backend, "is_remote", False))
    except Exception:
        return False


def _as_dict_list(raw: Any) -> list:
    """Accept a single mapping or a list; anything else is treated as empty."""
    return [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []


def _clean_str(value: Any) -> str | None:
    """Stripped string when *value* is a non-blank str, else None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _get_required_environment_variables(frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge required_environment_variables, setup.collect_secrets and legacy
    prerequisites.env_vars into one deduped, validated list (first entry wins)."""
    setup = frontmatter.get("setup")
    setup = setup if isinstance(setup, dict) else {}
    setup_help = _clean_str(setup.get("help"))
    prereqs = frontmatter.get("prerequisites")
    legacy = (prereqs.get("env_vars") if isinstance(prereqs, dict) else None) or []
    legacy = [legacy] if isinstance(legacy, str) else legacy
    required: dict[str, dict[str, Any]] = {}  # env name -> entry, insertion-ordered, first wins
    declared = _as_dict_list(frontmatter.get("required_environment_variables"))
    entries = [{"name": i} if isinstance(i, str) else i for i in declared
               if isinstance(i, (str, dict))]
    # collect_secrets entries: env_var is the name; provider_url (or url) doubles as help.
    entries += [
        {"name": i.get("env_var"), "prompt": i.get("prompt"),
         "url": str(i.get("provider_url") or i.get("url") or "").strip() or None}
        for i in _as_dict_list(setup.get("collect_secrets")) if isinstance(i, dict)]
    entries += [{"name": str(v)} for v in legacy if str(v).strip()]
    for entry in entries:
        env_name = str(entry.get("name") or entry.get("env_var") or "").strip()
        if not env_name or env_name in required or not _ENV_VAR_NAME_RE.match(env_name):
            continue
        normalized: dict[str, Any] = {
            "name": env_name,
            "prompt": str(entry.get("prompt") or f"Enter value for {env_name}").strip()}
        if help_text := _clean_str(
                entry.get("help") or entry.get("provider_url") or entry.get("url") or setup_help):
            normalized["help"] = help_text
        if required_for := _clean_str(entry.get("required_for")):
            normalized["required_for"] = required_for
        if entry.get("optional"):
            normalized["optional"] = True
        required[env_name] = normalized
    return list(required.values())


def _get_required_commands(frontmatter: dict[str, Any]) -> list[str]:
    """Parse ``prerequisites.commands`` into a deduped, validated list of command names."""
    prereqs = frontmatter.get("prerequisites")
    raw = prereqs.get("commands") if isinstance(prereqs, dict) else None
    items = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
    required: dict[str, None] = {}
    for item in items:
        if not isinstance(item, str):
            continue
        cmd = item.strip()
        if cmd and _COMMAND_NAME_RE.match(cmd):
            required.setdefault(cmd, None)
    return list(required)


def _normalize_snapshot_path(
    snapshot_path: str, *, is_windows: bool | None = None, pathsep: str | None = None,
    to_native: Callable[[str], str] | None = None) -> str:
    """Convert a Git Bash ``export -p`` snapshot ``PATH`` to native ``os.pathsep`` form on Windows."""
    if not snapshot_path:
        return ""
    if is_windows is None or to_native is None:
        from tools.environments.local import _IS_WINDOWS, _msys_to_windows_path
        if is_windows is None:
            is_windows = _IS_WINDOWS
        if to_native is None:
            to_native = _msys_to_windows_path
    if not is_windows:
        return snapshot_path
    sep = os.pathsep if pathsep is None else pathsep
    return sep.join(to_native(entry) for entry in snapshot_path.split(":") if entry)


def _live_local_snapshot_path(task_id: str | None = None) -> tuple[dict[str, str], str | None]:
    """Return ``(backend_env, snapshot_path)`` from the active local terminal session, if any."""
    try:
        from tools.terminal_tool_lifecycle import get_active_env
        env = get_active_env(task_id or "default")
    except Exception:
        logger.debug("Could not inspect active terminal env for skill readiness", exc_info=True)
        return {}, None
    if env is None or not getattr(env, "is_local", False):
        return {}, None
    backend_env = getattr(env, "env", None)
    backend_env = dict(backend_env) if isinstance(backend_env, dict) else {}
    snap_file = getattr(env, "_snapshot_path", None) if getattr(env, "_snapshot_ready", False) else None
    if not isinstance(snap_file, str) or not snap_file:
        return backend_env, None
    try:
        content = Path(snap_file).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return backend_env, None
    matches = list(_SNAPSHOT_PATH_RE.finditer(content))
    if not matches:
        return backend_env, None
    last = matches[-1]
    raw = last.group(1) if last.group(1) is not None else (last.group(2) if last.group(2) is not None else last.group(3))
    if raw and last.group(1) is not None:
        raw = raw.replace(r"\\", "\\").replace(r'\"', '"').replace(r"\$", "$").replace(r"\`", "`")
    normalized = _normalize_snapshot_path(raw) if raw else ""
    return backend_env, (normalized or None)


def _probe_missing_via_login_shell(commands: list[str], run_env: dict[str, str], path_key: str) -> list[str]:
    """Re-resolve *commands* in one ``bash -lc`` login shell matching ``LocalEnvironment._run_bash(login=True)``.

    Any lookup failure or timeout treats the commands as unknown (returns ``[]``) and logs at debug.
    """
    proc = None
    owned_pgid = None
    try:
        from hermes_cli._subprocess_compat import windows_hide_flags
        from hermes_cli.process_identity import own_process_groups, release_process_groups
        from tools.environments.local import (
            _IS_WINDOWS, _append_dropped_path_entries, _find_bash,
            _kill_process_group_posix, _kill_process_windows,
            _prepend_shell_init, _resolve_shell_init_files)
        bash = _find_bash()
        quoted = " ".join(shlex.quote(c) for c in commands)
        probe = (
            f"for __hermes_c in {quoted}; do "
            f'if command -v -- "$__hermes_c" >/dev/null 2>&1; then '
            f'printf "{_LOGIN_SHELL_PROBE_MARKER}:1:%s\\n" "$__hermes_c"; '
            f'else printf "{_LOGIN_SHELL_PROBE_MARKER}:0:%s\\n" "$__hermes_c"; fi; '
            "done"
        )
        script = _append_dropped_path_entries(probe, run_env.get(path_key, ""))
        script = _prepend_shell_init(script, _resolve_shell_init_files())
        proc = subprocess.Popen(
            [bash, "-l", "-c", script], env=run_env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            # detached-spawn: ok — owned via own_process_groups right below and group-killed on timeout
            start_new_session=True,
            **({"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}))
        if not _IS_WINDOWS:
            own_process_groups([proc.pid])
            owned_pgid = proc.pid
            with contextlib.suppress(ProcessLookupError):
                proc._hermes_pgid = os.getpgid(proc.pid)
                from gateway.status import get_process_start_time
                proc._hermes_pgid_start = get_process_start_time(proc.pid)
        try:
            stdout, _ = proc.communicate(timeout=_LOGIN_SHELL_PROBE_TIMEOUT_SECONDS)
        except BaseException:
            with contextlib.suppress(Exception):
                (_kill_process_windows if _IS_WINDOWS else _kill_process_group_posix)(proc)
            with contextlib.suppress(Exception):
                proc.communicate(timeout=0.5)
            raise
        finally:
            if owned_pgid is not None:
                release_process_groups([owned_pgid])
    except Exception:
        logger.debug("Login-shell command readiness lookup failed for %s; treating as unknown", commands, exc_info=True)
        return []
    verdicts: dict[str, bool] = {}
    prefix = f"{_LOGIN_SHELL_PROBE_MARKER}:"
    for line in (stdout or "").splitlines():
        if line.startswith(prefix):
            _, status, name = line.split(":", 2)
            if status in ("0", "1") and name:
                verdicts[name] = status == "1"
    if not all(c in verdicts for c in commands):
        logger.debug(
            "Login-shell command readiness lookup returned incomplete output (rc=%s) for %s; treating as unknown",
            proc.returncode, commands)
        return []
    return [c for c in commands if not verdicts[c]]


def _resolve_required_commands(
    frontmatter: dict[str, Any], backend: str, task_id: str | None = None) -> tuple[list[str], list[str]]:
    """Return ``(required_commands, missing_required_commands)`` for *frontmatter*.

    On the ``local`` terminal backend, each command is first checked via
    ``hermes_platform.resolver.locate_command`` against the spawn ``PATH``
    (``_make_run_env``, plus any live local session snapshot ``PATH``). Commands
    still unresolved are checked in a single ``bash -lc`` login-shell call that
    sources user init files and ``terminal.shell_init_files`` just like
    ``LocalEnvironment._run_bash(..., login=True)``; if that probe fails or times
    out, unresolved commands are treated as unknown (not missing).
    On non-local backends (``docker``, ``ssh``, ``modal``, ``daytona``,
    ``vercel_sandbox``, ...), the host ``PATH`` says nothing about what is
    installed inside the sandbox, so ``required_commands`` is still reported for
    visibility while ``missing_required_commands`` stays empty.
    """
    required_commands = _get_required_commands(frontmatter)
    if not required_commands or backend != "local":
        return required_commands, []
    from hermes_platform.resolver import LookupContext, locate_command
    from tools.environments.local import _make_run_env, _path_env_key
    backend_env, snapshot_path = _live_local_snapshot_path(task_id)
    # _make_run_env receives the live LocalEnvironment's self.env when a session exists (else {}).
    run_env = _make_run_env(backend_env)
    path_key = _path_env_key(run_env) or "PATH"
    spawn_path = run_env.get(path_key) or os.environ.get("PATH") or ""
    fast_path = f"{snapshot_path}{os.pathsep}{spawn_path}" if snapshot_path and spawn_path else (snapshot_path or spawn_path)
    missing = [cmd for cmd in required_commands if not locate_command(cmd, LookupContext(path=fast_path)).found]
    if not missing or snapshot_path:
        return required_commands, missing
    return required_commands, _probe_missing_via_login_shell(missing, run_env, path_key)


def _capture_result(missing_names, setup_skipped=False, gateway_setup_hint=None):
    return {"missing_names": missing_names, "setup_skipped": setup_skipped, "gateway_setup_hint": gateway_setup_hint}


def _capture_required_environment_variables(
    skill_name: str, missing_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Prompt for missing secrets via the registered capture callback (if any)."""
    from tools import skills_tool as _st
    if not missing_entries:
        return _capture_result([])
    missing_names = [entry["name"] for entry in missing_entries]
    callback = _st._turn_secret_capture.get()
    # Messaging-platform gateway surfaces can't prompt for a secret, so they get the "unsupported"
    # hint. Interactive gateway surfaces (desktop app / TUI) set HERMES_INTERACTIVE (same flag
    # tools/approval.py uses) and register a callback routing to a secure secret.request overlay.
    if callback is None and _is_gateway_surface() and not env_var_enabled("HERMES_INTERACTIVE"):
        try:
            from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE as hint
        except Exception:
            hint = (f"Secure secret entry is not available. Load this skill in the local CLI to be "
                    f"prompted, or add the key to {display_hermes_home()}/.env manually.")
        return _capture_result(missing_names, gateway_setup_hint=hint)
    if callback is None and (callback := _st._secret_capture_callback) is None:
        return _capture_result(missing_names)
    remaining_names: list[str] = []
    for entry in missing_entries:
        metadata = {"skill_name": skill_name, **{k: entry[k] for k in ("help", "required_for") if entry.get(k)}}
        try:
            callback_result = callback(entry["name"], entry["prompt"], metadata)
        except Exception:
            logger.warning(f"Secret capture callback failed for {entry['name']}", exc_info=True)
            callback_result = {"success": False, "stored_as": entry["name"], "validated": False, "skipped": True}
        ok = isinstance(callback_result, dict) and callback_result.get("success")
        if not (ok and not callback_result.get("skipped")):
            remaining_names.append(entry["name"])
    return _capture_result(remaining_names, bool(remaining_names))


def _is_gateway_surface() -> bool:
    if env_var_enabled("HERMES_GATEWAY_SESSION"):
        return True
    from gateway.session_context import get_session_env
    return bool(get_session_env("HERMES_SESSION_PLATFORM"))


def _is_env_var_persisted(var_name: str, env_snapshot: dict[str, str]) -> bool:
    """Return whether a requirement is present in this profile's secret sources.

    The snapshot keeps the existing ``.env`` precedence.  A miss must use
    ``get_secret`` rather than reading ``os.environ`` directly so an active
    multiplex worker can see its hydrated external secrets without seeing a
    different profile's process environment.
    """
    if var_name in env_snapshot:
        return bool(env_snapshot[var_name])
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
        return bool(get_secret(var_name))
    except UnscopedSecretError:
        return False


def _build_setup_note(
    readiness_status: SkillReadinessStatus, missing: list[str],
    setup_help: str | None = None) -> str | None:
    if readiness_status != SkillReadinessStatus.SETUP_NEEDED:
        return None
    note = f"Setup needed before using this skill: missing {', '.join(missing) if missing else 'required prerequisites'}."
    return f"{note} {setup_help}" if setup_help else note
