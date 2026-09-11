#!/usr/bin/env python3
"""Terminal tool: run shell commands in the configured backend.

Backends (``TERMINAL_ENV``): local (default), docker, singularity, modal
(direct or managed gateway), daytona, vercel_sandbox, ssh, plus
plugin-registered backends. Handles background processes, sandbox lifecycle
(per-task cache, idle reaper, atexit teardown) and sudo password plumbing.
Cloud-sandbox persistent filesystems preserve working state across sandbox
recreation but do NOT guarantee the same live sandbox or long-running
processes survive cleanup, idle reaping, or Hermes exit.

Companion modules (re-exported here, so ``tools.terminal_tool.<name>`` stays the
import/patch target): ``terminal_tool_config`` (TERMINAL_* reads, ``_quiet``),
``terminal_tool_backends`` (env builders + requirement checkers),
``terminal_tool_lifecycle`` (reaper/teardown/ensure_task_env),
``terminal_tool_sudo`` (sudo password + shell rewrites), ``terminal_tool_guards``
(pre-exec blocks), ``terminal_tool_background`` (background spawn),
``terminal_tool_result`` (foreground result post-processing).
"""

import json
import logging
import os
import re
import shutil
import sys
import time
import threading
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

# ANSI stripper internals are reused by _IncrementalAnsiStripper so the
# streaming spill pipeline applies the SAME escape grammar as the visible
# output path (tools.ansi_strip.strip_ansi).
from tools.ansi_strip import _ANSI_ESCAPE_RE, _HAS_ESCAPE as _HAS_ANSI_BYTE

logger = logging.getLogger(__name__)


def _redact_terminal_error_text(value: Any) -> str:
    """Force-redact text before serializing a terminal error envelope."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text("" if value is None else str(value), force=True)


from tools.registry import tool_error
from tools.terminal_tool_lifecycle import (
    _check_disk_usage_warning, _cleanup_inactive_envs, _create_configured_env,
    _evict_environment_for_task, cleanup_all_environments, ensure_task_env,
)
from tools.terminal_tool_config import (
    _is_container_backend, _is_host_cwd, _is_unusable_container_cwd, _parse_env_var,
    _plugin_env_flag, _quiet, _safe_getcwd, _tenv, _tenv_bool,
)
from tools.terminal_tool_backends import (
    _REQUIREMENT_CHECKERS, _VERCEL_SANDBOX_DEFAULT_CWD, _check_plugin_requirements,
)
# display_hermes_home imported lazily at call site (stale-module safety during hermes update)
from tools.tool_backend_helpers import coerce_modal_mode, managed_nous_tools_enabled


def _safe_parse_import_env(name: str, default: Any, converter, type_label: str):
    """Parse a module-level numeric env var; a malformed value must never make
    the module unloadable at import time (CLI, ACP, tests, tool discovery)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s: %r (expected %s). Falling back to %r.",
            name, raw, type_label, default,
        )
        return default


# Hard cap on foreground timeout; override via TERMINAL_MAX_FOREGROUND_TIMEOUT env var.
FOREGROUND_MAX_TIMEOUT = _safe_parse_import_env("TERMINAL_MAX_FOREGROUND_TIMEOUT", 600, int, "integer")

# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = _safe_parse_import_env("TERMINAL_DISK_WARNING_GB", 500.0, float, "number")


# Approval / sudo-prompt UI callbacks (CLI registers prompt_toolkit-aware
# ones). Thread-local so overlapping ACP sessions, each on its own executor
# thread, can't stomp on each other (GHSA-qg5c-hvr5-hjgr). Gateway mode
# resolves approvals via the per-session queue in tools.approval instead.
_callback_tls = threading.local()


def _get_sudo_password_callback():
    return getattr(_callback_tls, "sudo_password", None)


def _get_approval_callback():
    return getattr(_callback_tls, "approval", None)


def set_sudo_password_callback(cb):
    """Register the CLI's sudo password prompt callback (per-thread slot)."""
    _callback_tls.sudo_password = cb


def set_approval_callback(cb):
    """Register the dangerous-command approval prompt callback (per-thread slot)."""
    _callback_tls.approval = cb


def _current_session_key() -> str:
    """Active gateway/WebUI session key, or "" outside sessions (ContextVar with
    the ``get_session_env`` os.environ fallback for CLI/cron/tests)."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_KEY", "")


def _current_session_profile() -> str:
    """Active session's Hermes profile name, or "" (same lookup discipline as
    :func:`_current_session_key`)."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_PROFILE", "")


from tools.approval import (
    check_all_command_guards as _check_all_guards_impl,
)


def _docker_volume_uses_host_path(volume_spec: str) -> bool:
    """Return True when a docker volume spec bind-mounts a host path."""
    if not isinstance(volume_spec, str):
        return False
    vol = volume_spec.strip()
    return bool(vol) and (
        vol.startswith(("/", "~", "./", "../")) or
        (len(vol) >= 3 and vol[1] == ":" and vol[2] in ("/", "\\"))
    )


def _docker_has_host_access(config: Dict[str, Any]) -> bool:
    """Return True when a Docker sandbox exposes host paths through bind mounts."""
    if config.get("env_type") != "docker":
        return False
    if config.get("host_cwd") and config.get("docker_mount_cwd_to_workspace"):
        return True
    return any(_docker_volume_uses_host_path(vol) for vol in config.get("docker_volumes", []))


def _check_all_guards(command: str, env_type: str,
                      has_host_access: bool = False) -> dict:
    """Delegate to consolidated guard (tirith + dangerous cmd) with CLI callback."""
    return _check_all_guards_impl(command, env_type,
                                  approval_callback=_get_approval_callback(),
                                  has_host_access=has_host_access)


from tools.environments.base import EnvironmentConnectionError


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands. The host OS, shell, and terminal backend are stated in your environment section — write commands for THAT platform. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail (use read_file), grep/rg/find/ls (use search_files), sed/awk (use patch), or echo/heredoc file creation (use write_file). Reserve terminal for: builds, installs, git, processes, scripts, network, package managers — anything that needs a shell. Output is auto-truncated with the full text saved to a file — never pipe through tail/head to shorten it.
Environment state persists: activate a virtualenv or export variables once per session, not before every command.

Foreground (default): returns INSTANTLY when the command finishes, even with a high timeout — set timeout generously for long builds.
Background: set background=true (returns a session_id); add notify=true for bounded tasks, leave silent only for servers/daemons that never exit. After starting a server, verify readiness with a health check in a separate call (no blind sleep loops); manage with process(action="poll"/"wait").
Working directory: use 'workdir' for per-command cwd; when a command changes the session cwd (cd, pushd), trust the result's "cwd" field instead of prefixing every command with 'cd'.
PTY: pty=true + background=true for interactive CLIs (they hang without a terminal); drive them with process(action="write"/"submit"). Local backend only.
"""

# Environment lifecycle state.
_active_environments: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_creation_locks: Dict[str, threading.Lock] = {}  # Per-task locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Once-per-process guard for the docker orphan reaper.
_docker_orphan_reaper_ran = False
_docker_orphan_reaper_lock = threading.Lock()


def _maybe_reap_docker_orphans(container_config: Dict[str, Any]) -> None:
    """Run the docker orphan reaper once per process, if enabled.

    Sweeps Exited containers labeled ``hermes-agent=1`` for the current
    profile — leftovers of Hermes processes that died without firing
    ``atexit`` (SIGKILL, OOM, closed terminal). Conservative: only containers
    older than ``2 × lifetime_seconds``, profile-scoped. Gates:
    ``terminal.docker_orphan_reaper: false`` (operator opt-out, e.g. several
    Hermes processes sharing a profile) and the once-per-interpreter flag so
    parallel subagent / RL-rollout calls don't re-sweep.
    """
    global _docker_orphan_reaper_ran
    if not container_config.get("docker_orphan_reaper", True):
        return
    if _docker_orphan_reaper_ran:  # double-checked locking
        return
    with _docker_orphan_reaper_lock:
        if _docker_orphan_reaper_ran:
            return
        _docker_orphan_reaper_ran = True

    # 2 × lifetime gives sibling processes a grace window; floor at 60s so
    # TERMINAL_LIFETIME_SECONDS=0 can't instant-reap a sibling's own setup.
    # container_config only carries container_* keys, so read the env var.
    try:
        lifetime = int(_tenv("TERMINAL_LIFETIME_SECONDS", "300"))
    except (TypeError, ValueError):
        lifetime = 300
    max_age = max(60, lifetime) * 2

    try:
        from tools.environments.docker import reap_orphan_containers, _container_identity
    except ImportError:
        return
    # Never fail the env-creation path because of a janitor problem.
    with _quiet("Docker orphan reaper raised"):
        profile = _container_identity(container_config.get("docker_shared_container_key", ""))
        removed = reap_orphan_containers(max_age_seconds=max_age, profile_filter=profile)
        if removed:
            logger.info(
                "Docker orphan reaper removed %d stale container(s) for profile %s",
                removed, profile,
            )


# Per-task environment overrides (never exposed to the model). RL/benchmark
# envs and ACP register a custom image / cwd for a task_id BEFORE the agent
# loop; sandbox creation consults this first, then the TERMINAL_* env vars.
_task_env_overrides: Dict[str, Dict[str, Any]] = {}

# Per-session cwd records: the durable source of truth for "which directory
# is THIS session in". Keyed by the raw session/task key, NOT the collapsed
# container id — the env is shared across sessions, so cwd state stored on
# it is a global mutable timeshared between sessions (the wrong-worktree bug
# class). Written after every completed command and on cwd-override
# registration; readers resolve against it before any env-side cwd.
_session_cwd: Dict[str, str] = {}
_session_cwd_lock = threading.Lock()

# Subagent → parent container aliasing. delegate_task children have their own
# task_id but must share the PARENT's container; under per-session isolation
# the collapse-to-"default" shortcut no longer provides that, so the spawn
# site registers an explicit alias.
_container_aliases: Dict[str, str] = {}
_container_alias_lock = threading.Lock()


def record_session_cwd(session_key: Optional[str], cwd: Optional[str]) -> None:
    """Record *cwd* as *session_key*'s working directory (after a completed
    command, or on workspace-override registration). None/empty keys collapse
    to ``"default"``; non-string / empty cwds are ignored."""
    if not isinstance(cwd, str) or not cwd.strip():
        return
    key = str(session_key or "default")
    with _session_cwd_lock:
        if _session_cwd.get(key) != cwd:
            _session_cwd[key] = cwd


def get_session_cwd(session_key: Optional[str]) -> Optional[str]:
    """Recorded cwd for *session_key*, or None. No fallback chain on purpose:
    callers decide what an absent record means. None/empty keys read ``"default"``."""
    with _session_cwd_lock:
        return _session_cwd.get(str(session_key or "default"))


def clear_session_cwd(session_key: str) -> None:
    """Drop a session's cwd record (session teardown)."""
    with _session_cwd_lock:
        _session_cwd.pop(session_key, None)


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """Register per-task sandbox overrides (``docker_image``/``modal_image``/
    ``singularity_image``/``daytona_image``, ``env_type``, ``cwd``) before the
    agent loop runs.

    A ``cwd`` override takes effect immediately: it becomes the session's
    recorded cwd (until a ``cd`` changes it) and any live env's cwd is updated
    too, so env-side seeding stays consistent (ACP switching project root
    mid-session via ``session/load``).
    """
    _task_env_overrides[task_id] = overrides

    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        record_session_cwd(task_id, new_cwd)
        # Live env may be cached under the raw task_id (per-session surfaces)
        # or the collapsed container id (isolation-keyed rollouts); try both so
        # a CWD-only override (which collapses to "default") still finds it.
        container_id = _resolve_container_task_id(task_id)
        with _env_lock:
            env = _active_environments.get(task_id) or _active_environments.get(container_id)
        if env is not None and getattr(env, "cwd", None) is not None:
            env.cwd = new_cwd


def clear_task_env_overrides(task_id: str):
    """Drop a task's overrides, cwd record and container alias (rollout cleanup)."""
    _task_env_overrides.pop(task_id, None)
    clear_session_cwd(task_id)
    with _container_alias_lock:
        _container_aliases.pop(task_id, None)


def register_container_alias(child_task_id: str, parent_task_id: Optional[str]) -> None:
    """Make *child_task_id* resolve to *parent_task_id*'s container (called at
    delegate_task spawn). A missing parent id aliases to ``"default"``."""
    if not child_task_id:
        return
    with _container_alias_lock:
        _container_aliases[child_task_id] = str(parent_task_id or "default")


def _resolve_container_alias(task_id: str) -> str:
    """Follow the child→parent alias chain (cycle-safe) for *task_id*."""
    seen = set()
    key = task_id
    with _container_alias_lock:
        while key in _container_aliases and key not in seen:
            seen.add(key)
            key = _container_aliases[key]
    return key


_ISOLATION_OVERRIDE_KEYS = frozenset({
    "docker_image", "modal_image", "singularity_image",
    "daytona_image", "env_type",
})


def _has_isolation_overrides(task_id: Optional[str]) -> bool:
    """True when *task_id* registered image/env_type overrides — the single
    "isolated RL/benchmark rollout" predicate shared by key resolution and
    container creation so the two can't drift."""
    if not task_id or task_id not in _task_env_overrides:
        return False
    return bool(set(_task_env_overrides[task_id].keys()) & _ISOLATION_OVERRIDE_KEYS)


@dataclass(frozen=True)
class _SessionScope:
    """Backend identity + scoping predicates for one call, read once.

    ``env_type`` is the scope-aware TERMINAL_ENV; ``persistent`` is
    ``TERMINAL_CONTAINER_PERSISTENT``. Derived predicates:

    * ``session_isolated`` — non-persistent sandboxes get per-session identities:
      ``container_persistent: false`` means state must not survive or be shared
      across sessions, so one shared sandbox contradicts it. Docker, plus plugin
      backends declaring ``session_isolated_when_nonpersistent`` (sandboxes resumed
      by name, where a shared deterministic name would let two ephemeral runs
      attach one VM and delete it under each other).
    * ``docker_session_isolated`` — docker-only view: the workspace mount and
      session-scoped teardown paths must not fire for other backends.
    * ``docker_profile_scoped`` — docker + ``container_persistent: true``: ONE
      long-lived container per profile shared by every session (CLI, gateway,
      WebUI). The session-key fallback in :func:`_resolve_container_task_id` stops
      cross-profile SSH reuse; ungated it fragmented persistent Docker into one
      container per gateway session, so this restores profile scoping for exactly
      this backend/mode.
    """
    env_type: str
    persistent: bool

    @property
    def session_isolated(self) -> bool:
        if self.env_type != "docker" and not _plugin_env_flag(
            self.env_type, "session_isolated_when_nonpersistent"
        ):
            return False
        return not self.persistent

    @property
    def docker_session_isolated(self) -> bool:
        return self.env_type == "docker" and self.session_isolated

    @property
    def docker_profile_scoped(self) -> bool:
        return self.env_type == "docker" and self.persistent


def _session_scope() -> _SessionScope:
    """Bridge config → env once, then snapshot the backend scope for this call."""
    _ensure_terminal_env_bridged()
    return _SessionScope(
        env_type=_tenv("TERMINAL_ENV", "local"),
        persistent=_tenv_bool("TERMINAL_CONTAINER_PERSISTENT", "true"),
    )


def _docker_session_isolation_enabled() -> bool:
    """See :attr:`_SessionScope.docker_session_isolated` (used by the docker builder)."""
    return _session_scope().docker_session_isolated


def _resolve_container_task_id(task_id: Optional[str]) -> str:
    """Map a tool-call ``task_id`` to the ``_active_environments`` key. Order matters —
    earlier branches are authoritative where they apply:

    1. Image/``env_type`` overrides (RL/benchmark rollouts) key their own sandbox;
       CWD-only overrides (ACP workspace tracking) are NOT isolation signals.
    2. Per-session isolation (docker + ``container_persistent: false``): each
       session's task_id is its own key (a fresh chat gets a fresh sandbox with only
       ITS mounts); delegate_task children follow the alias registry to the parent.
    3. Session key present (WebUI per-session, gateway per-message): persistent
       Docker is PROFILE-scoped — ``shared:<key>`` opt-in, else ``profile:<name>``,
       with the default profile staying literally ``"default"`` so CLI and
       default-profile gateway sessions share ONE container; other backends key
       ``session:<key>`` so switching profiles can't reuse another profile's
       SSHEnvironment on the wrong host.
    4. No session key (CLI): ``shared:<key>`` when opted in (else a CLI run of a
       keyed profile would split from its gateway sessions), else ``"default"``,
       which subagent ids collapse onto to share the parent's container.
    """
    if task_id and _has_isolation_overrides(task_id):
        return task_id
    scope = _session_scope()
    if task_id and scope.session_isolated:
        return _resolve_container_alias(task_id)
    # Per-session isolation: when a session key is present (the WebUI streaming layer sets it per-session,
    # the gateway per-message via contextvars), scope the container to it so switching profiles can't reuse
    # a previous profile's SSHEnvironment and silently run commands on the wrong remote host. Subagents
    # inherit the same session key, so they still collapse onto the parent's container (the #16177
    # shared-container intent). CLI mode has no session key and falls through to "default", behaviour
    # unchanged. See commit e00f940a9. This runs *after* the isolation-override and
    # docker/container_persistent branches above: those paths already key containers per task_id, so they
    # stay authoritative where they apply and this only covers the cases that would otherwise collapse to
    # the shared "default" key (notably SSH).
    session_key = _current_session_key()
    shared = _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip() if scope.docker_profile_scoped else ""
    if shared:
        # Explicit opt-in: trusted profiles configuring the same terminal.docker_shared_container_key share
        # ONE container/cache slot (and sandbox dir) regardless of profile name (#84671).
        return f"shared:{shared}"
    if not session_key:
        return "default"
    if not scope.docker_profile_scoped:
        return f"session:{session_key}"
    profile = _current_session_profile() or "default"
    return "default" if profile == "default" else f"profile:{profile}"


def resolve_task_overrides(task_id: Optional[str]) -> Dict[str, Any]:
    """Return the env overrides for *task_id*, raw key first then collapsed.

    ``register_task_env_overrides`` writes under the *raw* task/session id, but
    a CWD-only override collapses (:func:`_resolve_container_task_id`) to the
    shared ``"default"`` container. Callers must therefore read the raw id
    FIRST and only fall back to the collapsed container id, or the originating
    session's override is silently dropped. Single source of that lookup so
    the terminal and file layers can't drift apart.
    """
    raw = task_id or "default"
    return (
        _task_env_overrides.get(raw)
        or _task_env_overrides.get(_resolve_container_task_id(raw))
        or {}
    )


# Backends that take an image, keyed to the override/config key carrying it.
_IMAGE_KEY_BY_BACKEND = {
    "docker": "docker_image",
    "singularity": "singularity_image",
    "modal": "modal_image",
    "daytona": "daytona_image",
}


def _select_image(env_type: str, overrides: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Image for *env_type*: per-task override first, then config; "" for imageless backends."""
    key = _IMAGE_KEY_BY_BACKEND.get(env_type)
    if key is None:
        return ""
    return overrides.get(key) or config[key]


def _lookup_active_env(effective_task_id: str, task_id: Optional[str]):
    """Return the cached env for the collapsed id, else for the raw task_id, else None.

    Caller holds ``_env_lock``. Per-session surfaces (ACP/gateway/dashboard)
    with a CWD-only override collapse to ``"default"`` for container sharing,
    yet an env may already be cached under the originating task_id; honor it
    instead of spawning a duplicate. Refreshes ``_last_activity`` on a hit.
    """
    for key in (effective_task_id, task_id):
        if key and key in _active_environments:
            _last_activity[key] = time.time()
            return _active_environments[key]
    return None


def _resolve_task_host_cwd(config: Dict[str, Any], task_id: Optional[str]) -> Optional[str]:
    """Host directory to bind-mount at ``/workspace`` for *task_id*'s container.

    Single owner of the cwd-mount policy for every creation site. Shared-
    container mode: the ``TERMINAL_CWD``-derived ``config["host_cwd"]``.
    Per-session isolation (docker + ``container_persistent: false``): only
    the SESSION's own registered workspace may mount — the process env var is
    a launch artifact that outlives the session that set it, so deriving a
    fresh session's mount from it would leak the previous session's directory.
    Overrides tagged ``cwd_source: "process"`` are refused for the same reason;
    ``cwd_source: "session"`` or untagged (ACP/RL) overrides mount.
    """
    if config.get("env_type") != "docker" or not config.get("docker_mount_cwd_to_workspace"):
        return None
    # Top-level CLI parent ("default") is a single-session process — legacy behavior.
    if not _docker_session_isolation_enabled() or _resolve_container_task_id(task_id) == "default":
        return config.get("host_cwd")
    overrides = resolve_task_overrides(task_id)
    candidate = overrides.get("cwd")
    if overrides.get("cwd_source") == "process" or not isinstance(candidate, str) or not candidate.strip():
        return None
    candidate = os.path.abspath(os.path.expanduser(candidate))
    # Must exist on the host and not already be an in-container path.
    if not os.path.isdir(candidate) or candidate.startswith(("/workspace", "/root")):
        return None
    return candidate


# One-shot guard for the config-fallback bridge: after the first attempt
# either TERMINAL_ENV is set or the import failed, so retrying is wasted work.
_terminal_config_bridge_attempted = False


def _ensure_terminal_env_bridged() -> None:
    """Backfill TERMINAL_* env vars from config.yaml when no launcher did.

    CLI, gateway and TUI/dashboard PTY launches bridge ``terminal.*`` into env vars
    at startup; processes that skip those paths (``hermes serve``, Desktop
    in-process agents, desktop cron ticker, ACP) would otherwise fall back to the
    local backend even when config selects docker — running on the host the user
    meant to sandbox. Explicit keys in the ``terminal`` section override matching
    env values (possibly stale from ``hermes setup``); env values for omitted keys
    are preserved. Without a terminal section an existing TERMINAL_ENV is kept and
    defaults are backfilled only when none is set. A per-turn terminal scope
    suppresses the bridge entirely: writing scope values into the process-global
    env would re-create the first-writer-wins cross-profile leak the scope fixes.

    terminal_tool reads ALL terminal settings from os.environ (TERMINAL_*). See #61115, #65696.
    """
    from tools.terminal_scope import get_terminal_scope

    if get_terminal_scope() is not None:
        return
    global _terminal_config_bridge_attempted
    if _terminal_config_bridge_attempted:
        return
    _terminal_config_bridge_attempted = True
    # Never let a config problem take the terminal tool down.
    with _quiet("terminal config → env fallback bridge failed"):
        from hermes_cli.config import apply_terminal_config_to_env, read_raw_config

        raw_config = read_raw_config()
        if isinstance(raw_config.get("terminal"), dict):
            apply_terminal_config_to_env(env=None, override=True)
        elif "TERMINAL_ENV" not in os.environ:
            apply_terminal_config_to_env(env=None, override=False)


# Default cwd per backend; anything else (container backends, plugins) is "/root".
_DEFAULT_CWD_BY_BACKEND = {"ssh": "~", "vercel_sandbox": _VERCEL_SANDBOX_DEFAULT_CWD}


def _resolve_config_cwd(env_type: str, mount_docker_cwd: bool) -> tuple:
    """``(cwd, host_cwd)`` from TERMINAL_CWD for *env_type*.

    Container backends are sanity-checked: with Docker cwd passthrough the host
    path is remapped to /workspace and tracked as host_cwd; otherwise host paths
    are discarded in favor of the backend default.
    """
    default_cwd = _safe_getcwd() if env_type == "local" else _DEFAULT_CWD_BY_BACKEND.get(env_type, "/root")
    cwd = _tenv("TERMINAL_CWD", default_cwd)
    from hermes_cli.config import _is_ssh_remote_tilde_cwd
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        candidate = os.path.abspath(os.path.expanduser(_tenv("TERMINAL_CWD") or _safe_getcwd()))
        if (
            _is_host_cwd(candidate)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif _is_container_backend(env_type) and cwd and _is_unusable_container_cwd(cwd) and cwd != default_cwd:
        logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                    "(host/relative path won't work in sandbox). Using %r instead.",
                    cwd, env_type, default_cwd)
        cwd = default_cwd
    return cwd, host_cwd


def _get_env_config() -> Dict[str, Any]:
    """Resolve the terminal configuration dict from TERMINAL_* env vars."""
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    _ensure_terminal_env_bridged()
    env_type = _tenv("TERMINAL_ENV", "local")
    mount_docker_cwd = _tenv_bool("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false")

    # Container/docker-only payloads are parsed only when such a backend is
    # selected: a stale or invalid Docker value bridged from config.yaml must
    # not make local terminal/execute_code unusable.
    if _is_container_backend(env_type):
        container_cpu = _parse_env_var("TERMINAL_CONTAINER_CPU", "1", float, "number")
        container_memory = _parse_env_var("TERMINAL_CONTAINER_MEMORY", "5120")
        container_disk = _parse_env_var("TERMINAL_CONTAINER_DISK", "51200")
    else:
        container_cpu, container_memory, container_disk = 1.0, 5120, 51200

    if env_type == "docker":
        docker_forward_env = _parse_env_var("TERMINAL_DOCKER_FORWARD_ENV", "[]", json.loads, "valid JSON")
        docker_volumes = _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")
        docker_env = _parse_env_var("TERMINAL_DOCKER_ENV", "{}", json.loads, "valid JSON")
        docker_extra_args = _parse_env_var("TERMINAL_DOCKER_EXTRA_ARGS", "[]", json.loads, "valid JSON")
        docker_shm_size = _tenv("TERMINAL_DOCKER_SHM_SIZE", "1g")
    else:
        docker_forward_env, docker_volumes, docker_env, docker_extra_args, docker_shm_size = [], [], {}, [], "1g"

    cwd, host_cwd = _resolve_config_cwd(env_type, mount_docker_cwd)

    return {
        "env_type": env_type,
        "modal_mode": coerce_modal_mode(_tenv("TERMINAL_MODAL_MODE", "auto")),
        "docker_image": _tenv("TERMINAL_DOCKER_IMAGE", default_image),
        "docker_forward_env": docker_forward_env,
        "singularity_image": _tenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": _tenv("TERMINAL_MODAL_IMAGE", default_image),
        "daytona_image": _tenv("TERMINAL_DAYTONA_IMAGE", default_image),
        "vercel_runtime": _tenv("TERMINAL_VERCEL_RUNTIME", "").strip(),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _parse_env_var("TERMINAL_TIMEOUT", "180"),
        "lifetime_seconds": _parse_env_var("TERMINAL_LIFETIME_SECONDS", "300"),
        # SSH-specific config
        "ssh_host": _tenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": _tenv("TERMINAL_SSH_USER", ""),
        "ssh_port": _parse_env_var("TERMINAL_SSH_PORT", "22"),
        "ssh_key": _tenv("TERMINAL_SSH_KEY", ""),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting; local is always opt-in. Per-backend env vars override.
        "ssh_persistent": _tenv_bool(
            "TERMINAL_SSH_PERSISTENT", _tenv("TERMINAL_PERSISTENT_SHELL", "true"),
        ),
        "local_persistent": _tenv_bool("TERMINAL_LOCAL_PERSISTENT", "false"),
        # Container resources (MB); ignored for local/ssh.
        "container_cpu": container_cpu,
        "container_memory": container_memory,
        "container_disk": container_disk,
        "container_persistent": _tenv_bool("TERMINAL_CONTAINER_PERSISTENT", "true"),
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": _tenv_bool("TERMINAL_DOCKER_RUN_AS_HOST_USER", "false"),
        "docker_snap_compat": _tenv_bool("TERMINAL_DOCKER_SNAP_COMPAT", "false"),
        "docker_network": _tenv_bool("TERMINAL_DOCKER_NETWORK", "true"),
        "docker_extra_args": docker_extra_args,
        "docker_shm_size": docker_shm_size,
        # Cross-process reuse: attach to a labeled container at startup
        # instead of starting fresh; false = per-process isolation.
        "docker_persist_across_processes": _tenv_bool("TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "true"),
        "docker_shared_container_key": _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip(),
        "docker_orphan_reaper": _tenv_bool("TERMINAL_DOCKER_ORPHAN_REAPER", "true"),
    }


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    while _cleanup_running:
        with _quiet("Error in cleanup thread", level=logging.WARNING):
            _cleanup_inactive_envs(_get_env_config()["lifetime_seconds"])
        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _env_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()


def _stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        try:
            _cleanup_thread.join(timeout=5)
        except (SystemExit, KeyboardInterrupt):
            pass


def _atexit_cleanup():
    """Stop the cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        logger.info("Shutting down %d remaining sandbox(es)...", len(_active_environments))
        # Snapshot BEFORE cleanup_all_environments empties the dict, then
        # block briefly so docker stop/rm completes before the interpreter
        # exits — otherwise daemon cleanup threads die mid-`docker stop` and
        # Exited containers pile up on the host.
        envs_to_wait = list(_active_environments.values())
        cleanup_all_environments()
        for env in envs_to_wait:
            wait_fn = getattr(env, "wait_for_cleanup", None)
            if wait_fn is not None:
                with _quiet("wait_for_cleanup raised on exit"):  # never block shutdown on a bad backend
                    wait_fn(timeout=15.0)
    # Workers of envs the idle reaper already detached are not in the registry (#86317).
    if "tools.environments.docker" in sys.modules:
        with _quiet("teardown drain raised on exit"):
            sys.modules["tools.environments.docker"].DockerEnvironment.wait_for_all_teardowns(timeout=15.0)

atexit.register(_atexit_cleanup)


def _command_requires_pipe_stdin(command: str) -> bool:
    """True when PTY mode would break a stdin-driven command: `gh auth login
    --with-token` waits for EOF on piped stdin, and under a PTY
    `process.submit()` only sends a newline, so it hangs forever."""
    normalized = " ".join(command.lower().split())
    return normalized.startswith("gh auth login") and "--with-token" in normalized


from tools.terminal_tool_guards import (
    _foreground_background_guidance, _safe_command_preview, _validate_workdir,
    gateway_lifecycle_block, self_repo_block,
)
from tools.terminal_tool_background import _YIELDED_NOTE, spawn_background_process, yield_to_background_handler
from tools.terminal_tool_result import finalize_foreground_result


def _resolve_notification_flag_conflict(*, notify_on_complete: bool, watch_patterns, background: bool) -> tuple:
    """Resolve notify_on_complete + watch_patterns both set: drop watch_patterns
    (combined they produce duplicate async notifications — one per match plus
    one on exit — that can spam the user long after the process ends).
    Returns ``(watch_patterns_to_use, conflict_note)``; note is "" without conflict."""
    if background and notify_on_complete and watch_patterns:
        return None, (
            "watch_patterns ignored because notify_on_complete=True; "
            "these two flags produce duplicate notifications when combined"
        )
    return watch_patterns, ""


# ---------------------------------------------------------------------------
# Streaming spill redaction (bounded-memory full-file sanitization)
# ---------------------------------------------------------------------------

# Chunk size for _stream_redact_spill: bytes per binary read.  The whole
# spill must never be loaded into RAM, so every read is a fixed-size slice.
_SPILL_STREAM_CHUNK_BYTES = 64 * 1024

# Candidate sensitive-record budget for _StreamingSpillRedactor.  A potential
# sensitive record (token run, KEY=value, Authorization value, JWT, ...) is
# buffered up to this many characters so the WHOLE record can be validated by
# the same redaction rules as the visible output — a greedy rule sees the
# complete record, so any secret inside it is masked in full.  A record
# longer than the budget is fail-closed: it is replaced with a redaction
# marker and the remainder is consumed WITHOUT ever emitting raw text, so a
# >64 KiB token can never leak its second half.
_SPILL_CONFIRM_LIMIT_CHARS = 256 * 1024

# Tail carry kept in the normal state of _StreamingSpillRedactor: large
# enough to cover the longest ANSI escape sequence (OSC <= ~1 KiB) and the
# longest sensitive prefix (ENV key width <= 110 chars) so a construct
# straddling a feed boundary is completed before any text is emitted.  This
# carry is a SYNTAX-BOUNDARY guard only — it is not a guarantee for
# arbitrarily long secrets (those are handled by the confirm budget /
# fail-closed path).
_SPILL_TAIL_CARRY_CHARS = 4 * 1024

# Redaction marker used when a candidate sensitive record exceeds the confirm
# budget (fail-closed) — the record's raw content is never partially emitted.
_SPILL_OVERSIZE_MARKER = "«redacted:secret…»"


def _derive_token_prefixes() -> tuple[str, ...]:
    """Literal token prefixes derived from agent/redact.py's prefix rules.

    Derived (not hand-maintained) so a future prefix added to the redaction
    rules is automatically detected by the streaming redactor.
    """
    from agent.redact import _extract_literal_prefix, _PREFIX_PATTERNS

    prefixes: list[str] = []
    for pattern in _PREFIX_PATTERNS:
        literal = _extract_literal_prefix(pattern)
        if len(literal) >= 3:
            prefixes.append(re.escape(literal))
        else:
            # Very short literals (e.g. "SG" from r"SG\.[...]") need at least
            # one following token char so plain prose is not flagged; the
            # confirm state still validates against the full prefix rule.
            prefixes.append(re.escape(literal) + r"[A-Za-z0-9_.\-=]")
    # Longest first so alternation prefers the more specific prefix.
    prefixes.sort(key=len, reverse=True)
    return tuple(prefixes)


def _build_potential_start_re() -> "re.Pattern":
    """Regex that finds the start of any potentially sensitive record.

    Each named group maps to a confirm-state kind whose syntactic terminator
    is known; the buffered record is then validated by
    ``redact_terminal_output`` itself, so the streaming path applies EXACTLY
    the same rules as the visible output.  The token group carries the same
    ``(?<![A-Za-z0-9_-])`` guard as agent/redact._PREFIX_RE.

    KEY=value / KEY: value candidates are NOT part of this regex: their
    lower-case key classes backtrack catastrophically on long plain-text
    runs, so they are detected by the linear keyword pre-screen
    :func:`_find_key_value_start` instead (mirroring agent/redact's own
    pre-check discipline).
    """
    return re.compile(
        r"(?P<token>(?<![A-Za-z0-9_-])"
        + "|".join(_derive_token_prefixes())
        + r"[A-Za-z0-9_.\-=]*)"
        r"|(?P<auth>(?:Proxy-)?Authorization\s*:\s*)"
        r"|(?P<header>(?:x-api-key|x-goog-api-key|api-key|apikey|x-api-token|x-auth-token|x-access-token)\s*:\s*)"
        r"|(?P<json>\"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|raw_secret|secret_input|key_material)\"\s*:\s*\")"
        r"|(?P<jwt>eyJ[A-Za-z0-9_-]*)"
        r"|(?P<telegram>(?:bot)?\d{8,}:)"
        r"|(?P<phone>\+[1-9]\d{0,14})"
        r"|(?P<db>(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:]+:)"
        r"|(?P<bareurl>(?:https?|wss?|git|ssh|ftp|ftps|sftp)://[^\s:@/]*@)",
        re.IGNORECASE,
    )


_KEY_CHAR_RE = re.compile(r"[A-Za-z0-9_.\-]")


def _find_key_value_start(text: str) -> "int | None":
    """Linear scan for a ``KEY=value`` / ``KEY: value`` candidate start.

    Mirrors agent/redact's config-key passes: a secret keyword
    (API_KEY/TOKEN/SECRET/...) inside a key followed by ``=`` or ``:``.
    Returns the index of the key start, or None.  This linear pre-screen
    avoids running the redaction rules' backtrack-heavy ``_CFG_*`` regexes on
    long plain-text runs; the expensive regexes only ever run on the short
    buffered candidate record in the confirm state.
    """
    from agent.redact import _CFG_SECRET_WORD_RE, _SENSITIVE_QUERY_PARAMS

    for m in _CFG_SECRET_WORD_RE.finditer(text):
        start = m.start()
        while start > 0 and _KEY_CHAR_RE.match(text[start - 1]):
            start -= 1
        end = m.end()
        while end < len(text) and _KEY_CHAR_RE.match(text[end]):
            end += 1
        j = end
        while j < len(text) and text[j] in " \t":
            j += 1
        if j < len(text) and text[j] in "=:":
            return start
    # Exact-match sensitive body/query keys (``code=``, ``session=``,
    # ``signature=``, ``x-amz-signature=``, ``id_token=``, ...).  Canonical
    # redaction masks these only inside a PURE k=v&k=v form body
    # (agent/redact._redact_form_body); buffering the whole line in confirm
    # and letting redact_terminal_output decide keeps the streaming result
    # byte-identical to the canonical pass.
    for m in _BODY_KEY_RE.finditer(text):
        if m.group(1).casefold() in _SENSITIVE_QUERY_PARAMS:
            return m.start()
    return None


# Exact-match sensitive body/query key scan used by _find_key_value_start.
_BODY_KEY_RE = re.compile(r"([A-Za-z0-9_.\-]+)=")

# String-sequence terminators for _IncrementalAnsiStripper (OSC: BEL or ST;
# DCS/SOS/PM/APC: ST only; 8-bit OSC: BEL or 0x9c).
_OSC_END_RE = re.compile(r"\x07|\x1b\\")
_DCS_END_RE = re.compile(r"\x1b\\")
_OSC_8BIT_END_RE = re.compile(r"\x07|\x9c")


# ---------------------------------------------------------------------------
# Incremental ANSI stripping (must run BEFORE the secret state machine)
# ---------------------------------------------------------------------------


class _IncrementalAnsiStripper:
    """Incremental ANSI/ECMA-48 stripper that mirrors strip_ansi().

    Runs BEFORE ``_StreamingSpillRedactor`` so the secret state machine only
    ever sees clean plain text.  This closes the ANSI-insertion bypass: a
    token split by an escape sequence (``sk-\\x1b[31mAAA\\x1b[0m``) is
    reassembled by the stripper BEFORE redaction, exactly like the canonical
    ``redact_terminal_output(strip_ansi(...))`` ordering.

    The stripper is stateful across chunk boundaries: a CSI/OSC/DCS sequence
    split by a ``feed()`` boundary is completed by the next feed, never
    independently stripped per chunk.  Unterminated sequences at EOF are
    preserved verbatim, matching strip_ansi()'s behaviour on a whole string
    (so differential tests stay byte-identical); a pathological unterminated
    string sequence longer than the budget is dropped wholesale to keep
    memory bounded — such input is outside the ECMA-48 grammar anyway.
    """

    _OSC_BUDGET = 8 * 1024

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        if self._pending or _HAS_ANSI_BYTE.search(text):
            self._pending += text
            return self._drain()
        # Fast path: no ESC/C1 bytes at all — nothing to strip.
        return text

    def flush(self) -> str:
        # Unterminated sequences are preserved (strip_ansi parity).
        out = self._pending
        self._pending = ""
        return out

    def _drain(self) -> str:
        out: list[str] = []
        while True:
            # STRING-type sequences (OSC/DCS/SOS/PM/APC) must be completed as
            # a whole: the generic regex would otherwise split an incomplete
            # OSC via its single-byte Fp fallback (``\x1b]`` consumed alone,
            # leaking ``0;title``).  Detect the sequence head explicitly and
            # wait for its terminator (BEL/ST/0x9c) across feed boundaries.
            if self._pending:
                ch0 = self._pending[0]
                if ch0 == "\x1b" and len(self._pending) >= 2:
                    ch1 = self._pending[1]
                    if ch1 == "]":
                        end = _OSC_END_RE.search(self._pending[1:])
                        if end:
                            self._pending = self._pending[end.end() + 1:]
                            continue
                        if len(self._pending) > self._OSC_BUDGET:
                            self._pending = ""
                            break
                        break
                    if ch1 in "PX^_":
                        end = _DCS_END_RE.search(self._pending[1:])
                        if end:
                            self._pending = self._pending[end.end() + 1:]
                            continue
                        if len(self._pending) > self._OSC_BUDGET:
                            self._pending = ""
                            break
                        break
                elif ch0 == "\x9d":
                    end = _OSC_8BIT_END_RE.search(self._pending[1:])
                    if end:
                        self._pending = self._pending[end.end() + 1:]
                        continue
                    if len(self._pending) > self._OSC_BUDGET:
                        self._pending = ""
                        break
                    break
            m = _ANSI_ESCAPE_RE.search(self._pending)
            if m:
                out.append(self._pending[:m.start()])
                self._pending = self._pending[m.end():]
                continue
            # No complete sequence in the buffer: emit everything before the
            # last escape candidate (that candidate may still complete next
            # feed), keeping the buffer bounded.
            escape_idx = -1
            for i in range(len(self._pending) - 1, -1, -1):
                ch = self._pending[i]
                if ch in "\x1b\x9b\x9d" or "\x80" <= ch <= "\x9f":
                    escape_idx = i
                    break
            if escape_idx < 0:
                out.append(self._pending)
                self._pending = ""
                break
            if escape_idx > 0:
                out.append(self._pending[:escape_idx])
                self._pending = self._pending[escape_idx:]
                continue
            # pending starts with an escape candidate; wait for completion
            # unless it is a pathological unterminated string sequence.
            if len(self._pending) > self._OSC_BUDGET:
                self._pending = ""
                break
            break
        return "".join(out)


class _StreamingSpillRedactor:
    """Incremental fail-closed secret redactor for spill streaming.

    The redaction rules (agent/redact.py) contain UNBOUNDED constructs
    (``sk-...{10,}`` tokens, ``KEY=\\S+`` values, JWTs, PEM blocks), so no
    fixed-size window can guarantee that a secret crossing a window boundary
    is fully masked.  This redactor therefore works on RECORD boundaries:

    - normal: scans for a potential sensitive start (a bounded literal
      prefix); plain text is streamed out directly (keeping a small tail
      carry so a prefix or ANSI sequence straddling a feed boundary is not
      split).
    - confirm: a candidate sensitive record is buffered until its syntactic
      terminator (whitespace/quote for values, a non-token character for
      tokens, the closing quote for JSON), then the WHOLE record is validated
      by ``redact_terminal_output`` — the same rules as the visible output —
      so a greedy rule sees the complete record and masks any secret in it.
    - fail-closed: a candidate record longer than
      ``_SPILL_CONFIRM_LIMIT_CHARS`` is NEVER partially emitted; it is
      replaced by ``_SPILL_OVERSIZE_MARKER`` and the remainder is consumed
      (masked) until the terminator.  A >64 KiB token therefore cannot leak
      its second half, and a "potential start" that turns out not to be a
      secret (e.g. ``sk-ill``) is still emitted verbatim after validation.
    - PEM blocks (the only sensitive construct that spans line boundaries)
      use a dedicated cross-feed state from BEGIN to END.
    """

    _START_RE: "re.Pattern | None" = None
    _PEM_BEGIN_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----")
    _PEM_END_RE = re.compile(r"-----END[A-Z ]*PRIVATE KEY-----")

    def __init__(self, command: str):
        self._command = command
        self._state = "normal"   # normal | confirm | mask | pem
        self._kind: "str | None" = None
        self._buf = ""           # confirm buffer / PEM budget counter (raw)
        self._carry = ""         # confirmed-plain tail held for ANSI safety
        self._pending = ""       # unprocessed decoded input
        self._pem_emitted = False
        # ANSI stripping happens BEFORE the secret state machine so the
        # redactor only ever sees clean plain text (canonical ordering:
        # strip_ansi -> redact).  An escape sequence inserted inside a token
        # is removed first, so it can never split or hide a secret.
        self._ansi = _IncrementalAnsiStripper()

    @classmethod
    def _start_re(cls) -> "re.Pattern":
        if cls._START_RE is None:
            cls._START_RE = _build_potential_start_re()
        return cls._START_RE

    # -- public ------------------------------------------------------------

    def feed(self, text: str) -> str:
        """Process a decoded chunk; return the text that is safe to emit."""
        text = self._ansi.feed(text)
        if not text and not self._pending:
            return ""
        out: list[str] = []
        self._pending += text
        while self._pending:
            if self._state == "pem":
                if not self._step_pem(out):
                    break
            elif self._state == "mask":
                if not self._step_mask(out):
                    break
            elif self._state == "confirm":
                if not self._step_confirm(out):
                    break
            else:
                if not self._step_normal(out):
                    break
        return "".join(out)

    def flush(self) -> str:
        """Finalize the stream at EOF; return remaining safe-to-emit text."""
        # Any ANSI sequence still pending is finalized first (matching
        # strip_ansi on a whole string), then the redactor state is closed.
        tail = self._ansi.flush()
        out: list[str] = []
        self._pending += tail
        while self._pending:
            if self._state == "pem":
                if not self._step_pem(out):
                    break
            elif self._state == "mask":
                if not self._step_mask(out):
                    break
            elif self._state == "confirm":
                if not self._step_confirm(out):
                    break
            else:
                if not self._step_normal(out):
                    break
        if self._state == "pem":
            # EOF inside a PEM block: the block is unterminated.  Drop ALL
            # buffered body/carry/pending material — no PEM fragment (head,
            # middle, or tail) may ever reach the durable spill — and emit a
            # single explicit marker.  This is fail-closed and intentionally
            # stricter than the canonical regex (which can only replace a
            # complete BEGIN..END block).
            if not self._pem_emitted:
                out.append("[REDACTED PRIVATE KEY]")
            self._pending = ""
            self._buf = ""
            self._carry = ""
            self._state = "normal"
        elif self._state == "confirm":
            if len(self._buf) > _SPILL_CONFIRM_LIMIT_CHARS:
                if self._carry:
                    out.append(self._sanitize_plain(self._carry))
                    self._carry = ""
                out.append(_SPILL_OVERSIZE_MARKER)
            else:
                if self._carry:
                    out.append(self._sanitize_plain(self._carry))
                    self._carry = ""
                out.append(self._finalize_record())
            self._state = "normal"
            self._buf = ""
        elif self._state == "mask":
            # Marker already emitted for the oversize record; nothing more.
            self._state = "normal"
        if self._pending:
            out.append(self._sanitize_plain(self._pending))
            self._pending = ""
        return "".join(out)

    # -- state steps -------------------------------------------------------

    def _step_normal(self, out: list[str]) -> bool:
        m = self._start_re().search(self._pending)
        pem = self._PEM_BEGIN_RE.search(self._pending)
        cfg = _find_key_value_start(self._pending)
        candidates = []
        if m is not None:
            candidates.append((m.start(), "start"))
        if pem is not None:
            candidates.append((pem.start(), "pem"))
        if cfg is not None:
            candidates.append((cfg, "cfg"))
        if not candidates:
            # No sensitive candidate: stream plain text out, keeping only a
            # small tail carry for ANSI safety.
            safe = max(0, len(self._pending) - _SPILL_TAIL_CARRY_CHARS)
            if safe:
                out.append(self._sanitize_plain(self._pending[:safe]))
                self._pending = self._pending[safe:]
                return True
            return False
        # At the same position, the literal-prefix candidates (auth/header/
        # token/...) are more precise than the cfg keyword pre-screen (e.g.
        # "Authorization:" must not be treated as a bare KEY: value).
        _PRIORITY = {"start": 0, "cfg": 1, "pem": 2}
        pos, kind = min(candidates, key=lambda c: (c[0], _PRIORITY[c[1]]))
        pre = self._pending[:pos]
        safe = max(0, len(pre) - _SPILL_TAIL_CARRY_CHARS)
        if safe:
            out.append(self._sanitize_plain(pre[:safe]))
        self._carry = pre[safe:]
        if kind == "pem":
            # The prefix text is confirmed plain (no earlier candidate), so
            # emit it whole and mask from BEGIN to END.
            if self._carry:
                out.append(self._sanitize_plain(self._carry))
                self._carry = ""
            self._pending = self._pending[pos:]
            self._state = "pem"
            self._pem_emitted = False
            self._buf = ""
            return True
        if kind == "cfg":
            # KEY=value / KEY: value candidate from the linear pre-screen.
            self._buf = self._pending[pos:]
            self._kind = "cfg"
            self._pending = ""
            self._state = "confirm"
            return True
        # kind == "start": literal-prefix candidate (token/auth/header/...).
        self._buf = m.group(0)
        self._kind = m.lastgroup
        self._pending = self._pending[m.end():]
        self._state = "confirm"
        return True

    def _step_confirm(self, out: list[str]) -> bool:
        term = self._find_terminator(self._pending, self._kind)
        if term is not None:
            self._buf += self._pending[:term]
            if self._carry:
                out.append(self._sanitize_plain(self._carry))
                self._carry = ""
            out.append(self._finalize_record())
            self._pending = self._pending[term:]
            self._state = "normal"
            return True
        self._buf += self._pending
        self._pending = ""
        if len(self._buf) > _SPILL_CONFIRM_LIMIT_CHARS:
            # Fail-closed: never emit an unconfirmed raw tail of an
            # unbounded secret — emit the confirmed-plain carry, mask the
            # whole record with the marker, and keep consuming (still
            # masked) until its syntactic terminator.
            if self._carry:
                out.append(self._sanitize_plain(self._carry))
                self._carry = ""
            out.append(_SPILL_OVERSIZE_MARKER)
            self._state = "mask"
            self._buf = ""
        return False

    def _step_mask(self, out: list[str]) -> bool:
        term = self._find_terminator(self._pending, self._kind)
        if term is not None:
            self._pending = self._pending[term:]
            self._state = "normal"
            return True
        self._pending = ""
        return False

    def _step_pem(self, out: list[str]) -> bool:
        end = self._PEM_END_RE.search(self._pending)
        if end:
            if not self._pem_emitted:
                out.append("[REDACTED PRIVATE KEY]")
                self._pem_emitted = True
            self._pending = self._pending[end.end():]
            self._state = "normal"
            return True
        # The END marker may straddle a feed boundary: keep a tail of at
        # least the marker length so a split marker is completed next feed.
        keep = min(len(self._pending), 64)
        consumed = self._pending[:-keep] if keep else self._pending
        self._buf += consumed
        self._pending = self._pending[-keep:] if keep else ""
        if len(self._buf) > _SPILL_CONFIRM_LIMIT_CHARS:
            # Fail-closed for an absurd/never-terminating PEM block: emit the
            # marker once and keep consuming — no raw key material is ever
            # written.  Reset the counter; we are already masking.
            if not self._pem_emitted:
                out.append("[REDACTED PRIVATE KEY]")
                self._pem_emitted = True
            self._buf = ""
        return False

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _sanitize_plain(text: str) -> str:
        from tools.ansi_strip import strip_ansi

        return strip_ansi(text)

    def _finalize_record(self) -> str:
        """Validate the buffered candidate record with the real redaction rules.

        ``force=True`` makes the durable spill a MANDATORY safety boundary:
        even when the global model-output redaction is disabled
        (``HERMES_REDACT_SECRETS=false`` / ``security.redact_secrets: false``),
        the persistent spill is still sanitized.  The model-visible window
        keeps main's original semantics (it may respect the global opt-out);
        the durable file must never.
        """
        from agent.redact import redact_terminal_output

        return redact_terminal_output(
            self._sanitize_plain(self._buf), self._command, force=True,
        )

    @staticmethod
    def _find_terminator(text: str, kind: "str | None") -> "int | None":
        if kind in ("token", "jwt", "telegram"):
            m = re.search(r"[^A-Za-z0-9_.\-=]", text)
            return m.start() if m else None
        if kind == "phone":
            m = re.search(r"[^0-9]", text)
            return m.start() if m else None
        if kind == "auth":
            # Authorization values are "<scheme> <token>" (or a bare token);
            # buffer to EOL/quote so the WHOLE header is validated at once —
            # agent/redact masks the token and preserves the scheme, and a
            # buffer cut after the first whitespace would let the token leak
            # into the normal state.
            m = re.search(r"[\r\n\"']", text)
            return m.start() if m else None
        m = re.search(r"[ \t\r\n\"']", text)
        return m.start() if m else None


def _stream_redact_spill(
    src_path: str, dest_path: str, command: str
) -> tuple[str, int]:
    """Stream-sanitize *src_path* into *dest_path* with bounded memory.

    The full output is never loaded into RAM.  The file is read in fixed-size
    binary chunks, decoded incrementally (a UTF-8 multi-byte character
    straddling a chunk boundary is completed by the incremental decoder), and
    fed to :class:`_StreamingSpillRedactor`, which emits only text that has
    passed the same ``strip_ansi`` + ``redact_terminal_output`` rules as the
    visible output — while never emitting the unconfirmed raw tail of a
    record that may still grow.

    The sanitized result is written to a ``.partial`` temp file in the same
    directory as *dest_path* (same volume, so ``os.replace`` is atomic),
    flushed/fsynced, then atomically renamed onto *dest_path*: the durable
    path only ever holds sanitized content, even if the process crashes
    mid-stream (only a ``.partial`` file is left, never a raw spill).

    Returns ``(final_path, total_chars)`` — *total_chars* is the true Unicode
    character count of the raw stream (counted at decode time, before
    stripping/redaction), matching the terminal pipeline's
    ``output_total_chars`` convention (characters, not bytes).

    On failure the temporary sanitized file is removed and the exception
    propagates; the caller drops the spill handle and unlinks the raw source,
    so no unredacted file is left behind.
    """
    import codecs
    import tempfile

    src = Path(src_path)
    dest = Path(dest_path)
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Temp file in the SAME directory as the final dest so os.replace is
    # atomic (same volume).  The ".partial" suffix marks an in-progress
    # sanitized file that must never be surfaced as full_output_path.
    fd, tmp_name = tempfile.mkstemp(
        prefix=dest.name + ".tmp-", suffix=".partial", dir=str(parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fout:
            with open(src, "rb") as fin:
                redactor = _StreamingSpillRedactor(command)
                total_chars = 0
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    raw = fin.read(_SPILL_STREAM_CHUNK_BYTES)
                    if not raw:
                        break
                    decoded = decoder.decode(raw)
                    total_chars += len(decoded)
                    cleaned = redactor.feed(decoded)
                    if cleaned:
                        fout.write(cleaned.encode("utf-8", errors="replace"))
                # Final partial chunk (flush the incremental decoder) + any
                # record still open at EOF.
                decoded = decoder.decode(b"", final=True)
                total_chars += len(decoded)
                cleaned = redactor.feed(decoded)
                tail = redactor.flush()
                if cleaned or tail:
                    fout.write((cleaned + tail).encode("utf-8", errors="replace"))
                fout.flush()
                os.fsync(fout.fileno())
        # Atomic swap: the durable path only ever holds sanitized content.
        os.replace(tmp_path, dest)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return str(dest), total_chars


def _new_elevated_spill_path() -> str:
    """Return a fresh durable spill path for a sanitized elevated output.

    The path is generated here (not by the executor) and is only ever
    returned as ``full_output_path`` AFTER the sanitized spill has been
    atomically written to it — the raw staged output never occupies it.
    """
    from hermes_constants import get_hermes_home

    spill_dir = get_hermes_home() / "cache" / "terminal-output"
    spill_dir.mkdir(parents=True, exist_ok=True)
    return str(
        spill_dir
        / f"out-elev-{int(time.time())}-{os.getpid()}-{os.urandom(4).hex()}.log"
    )


def _cleanup_raw_output(raw_path: str) -> None:
    """Remove a staged raw elevated output and its throwaway directory."""
    try:
        raw = Path(raw_path)
        if raw.exists():
            raw.unlink()
        parent = raw.parent
        if parent.is_dir():
            shutil.rmtree(parent, ignore_errors=True)
    except OSError:
        pass


def _resolve_command_cwd(
    *,
    workdir: Optional[str],
    default_cwd: str,
    session_key: Optional[str] = None,
    env_type: Optional[str] = None,
) -> str:
    """cwd for a command: explicit ``workdir`` > the session's own cwd record >
    ``default_cwd``.

    The record is written after every completed command of THIS session, so
    it is the session's ``cd`` state with no shared-env ambiguity. On
    container backends a recorded HOST path (a desktop/TUI surface registering
    its workspace) is unusable in the sandbox — ``cd <host path>`` fails with
    exit 126 — so it is discarded in favor of ``default_cwd``.

    Same guard class as the env-creation sanitizers (#50636, #54447); this is the per-command sibling site.
    """
    if workdir:
        return workdir
    recorded = get_session_cwd(session_key)
    if recorded and _is_container_backend(env_type) and _is_unusable_container_cwd(recorded):
        logger.info(
            "Ignoring recorded session cwd %r for %s backend "
            "(host/relative path won't work in sandbox). Using %r instead.",
            recorded, env_type, default_cwd,
        )
        return default_cwd
    return recorded or default_cwd


def _error_json(error: str, *, exit_code: int = -1, status: Optional[str] = None, **extra) -> str:
    """The terminal error envelope: ``output``/``exit_code``/``error`` (+ ``status``, extras)."""
    body: Dict[str, Any] = {"output": "", "exit_code": exit_code, "error": error}
    if status is not None:
        body["status"] = status
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


def _fatal_error_json(e: BaseException) -> str:
    """Log the traceback and return the redacted error+traceback envelope.

    Exception text can embed the failing command line (and any secrets inline
    in it), so both fields are force-redacted before reaching the model.
    """
    import traceback
    tb_str = traceback.format_exc()
    logger.error("terminal_tool exception:\n%s", tb_str)
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "error": _redact_terminal_error_text(f"Failed to execute command: {e}"),
        "traceback": _redact_terminal_error_text(tb_str),
        "status": "error"
    }, ensure_ascii=False)


class _Rejected(Exception):
    """Carries a finished tool-result JSON out of the planning/guard helpers, so
    each early-return site is one ``raise`` instead of an isinstance-checked
    ``str | plan`` union at the caller."""

    def __init__(self, result_json: str):
        super().__init__(result_json)
        self.result_json = result_json


@dataclass
class _ApprovalVerdict:
    """Outcome of the pre-exec guard pass.

    ``note`` is the audit note attached to the result. ``approved_run`` is True
    when the user explicitly approved (or pre-confirmed via ``force``); it drives
    the clean-interrupt-slate clear before ``env.execute`` so an approved command
    can't be SIGINT-killed by a bit that landed during the approval-wait.
    """
    note: Optional[str] = None
    approved_run: bool = False


def _run_approval_guards(command: str, env_type: str, config: Dict[str, Any], *, force: bool) -> _ApprovalVerdict:
    """Run tirith + dangerous-command guards; ``force`` skips them entirely.
    Raises :class:`_Rejected` when the command may not run (denied, or pending
    gateway approval)."""
    if force:
        return _ApprovalVerdict(approved_run=True)
    approval = _check_all_guards(command, env_type, has_host_access=_docker_has_host_access(config))
    if not approval["approved"]:
        if approval.get("status") == "pending_approval":  # gateway ask mode
            raise _Rejected(_error_json(
                "", status="pending_approval",
                approval_pending=True,
                command=approval.get("command", command),
                description=approval.get("description", "command flagged"),
                pattern_key=approval.get("pattern_key", ""),
                smart_denied=approval.get("smart_denied", False),
                allow_permanent=approval.get("allow_permanent", True),
            ))
        desc = approval.get("description", "command flagged")
        fallback_msg = (
            f"Command denied: {desc}. "
            "Use the approval prompt to allow it, or rephrase the command."
        )
        raise _Rejected(_error_json(approval.get("message", fallback_msg), status="blocked"))
    desc = approval.get("description", "flagged as dangerous")
    if approval.get("user_approved"):
        return _ApprovalVerdict(
            note=f"Command required approval ({desc}) and was approved by the user.",
            approved_run=True,
        )
    if approval.get("smart_approved"):
        return _ApprovalVerdict(note=f"Command was flagged ({desc}) and auto-approved by smart approval.")
    return _ApprovalVerdict()


@dataclass
class _ExecPlan:
    """Per-call execution parameters resolved before any environment is touched."""
    config: Dict[str, Any]
    env_type: str
    effective_task_id: str
    image: str
    cwd: str
    host_cwd: Optional[str]
    effective_timeout: int
    # Set when a foreground call asked for more than FOREGROUND_MAX_TIMEOUT and was promoted to a
    # tracked background process instead of being refused (the requested seconds, for the note).
    promoted_from_foreground_timeout: Optional[int] = None


_PROMOTED_NOTE = (
    "Requested foreground timeout {requested}s exceeds the {cap}s cap, so this command was started as a "
    "tracked background process with notify_on_complete=true instead of being refused. Do NOT re-run it. "
    "Its completion (exit code + output tail) arrives as a notification; poll with "
    "process(action=\"poll\", session_id=...) if you need it sooner."
)


def _plan_execution(
    command: Any, *, task_id: Optional[str], timeout: Optional[int],
    background: bool, pty: bool, elevated: bool, _host_local: bool,
) -> _ExecPlan:
    """Resolve backend, env-cache key, image, cwd and timeout for one call.

    Raises :class:`_Rejected` when the call is rejected up front (non-string
    command, non-positive or over-cap timeout, a foreground command that must
    run in the background).
    """
    if not isinstance(command, str):
        logger.warning("Rejected invalid terminal command value: %s", type(command).__name__)
        raise _Rejected(_error_json(
            f"Invalid command: expected string, got {type(command).__name__}", status="error",
        ))

    config = _get_env_config()
    env_type = "local" if _host_local else config["env_type"]

    # Elevated execution (Windows UAC) is a local-host capability.  Fail
    # fast BEFORE any backend is created/connected/recovered: a non-local
    # env_type must never initialize Docker/SSH/Daytona/Vercel-sandbox
    # machinery only to be rejected later.  background/pty are also
    # unsupported with elevated, so block them here too — before any
    # side effect (environment creation, cleanup thread, process spawn).
    # The gate is re-checked after the shared approval/command guards as
    # defense-in-depth, so no path can bypass it.
    if elevated and (env_type != "local" or background or pty):
        if env_type != "local":
            raise _Rejected(_error_json(
                (
                    "elevated=true requires the local Windows backend "
                    f"(current env_type={env_type!r}). Elevated execution "
                    "is only supported for env_type='local' and is blocked "
                    "for Docker/SSH/Daytona/Vercel-sandbox backends to "
                    "prevent UAC from touching the host from inside a "
                    "sandbox."
                ),
                status="blocked",
            ))
        if background:
            raise _Rejected(_error_json(
                "elevated=true cannot be combined with background=true. "
                "Elevated commands run synchronously with UAC.",
                status="error",
            ))
        raise _Rejected(_error_json(
            "elevated=true cannot be combined with pty=true. Elevated "
            "commands use file-based output capture, not PTY.",
            status="error",
        ))

    # Fail closed under a refusal scope: the routed profile's terminal
    # policy could not be resolved, so running with the launch process's
    # ambient policy is forbidden.
    # See #68559.
    if not _host_local:
        from tools.terminal_scope import enforce_no_refusal

        enforce_no_refusal()

    effective_task_id = _resolve_container_task_id(task_id)
    if _host_local:
        # Control-plane children run beside this interpreter, never inside
        # the configured Docker/SSH backend; keep their env cache separate.
        effective_task_id = f"host-local-{effective_task_id}"

    # Per-task overrides (RL/benchmark envs, ACP workspace cwd) win over
    # the global env-var config; ``resolve_task_overrides`` reads the raw
    # task id first, then the collapsed container id.
    overrides = resolve_task_overrides(task_id)
    image = _select_image(env_type, overrides, config)

    cwd = overrides.get("cwd") or get_session_cwd(task_id) or config["cwd"]
    host_cwd = _resolve_task_host_cwd(config, task_id)
    # config["cwd"] was sanitized for container backends in _get_env_config
    # but an override / session record is raw: a host path would reach
    # `docker run -w` and fail with exit 125. Re-apply the guard to the
    # resolved cwd; when the host path IS this session's mounted workspace,
    # remap to /workspace instead of discarding it.
    if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
        remapped = "/workspace" if host_cwd else config["cwd"]
        if cwd != remapped:
            logger.info(
                "Remapping host/relative cwd override %r for %s backend "
                "(won't exist in sandbox). Using %r instead.",
                cwd, env_type, remapped,
            )
        cwd = remapped
    # Reject non-positive timeouts before deadline math: ``timeout or
    # default`` would silently turn 0 into the default, and a negative
    # value is truthy and would fire an immediate "-Ns" timeout.
    if timeout is not None and timeout <= 0:
        raise _Rejected(tool_error(f"timeout must be a positive number of seconds (got {timeout})."))
    promoted = None
    if not background:
        # An over-cap foreground timeout is a bounded job the caller wants to wait for (test suites,
        # builds). Refusing it only bought a mechanical retry: 454 refusals in one run, every one
        # re-sent lower/split/background. Promote to a tracked background process instead; the
        # caller is told in the result. The `&`/nohup/server guidance below stays a refusal: those
        # need the command itself rewritten, which the tool cannot do safely.
        # The detachment guidance applies whether or not the call is promoted: a promoted `cmd &`
        # would start a tracked shell that exits at once while its payload runs untracked.
        guidance = _foreground_background_guidance(command)
        if guidance:
            raise _Rejected(_error_json(guidance, status="error"))
        if timeout and timeout > FOREGROUND_MAX_TIMEOUT:
            promoted = timeout

    return _ExecPlan(
        config=config, env_type=env_type, effective_task_id=effective_task_id,
        image=image, cwd=cwd, host_cwd=host_cwd, effective_timeout=timeout or config["timeout"],
        promoted_from_foreground_timeout=promoted,
    )


_PROMOTED_NOTE_POLL_ONLY = (
    "Requested foreground timeout {requested}s exceeds the {cap}s cap, so this command was started as a "
    "tracked background process instead of being refused. Do NOT re-run it. This session cannot receive "
    "completion notifications, so poll it with process(action=\"poll\", session_id=...) until it exits."
)


def _with_promoted_note(result_json: str, requested_timeout: int) -> str:
    """Attach the foreground->background promotion note to a spawn result (unchanged on error). The
    note only promises a notification when the spawn actually kept notify_on_complete (finite sessions
    such as one-shot runners cannot route one back; the spawn already said so and cleared the flag)."""
    try:
        data = json.loads(result_json)
    except (TypeError, ValueError):
        return result_json
    if not isinstance(data, dict) or data.get("error"):
        return result_json
    template = _PROMOTED_NOTE if data.get("notify_on_complete") else _PROMOTED_NOTE_POLL_ONLY
    data["promoted_from_foreground"] = template.format(requested=requested_timeout, cap=FOREGROUND_MAX_TIMEOUT)
    return json.dumps(data, ensure_ascii=False)


def _acquire_env(plan: _ExecPlan, task_id: Optional[str]) -> Any:
    """Cached env for the task, else create it under the per-task creation lock.

    Concurrent calls for the same task_id wait for the first sandbox instead
    of each creating their own; the cache is re-checked under that lock.
    Raises :class:`_Rejected` with the ``"disabled"`` envelope when creation
    raises ImportError.
    """
    _start_cleanup_thread()
    env_type, eff = plan.env_type, plan.effective_task_id

    with _env_lock:
        env: Any = _lookup_active_env(eff, task_id)
    if env is not None:
        return env

    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(eff, threading.Lock())

    with task_lock:
        with _env_lock:
            env = _lookup_active_env(eff, task_id)
        if env is not None:
            return env

        if env_type == "singularity":
            _check_disk_usage_warning()
        logger.info("Creating new %s environment for task %s...", env_type, eff[:8])
        try:
            new_env = _create_configured_env(
                plan.config, env_type, image=plan.image, cwd=plan.cwd,
                timeout=plan.effective_timeout, task_id=eff, host_cwd=plan.host_cwd,
                local_config=(
                    {"persistent": plan.config.get("local_persistent", False)}
                    if env_type == "local" else None
                ),
            )
        except ImportError as e:
            raise _Rejected(_error_json(
                _redact_terminal_error_text(f"Terminal tool disabled: environment creation failed ({e})"),
                status="disabled",
            ))

        with _env_lock:
            _active_environments[eff] = new_env
            _last_activity[eff] = time.time()
        logger.info("%s environment ready for task %s", env_type, eff[:8])
        return new_env


def _yield_kwargs(command: str, **ctx) -> dict:
    """``env.execute`` kwargs enabling yield-to-background (local backend only)."""
    handler = yield_to_background_handler(command=command, **ctx)
    return {"yield_handler": handler} if handler is not None else {}


def _run_foreground(
    command: str, env: Any, plan: _ExecPlan, *,
    task_id: Optional[str], session_id: Optional[str], session_key: str,
    workdir: Optional[str], approval_note: Optional[str], clear_interrupt: bool,
    elevated: bool = False,
) -> str:
    """Execute in the foreground with retry on transient errors, then finalize."""
    max_retries = 3
    env_type, eff, effective_timeout = plan.env_type, plan.effective_task_id, plan.effective_timeout

    # Clean interrupt slate for an approved command, ONCE before the retry
    # loop: drop a stale bit that landed during the approval-wait so it
    # can't SIGINT the just-approved run. Do NOT re-clear inside the loop —
    # a genuine interrupt during the backoff sleep must survive and abort
    # the next attempt (rc 130).
    if clear_interrupt:
        from tools.interrupt import clear_current_thread_interrupt
        clear_current_thread_interrupt()

    if elevated:
        # Elevated execution reuses the SAME foreground result pipeline
        # as env.execute: cwd resolution, unified output post-processing
        # (redaction, truncation, spill, exit notes, verification
        # evidence) below all apply. We only swap the executor.
        from tools.admin_executor import execute_elevated as _execute_elevated
        command_cwd = _resolve_command_cwd(
            workdir=workdir,
            default_cwd=plan.cwd,
            session_key=session_key,
        )
        elevated_result = _execute_elevated(
            command=command,
            cwd=command_cwd,
            timeout=effective_timeout,
        )
        # Normalize the elevated executor's dict to the env.execute
        # shape the unified pipeline expects below.
        elevated_output = elevated_result.get("output", "")
        result = {
            "output": elevated_output,
            "returncode": elevated_result.get("exit_code", -1),
        }
        if elevated_result.get("error"):
            result["error"] = elevated_result["error"]
        # Spill/truncation metadata: the elevated executor hands over
        # a STAGED RAW file (raw_output_path — a throwaway temp file,
        # never a durable path) when the output overflowed the
        # model-facing cap.  Forward it so the unified pipeline below
        # sanitizes it into a NEW durable spill path and reports
        # output_total_chars / full_output_path with the same
        # semantics as normal foreground output.  A plain
        # (non-overflowing) result carries no spill metadata and
        # degrades exactly like a small env.execute result.
        if elevated_result.get("raw_output_path"):
            result["output_total_chars"] = elevated_result["output_total_chars"]
            result["raw_output_path"] = elevated_result["raw_output_path"]
    else:
        for retry_count in range(max_retries + 1):
            try:
                command_cwd = _resolve_command_cwd(
                    workdir=workdir, default_cwd=plan.cwd, session_key=session_key, env_type=env_type,
                )
                # bounded_capture: model-facing output keeps a head/tail window
                # while streaming so a verbose command can't OOM the gateway;
                # internal env.execute() consumers stay unbounded.
                result = env.execute(
                    command, timeout=effective_timeout, cwd=command_cwd, bounded_capture=True,
                    **_yield_kwargs(command, env_type=env_type, cwd=command_cwd, effective_task_id=eff,
                                    task_id=task_id, session_key=session_key),
                )
                break
            except Exception as e:
                if "timeout" in str(e).lower():
                    return _error_json(f"Command timed out after {effective_timeout} seconds", exit_code=124)
                # Retry on transient errors
                if retry_count < max_retries:
                    wait_time = 2 ** (retry_count + 1)
                    logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                   wait_time, retry_count + 1, max_retries, _safe_command_preview(command), type(e).__name__, e, eff, env_type)
                    time.sleep(wait_time)
                    continue
                logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                             max_retries, _safe_command_preview(command), type(e).__name__, e, eff, env_type)
                return _error_json(_redact_terminal_error_text(f"Command execution failed: {type(e).__name__}: {e}"))
    if result.get("yielded_session_id"):
        return json.dumps({
            "output": result.get("output", ""), "exit_code": None, "error": None,
            "status": "yielded_to_background", "session_id": result["yielded_session_id"],
            "pid": result.get("pid"), "notify_on_complete": True, "note": _YIELDED_NOTE,
        }, ensure_ascii=False)
    return finalize_foreground_result(
        command=command, result=result, env=env, env_type=env_type, effective_task_id=eff,
        task_id=task_id, session_id=session_id, session_key=session_key, workdir=workdir,
        command_cwd=command_cwd, approval_note=approval_note,
    )


def _pre_exec_block(
    command: str, *, env: Any, env_type: str, cwd: str,
    workdir: Optional[str], session_key: str,
) -> None:
    """Raise :class:`_Rejected` with the blocked-result JSON when the command must not run.

    Order matters: gateway lifecycle first (protects the running gateway),
    then the dangerous-workdir check, then the self-repo guard (local only).
    """
    blocked = gateway_lifecycle_block(
        command=command, env=env, env_type=env_type, cwd=cwd, workdir=workdir, session_key=session_key,
    )
    if blocked:
        raise _Rejected(blocked)
    if workdir:
        workdir_error = _validate_workdir(workdir)
        if workdir_error:
            logger.warning("Blocked dangerous workdir: %s (command: %s)",
                           workdir[:200], _safe_command_preview(command))
            raise _Rejected(_error_json(workdir_error, status="blocked"))
    if env_type == "local":
        blocked = self_repo_block(command=command, cwd=cwd, workdir=workdir, session_key=session_key)
        if blocked:
            raise _Rejected(blocked)


_PTY_DISABLED_REASON = (
    "PTY disabled for this command because it expects piped stdin/EOF "
    "(for example gh auth login --with-token). For local background "
    "processes, call process(action='close') after writing so it receives "
    "EOF."
)


def _degraded_result(e: EnvironmentConnectionError, task_id: Optional[str]) -> str:
    """Infrastructure failure (SSH host down, Docker daemon unreachable), distinct
    from a nonzero exit. ``terminal.degraded_mode``: warn (default) returns a
    structured degraded result with a retry hint; fail preserves the historical
    error+traceback result."""
    if _tenv("TERMINAL_DEGRADED_MODE", "warn").strip().lower() == "fail":
        return _fatal_error_json(e)
    logger.warning("terminal backend degraded: %s", e.reason)
    # Evict the possibly-broken backend so the next call re-creates it.
    with _quiet("degraded-env eviction failed"):
        _evict_environment_for_task(task_id)
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "status": "degraded",
        "reason": e.reason,
        "retry_hint": e.retry_hint,
        "error": f"Terminal backend degraded: {e.reason}",
    }, ensure_ascii=False)


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[List[str]] = None,
    elevated: bool = False,
    _host_local: bool = False,
) -> str:
    """Execute *command* in the configured terminal environment; returns a JSON string.

    ``force`` (internal, not in the model schema) skips the dangerous-command
    check after the user confirmed. ``workdir`` is per-command and never
    recorded as the session cwd. ``pty`` applies to the local backend only.
    ``notify_on_complete`` and ``watch_patterns`` are mutually exclusive
    background-only flags: on conflict watch_patterns is dropped. watch_patterns
    is hard rate-limited (1 notification / 15s / process) and auto-disabled
    after repeated strikes or a lifetime cap, promoting to notify_on_complete —
    use it only for rare one-shot signals on long-lived processes.
    ``elevated`` (Windows) runs the command with administrator privileges
    through a UAC elevation prompt; it requires the local backend and
    cannot be combined with ``background`` or ``pty``.
    ``_host_local`` forces the local backend for Hermes-owned control-plane
    children (kept in a separate env cache from the configured backend).
    """
    try:
        plan = _plan_execution(
            command, task_id=task_id, timeout=timeout, background=background,
            pty=pty, elevated=elevated, _host_local=_host_local,
        )
        env = _acquire_env(plan, task_id)
        env_type, cwd, effective_task_id = plan.env_type, plan.cwd, plan.effective_task_id

        # Session key for cwd records: the contextvar doesn't cross tool-worker
        # threads, so fall back to the raw task_id (the top-level agent's
        # session_key) as a stable anchor.
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")

        _pre_exec_block(command, env=env, env_type=env_type, cwd=cwd, workdir=workdir, session_key=session_key)
        # Pre-exec security checks (tirith + dangerous command detection);
        # force=True means the user already confirmed.
        verdict = _run_approval_guards(command, env_type, plan.config, force=force)

        # Elevated execution via UAC (Windows only).  This gate sits AFTER the
        # shared security checks (gateway lifecycle guard + _check_all_guards
        # approval/command guard) and AFTER the workdir validation in
        # _pre_exec_block, so blocked/pending-approval commands and malicious
        # workdirs never reach the elevated executor and no guard is bypassable.
        #
        # Fail-closed combinations:
        # - env_type != "local": elevated would execute on the host Windows
        #   machine; from Docker/SSH/Daytona/Vercel-sandbox backends that would
        #   either be meaningless (inside the sandbox) or worse (escaping the
        #   backend to touch the host). Always blocked.
        # - background: elevated runs synchronously with a UAC prompt.
        # - pty: elevated uses file-based output capture, not a PTY.
        if elevated:
            if env_type != "local":
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": (
                        "elevated=true requires the local Windows backend "
                        f"(current env_type={env_type!r}). Elevated execution "
                        "is only supported for env_type='local' and is blocked "
                        "for Docker/SSH/Daytona/Vercel-sandbox backends to "
                        "prevent UAC from touching the host from inside a "
                        "sandbox."
                    ),
                    "status": "blocked",
                }, ensure_ascii=False)
            if background:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": "elevated=true cannot be combined with background=true. Elevated commands run synchronously with UAC.",
                    "status": "error",
                }, ensure_ascii=False)
            if pty:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": "elevated=true cannot be combined with pty=true. Elevated commands use file-based output capture, not PTY.",
                    "status": "error",
                }, ensure_ascii=False)

        pty_disabled = pty and _command_requires_pipe_stdin(command)
        if plan.promoted_from_foreground_timeout is not None:
            # Promotion implies notify_on_complete; watch_patterns is a background-only flag the
            # caller could not have meant for a foreground call, and the two are exclusive anyway.
            background, notify_on_complete, watch_patterns = True, True, None
        if background:
            result = spawn_background_process(
                command=command, env=env, env_type=env_type, effective_task_id=effective_task_id,
                task_id=task_id, session_key=session_key, workdir=workdir, cwd=cwd,
                effective_pty=pty and not pty_disabled, notify_on_complete=notify_on_complete,
                watch_patterns=watch_patterns, approval_note=verdict.note,
                pty_disabled_reason=_PTY_DISABLED_REASON if pty_disabled else None,
            )
            if plan.promoted_from_foreground_timeout is not None:
                result = _with_promoted_note(result, plan.promoted_from_foreground_timeout)
            return result
        return _run_foreground(
            command, env, plan,
            task_id=task_id, session_id=session_id, session_key=session_key,
            workdir=workdir, approval_note=verdict.note, clear_interrupt=verdict.approved_run,
            elevated=elevated,
        )
    except _Rejected as r:
        return r.result_json
    except EnvironmentConnectionError as e:
        return _degraded_result(e, task_id)
    except Exception as e:
        return _fatal_error_json(e)


def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met."""
    try:
        config = _get_env_config()
        checker = _REQUIREMENT_CHECKERS.get(config["env_type"], _check_plugin_requirements)
        return checker(config)
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e, exc_info=True)
        return False


from tools.registry import registry

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            },
            "background": {
                "type": "boolean",
                "description": "Run in the background, returning a session_id. Pair with notify=true for anything with a defined end (tests, builds, deploys) — without it the process runs silently. Only servers/watchers/daemons that never exit should stay silent. Short commands: prefer foreground with a generous timeout.",
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": f"Max seconds to wait (default: 180, foreground max: {FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command finishes — set high for long tasks, you won't wait unnecessarily. A foreground timeout above {FOREGROUND_MAX_TIMEOUT}s runs the command as a tracked background process with notify_on_complete=true instead (the result says so; do not re-run it).",
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path). Defaults to the session working directory."
            },
            "pty": {
                "type": "boolean",
                "description": "With background=true: run in a pseudo-terminal for interactive CLI tools (Codex, Claude Code, Python REPL). Local backend only. Default: false.",
                "default": False
            },
            "notify": {
                "description": "With background=true: notify=true fires exactly one notification when the process exits (the right choice for nearly every bounded task — builds, tests, deploys). notify=['pattern', ...] instead notifies when a line matches a pattern — ONLY for one-shot readiness signals on processes that never exit (e.g. ['Application startup complete']); rate-limited and auto-disabled if it over-fires. Omit for silent daemons.",
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}}
                ]
            },
            "elevated": {
                "type": "boolean",
                "description": "Run the command with Windows administrator privileges via UAC. Only works on Windows. When true, triggers a UAC elevation prompt — the user must approve. Output is captured via temp files (no PTY support). Cannot be used with background=true or pty=true. Default: false.",
                "default": False
            }
            # Legacy aliases (unadvertised, still accepted): notify_on_complete
            # (bool) and watch_patterns (list). notify=true|[...] maps onto
            # them in the dispatch wrapper; explicit notify wins on conflict.
        },
        "required": ["command"]
    }
}


def _handle_terminal(args, **kw):
    # Models sometimes send execute_code's ``code`` here; name the stray
    # argument and the right tool instead of failing on command=None.
    if "command" not in args and "code" in args:
        return tool_error(
            "terminal received a 'code' parameter, but it requires a shell "
            "command in 'command'. Use execute_code(code=...) for Python; "
            "for shell, retry as terminal(command=...)."
        )
    # `notify` is the advertised interface (true → notify_on_complete,
    # [...] → watch_patterns); the legacy args stay accepted, explicit
    # `notify` wins. Background-only modifiers on a foreground call fail
    # with the corrected call instead of being silently ignored.
    notify = args.get("notify")
    notify_on_complete = args.get("notify_on_complete", False)
    watch_patterns = args.get("watch_patterns")
    if not args.get("background", False):
        if notify or watch_patterns or notify_on_complete:
            return tool_error(
                "notify only applies to background commands (foreground "
                "results return directly). Either drop notify, or run as "
                "terminal(command=..., background=true, notify=...)."
            )
        if args.get("pty", False):
            return tool_error(
                "pty requires background=true (a PTY session is interacted "
                "with via process(action='write'/'submit'), which needs a "
                "tracked background process). Retry as terminal(command=..., "
                "background=true, pty=true)."
            )
    if notify is not None:
        if isinstance(notify, bool):
            notify_on_complete = notify
            watch_patterns = None
        elif isinstance(notify, list):
            watch_patterns = notify
            notify_on_complete = False
        else:
            return tool_error(
                "notify must be true/false (notify on exit) or a list of "
                "strings (notify on output pattern match)."
            )
    return terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=notify_on_complete,
        watch_patterns=watch_patterns,
        elevated=args.get("elevated", False),
    )


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
import importlib.util  # noqa: F401,E402
import platform  # noqa: F401,E402
import re  # noqa: F401,E402
import shlex  # noqa: F401,E402
import shutil  # noqa: F401,E402
import stat  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import sys  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'cleanup_vm': ('tools.terminal_tool_lifecycle', 'cleanup_vm'),
    'env_var_enabled': ('utils', 'env_var_enabled'),
    'get_active_env': ('tools.terminal_tool_lifecycle', 'get_active_env'),
    'has_direct_modal_credentials': ('tools.tool_backend_helpers', 'has_direct_modal_credentials'),
    'is_interrupted': ('tools.interrupt', 'is_interrupted'),
    'is_managed_tool_gateway_ready': ('tools.managed_tool_gateway', 'is_managed_tool_gateway_ready'),
    'is_persistent_env': ('tools.terminal_tool_lifecycle', 'is_persistent_env'),
    'nous_tool_gateway_unavailable_message': ('tools.tool_backend_helpers', 'nous_tool_gateway_unavailable_message'),
    'resolve_modal_backend_state': ('tools.tool_backend_helpers', 'resolve_modal_backend_state'),
    'strip_inert_heredoc_bodies': ('tools.shell_heredoc', 'strip_inert_heredoc_bodies'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
