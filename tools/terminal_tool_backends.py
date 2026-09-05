"""Execution-environment backends for the terminal tool: per-backend builders, config-to-kwargs
shapers, and requirement checkers, routed by dispatch/spec tables. Split out of ``tools/terminal_tool.py``;
the terminal tool imports the builders/checkers it uses from here."""

import functools
import importlib.util
import inspect
import logging
import shutil
import subprocess
from typing import Any, Dict, Optional

from tools.environments.docker import DockerEnvironment as _DockerEnvironment
from tools.environments.local import LocalEnvironment as _LocalEnvironment
from tools.environments.managed_modal import ManagedModalEnvironment as _ManagedModalEnvironment
from tools.environments.modal import ModalEnvironment as _ModalEnvironment
from tools.environments.singularity import SingularityEnvironment as _SingularityEnvironment
from tools.environments.ssh import SSHEnvironment as _SSHEnvironment
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
from tools.terminal_tool_config import _get_plugin_env_provider
from tools.tool_backend_helpers import (has_direct_modal_credentials, managed_nous_tools_enabled,
                                        nous_tool_gateway_unavailable_message, resolve_modal_backend_state)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")

_VERCEL_SANDBOX_DEFAULT_CWD = "/vercel/sandbox"
_SUPPORTED_VERCEL_RUNTIMES = ("node24", "node22", "python3.13")
_BUILTIN_BACKENDS = "local, docker, singularity, modal, daytona, vercel_sandbox, ssh"

# Config -> kwargs shapers, driven by (out_key, config_key, default) tables. The container table's
# (key, default) literal is intentionally greppable; tools/terminal_tool.py keeps its own for the AST test.
_SSH_KEYS = (("host", "ssh_host", ""), ("user", "ssh_user", ""), ("port", "ssh_port", 22),
             ("key", "ssh_key", ""), ("persistent", "ssh_persistent", False))
_RESOURCE_KEYS = (("cpu", "container_cpu", 1), ("memory", "container_memory", 5120),
                  ("disk", "container_disk", 51200), ("persistent_filesystem", "container_persistent", True))
_CONTAINER_KEYS = (
    ("container_cpu", 1), ("container_memory", 5120), ("container_disk", 51200),
    ("container_persistent", True), ("modal_mode", "auto"), ("vercel_runtime", ""),
    ("docker_volumes", []), ("docker_mount_cwd_to_workspace", False), ("docker_forward_env", []),
    ("docker_env", {}), ("docker_run_as_host_user", False), ("docker_extra_args", []),
    ("docker_shm_size", "1g"), ("docker_network", True), ("docker_persist_across_processes", True),
    ("docker_shared_container_key", ""), ("docker_orphan_reaper", True),
)
_DOCKER_KWARGS = (
    ("volumes", "docker_volumes", []), ("auto_mount_cwd", "docker_mount_cwd_to_workspace", False),
    ("forward_env", "docker_forward_env", []), ("env", "docker_env", {}),
    ("run_as_host_user", "docker_run_as_host_user", False), ("network", "docker_network", True),
    ("extra_args", "docker_extra_args", []), ("persist_across_processes", "docker_persist_across_processes", True),
    ("shared_container_key", "docker_shared_container_key", ""), ("shm_size", "docker_shm_size", "1g"),
)


def _ssh_config_from_config(config: Dict[str, Any]) -> dict:
    """``ssh_config`` for :func:`_create_environment` (shared with the lazy ``ensure_task_env``)."""
    return {out: config.get(key, default) for out, key, default in _SSH_KEYS}


def _container_config_from_config(config: Dict[str, Any]) -> dict:
    """``container_config`` for :func:`_create_environment` (shared with the lazy ``ensure_task_env``)."""
    return {k: config.get(k, d) for k, d in _CONTAINER_KEYS}


def _resources(cc: Dict[str, Any]) -> dict:
    """Common sandbox resource kwargs (cpu/memory in MB/disk in MB/persistence)."""
    return {out: cc.get(key, default) for out, key, default in _RESOURCE_KEYS}


def _is_supported_vercel_runtime(runtime: str) -> bool:
    return not runtime or runtime in _SUPPORTED_VERCEL_RUNTIMES


def _get_modal_backend_state(modal_mode: object | None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    return resolve_modal_backend_state(modal_mode, has_direct=has_direct_modal_credentials(),
                                       managed_ready=is_managed_tool_gateway_ready("modal"))


def _modal_unavailable_reason(modal_state: Dict[str, Any]) -> tuple[str, str]:
    """(log message, ValueError message) for a modal_state with no selected backend.
    Single decision shared by the requirements checker and the env builder."""
    gateway = nous_tool_gateway_unavailable_message("managed Modal execution")
    if modal_state["managed_mode_blocked"] or modal_state["mode"] == "managed":
        tail = (("Nous Tool Gateway access is not currently available and no direct Modal credentials/config "
                 f"were found. {gateway} Choose TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials.")
                if modal_state["managed_mode_blocked"] else f"the managed tool gateway is unavailable. {gateway}")
        return (f"Modal backend selected with TERMINAL_MODAL_MODE=managed, but {tail}",
                f"Modal backend is configured for managed mode, but {tail}")
    managed = managed_nous_tools_enabled()
    if modal_state["mode"] == "direct":
        return ("Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct Modal credentials/config "
                f"were found. Configure Modal or choose TERMINAL_MODAL_MODE={'managed/auto' if managed else 'auto'}.",
                "Modal backend is configured for direct mode, but no direct Modal credentials/config were found.")
    found = "or managed tool gateway was found" if managed else "was found"
    fix = ", set up the managed gateway, or" if managed else " or"
    return (f"Modal backend selected but no direct Modal credentials/config {found}. "
            f"Configure Modal{fix} choose a different TERMINAL_ENV.",
            f"Modal backend selected but no direct Modal credentials/config {found}.")


# --- Environment builders. Signature: (*, env_type, image, cwd, timeout, cc, task_id, ssh_config, host_cwd)
def _build_local_env(*, cwd, timeout, **_):
    return _LocalEnvironment(cwd=cwd, timeout=timeout)


def _build_docker_env(*, image, cwd, timeout, cc, task_id, host_cwd, **_):
    from tools.terminal_tool import (_docker_session_isolation_enabled, _has_isolation_overrides,
                                     _maybe_reap_docker_orphans)
    # One-shot reaper for labeled containers orphaned by prior Hermes processes that died before
    # atexit (SIGKILL / OOM / closed terminal); ``terminal.docker_orphan_reaper: false`` disables it.
    _maybe_reap_docker_orphans(cc)
    # A session-keyed container must not outlive its session, so cross-process reuse/persist is
    # disabled for it (cleanup_vm()/idle reaper stop+rm it). The shared "default" container and
    # RL/benchmark override sandboxes keep their existing lifecycle.
    session_scoped = (_docker_session_isolation_enabled() and task_id != "default"
                      and not _has_isolation_overrides(task_id))
    kwargs = {out: cc.get(key, default) for out, key, default in _DOCKER_KWARGS}
    if session_scoped:
        kwargs["persist_across_processes"] = False
    docker_env_obj = _DockerEnvironment(image=image, cwd=cwd, timeout=timeout, task_id=task_id, host_cwd=host_cwd,
                                        **_resources(cc), **kwargs)
    # Marker read by is_persistent_env(): a session-scoped container survives BETWEEN turns (skip
    # per-turn teardown) but is removed at session close / idle timeout. Test doubles may reject attrs.
    if session_scoped:
        try:
            docker_env_obj._session_scoped = True
        except AttributeError:
            pass
    return docker_env_obj


def _build_modal_env(*, image, cwd, timeout, cc, task_id, **_):
    res = _resources(cc)
    sandbox_kwargs = {k: res[k] for k in ("cpu", "memory") if res[k] > 0}
    if res["disk"] > 0:
        try:
            import modal
            if "ephemeral_disk" in inspect.signature(modal.Sandbox.create).parameters:
                sandbox_kwargs["ephemeral_disk"] = res["disk"]
        except Exception:
            pass
    modal_state = _get_modal_backend_state(cc.get("modal_mode"))
    selected = modal_state["selected_backend"]
    if selected not in ("managed", "direct"):
        raise ValueError(_modal_unavailable_reason(modal_state)[1])
    cls = _ManagedModalEnvironment if selected == "managed" else _ModalEnvironment
    return cls(image=image, cwd=cwd, timeout=timeout, modal_sandbox_kwargs=sandbox_kwargs,
               persistent_filesystem=res["persistent_filesystem"], task_id=task_id)


# env_type -> (class getter, takes image, extra kwargs from (cc, resource kwargs)). SDK-backed modules
# (daytona/vercel) are imported lazily so they are only required when that backend is selected.
_SANDBOX_ROWS = {
    "singularity": (lambda: _SingularityEnvironment, True, lambda cc, kw: {}),
    "daytona": (lambda: importlib.import_module("tools.environments.daytona").DaytonaEnvironment, True,
                lambda cc, kw: {"cpu": int(kw["cpu"])}),
    "vercel_sandbox": (lambda: importlib.import_module("tools.environments.vercel_sandbox").VercelSandboxEnvironment,
                       False, lambda cc, kw: {"runtime": cc.get("vercel_runtime") or None}),
}


def _build_sandbox_env(env_type, *, image, cwd, timeout, cc, task_id, **_):
    cls, with_image, extra = _SANDBOX_ROWS[env_type]
    kwargs = dict(cwd=cwd, timeout=timeout, task_id=task_id, **_resources(cc),
                  **({"image": image} if with_image else {}))
    kwargs.update(extra(cc, kwargs))
    return cls()(**kwargs)


_build_singularity_env = functools.partial(_build_sandbox_env, "singularity")
_build_daytona_env = functools.partial(_build_sandbox_env, "daytona")
_build_vercel_env = functools.partial(_build_sandbox_env, "vercel_sandbox")


def _build_ssh_env(*, cwd, timeout, ssh_config, **_):
    if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
        raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
    return _SSHEnvironment(host=ssh_config["host"], user=ssh_config["user"], port=ssh_config.get("port", 22),
                           key_path=ssh_config.get("key", ""), cwd=cwd, timeout=timeout)


def _build_plugin_env(*, env_type, image, cwd, timeout, cc, task_id, **_):
    provider = _get_plugin_env_provider(env_type)
    if provider is not None:
        env_obj = provider.create_environment(cwd=cwd, timeout=timeout, task_id=task_id, image=image,
                                              container_config=cc)
        # Stamp the backend name so path-resolution and progress surfaces can identify plugin
        # backends without class-name sniffing. Test doubles may reject attributes.
        try:
            env_obj._hermes_backend_name = provider.name.strip().lower()
        except AttributeError:
            pass
        return env_obj
    try:
        from agent.terminal_env_registry import plugin_backend_names
        plugin_names = plugin_backend_names()
    except Exception:
        plugin_names = []
    known = ", ".join(f"'{n}'" for n in _BUILTIN_BACKENDS.split(", ") + list(plugin_names))
    raise ValueError(f"Unknown environment type: {env_type}. Use {known}")


# Built-in backend -> builder. Anything else is looked up in the plugin registry.
_ENV_BUILDERS = {"local": _build_local_env, "docker": _build_docker_env, "singularity": _build_singularity_env,
                 "modal": _build_modal_env, "daytona": _build_daytona_env, "vercel_sandbox": _build_vercel_env,
                 "ssh": _build_ssh_env}


def _create_environment(env_type: str, image: str, cwd: str, timeout: int,
                        ssh_config: dict = None, container_config: dict = None,
                        local_config: dict = None, task_id: str = "default",
                        host_cwd: Optional[str] = None):
    """Create an execution environment (instance with ``execute()``) for *env_type*. ``image`` is ignored
    for local/ssh/vercel; ``container_config`` carries the container_*/docker_* resource keys; ``host_cwd`` is
    the host dir bound into Docker when cwd mounting is enabled. Unknown types fall through to plugin backends."""
    builder = _ENV_BUILDERS.get(env_type, _build_plugin_env)
    return builder(env_type=env_type, image=image, cwd=cwd, timeout=timeout, cc=container_config or {},
                   task_id=task_id, ssh_config=ssh_config, host_cwd=host_cwd)


# --- Requirement checkers: one generic path driven by _BACKEND_SPECS; optional fields, checked in order:
#   pre(config) -> True (satisfied) / False (rejected, already logged) / None (continue);
#   binary=(finder, version_arg, missing_log_or_None) runs ``<binary> <arg>``, ok iff rc == 0;
#   module=(find_spec name, log message when absent);  post(config) -> bool.
def _check_vercel(config: Dict[str, Any]) -> bool:
    """Runtime -> disk -> SDK -> auth (OIDC token, else the full TOKEN/PROJECT_ID/TEAM_ID tuple)."""
    runtime = (config.get("vercel_runtime") or "").strip()
    disk = config.get("container_disk", 51200)
    if not _is_supported_vercel_runtime(runtime):
        logger.error("Vercel Sandbox runtime %r is not supported. Set TERMINAL_VERCEL_RUNTIME to one of: %s.",
                     runtime, ", ".join(_SUPPORTED_VERCEL_RUNTIMES))
        return False
    if disk not in {0, 51200}:
        logger.error("Vercel Sandbox does not support custom TERMINAL_CONTAINER_DISK=%s. "
                     "Use the default shared setting (51200 MB).", disk)
        return False
    if importlib.util.find_spec("vercel") is None:
        logger.error("vercel is required for the Vercel Sandbox terminal backend: pip install vercel")
        return False
    from agent.secret_scope import get_secret
    if get_secret("VERCEL_OIDC_TOKEN"):
        return True
    present = [bool(get_secret(k)) for k in ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID")]
    if all(present):
        return True
    head = ("selected with token auth, but VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID must all be "
            "set together." if any(present) else
            "selected but no supported auth configuration was found. Set VERCEL_TOKEN, VERCEL_PROJECT_ID, "
            "and VERCEL_TEAM_ID for normal use.")
    logger.error(f"Vercel Sandbox backend {head} VERCEL_OIDC_TOKEN is supported for one-off local development only.")
    return False


def _modal_pre(config: Dict[str, Any]) -> Optional[bool]:
    modal_state = _get_modal_backend_state(config.get("modal_mode"))
    if modal_state["selected_backend"] == "managed":
        return True
    if modal_state["selected_backend"] != "direct":
        logger.error(_modal_unavailable_reason(modal_state)[0])
        return False
    return None


def _ssh_pre(config: Dict[str, Any]) -> bool:
    if config.get("ssh_host") and config.get("ssh_user"):
        return True
    logger.error("SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER "
                 "are not both set. Configure both or switch TERMINAL_ENV to 'local'.")
    return False


def _daytona_post(config: Dict[str, Any]) -> bool:
    from daytona import Daytona  # noqa: F401 — SDK presence check (ImportError propagates)
    from agent.secret_scope import get_secret
    return get_secret("DAYTONA_API_KEY") is not None


_BACKEND_SPECS: Dict[str, Dict[str, Any]] = {
    "local": {},
    "docker": {"binary": (lambda: importlib.import_module("tools.environments.docker").find_docker(), "version",
                          "Docker executable not found in PATH or common install locations")},
    "singularity": {"binary": (lambda: shutil.which("apptainer") or shutil.which("singularity"), "--version", None)},
    "ssh": {"pre": _ssh_pre},
    "modal": {"pre": _modal_pre,
              "module": ("modal", "modal is required for direct modal terminal backend: pip install modal")},
    "vercel_sandbox": {"pre": _check_vercel},
    "daytona": {"post": _daytona_post},
}


def _check_requirements(env_type: str, config: Dict[str, Any]) -> bool:
    spec = _BACKEND_SPECS[env_type]
    verdict = spec["pre"](config) if "pre" in spec else None
    if verdict is not None:
        return verdict
    if "binary" in spec:
        finder, arg, missing_msg = spec["binary"]
        executable = finder()
        if not executable:
            if missing_msg:
                logger.error(missing_msg)
            return False
        probe = subprocess.run([executable, arg], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
        return probe.returncode == 0
    if "module" in spec and importlib.util.find_spec(spec["module"][0]) is None:
        logger.error(spec["module"][1])
        return False
    return spec["post"](config) if "post" in spec else True


def _check_plugin_requirements(config: Dict[str, Any]) -> bool:
    env_type = config["env_type"]
    provider = _get_plugin_env_provider(env_type)
    if provider is not None:
        return bool(provider.check_requirements(config))
    logger.error("Unknown TERMINAL_ENV '%s'. Use one of: %s, or a plugin-registered backend.",
                 env_type, _BUILTIN_BACKENDS)
    return False


# Built-in backend -> requirements checker; unknown backends go to the plugin registry.
_REQUIREMENT_CHECKERS = {name: functools.partial(_check_requirements, name) for name in _BACKEND_SPECS}
_check_vercel_sandbox_requirements = _REQUIREMENT_CHECKERS["vercel_sandbox"]  # used by code_execution_tool
