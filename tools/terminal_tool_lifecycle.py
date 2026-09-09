"""Sandbox lifecycle for the terminal tool: idle reaping, teardown, manual/atexit
cleanup, and the lazy ensure_task_env bring-up. The env cache dicts and locks
stay in tools.terminal_tool (tests patch them there) and are read through it
at call time.

Split out of ``tools/terminal_tool.py``; every public/patched name is re-imported there,
so ``tools.terminal_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

import glob
import logging
import inspect
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from tools.environments.singularity import _get_scratch_dir
from tools.terminal_tool_backends import (
    _container_config_from_config,
    _ssh_config_from_config,
)
from tools.terminal_tool_config import _quiet

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")


# Advisory disk-usage check; cached so the recursive scan doesn't run on
# every command (a result up to 5 minutes stale is harmless).
_disk_usage_cache: dict = {"timestamp": 0.0, "result": False}

_DISK_USAGE_CACHE_TTL = 300.0  # seconds


def _scratch_paths():
    return glob.glob(str(_get_scratch_dir() / "hermes-*"))


def _check_disk_usage_warning():
    """True when hermes scratch dirs exceed the warning threshold (cached, advisory)."""
    from tools.terminal_tool import DISK_USAGE_WARNING_THRESHOLD_GB
    if time.monotonic() - _disk_usage_cache["timestamp"] < _DISK_USAGE_CACHE_TTL:
        return _disk_usage_cache["result"]
    try:
        total_bytes = 0
        for path in _scratch_paths():
            for f in Path(path).rglob('*'):
                if f.is_file():
                    with _quiet("Could not stat file %s", f, exc=OSError):
                        total_bytes += f.stat().st_size
        total_gb = total_bytes / (1024 ** 3)
        exceeded = total_gb > DISK_USAGE_WARNING_THRESHOLD_GB
        if exceeded:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
        _disk_usage_cache["timestamp"] = time.monotonic()
        _disk_usage_cache["result"] = exceeded
        return exceeded
    except Exception:
        # Don't update cache on error so the next call retries.
        logger.debug("Disk usage warning check failed", exc_info=True)
        return False


def _create_configured_env(
    config: Dict[str, Any], env_type: str, *, image: str, cwd: str, timeout: int,
    task_id: str, host_cwd: Optional[str], local_config: Optional[dict] = None,
):
    """``_create_environment`` with the ssh/container kwargs shaped from *config*
    (shared by the terminal tool and the lazy :func:`ensure_task_env` bring-up)."""
    from tools.terminal_tool_backends import _create_environment
    from tools.terminal_tool_config import _is_container_backend
    return _create_environment(
        env_type=env_type, image=image, cwd=cwd, timeout=timeout,
        ssh_config=_ssh_config_from_config(config) if env_type == "ssh" else None,
        container_config=(
            _container_config_from_config(config) if _is_container_backend(env_type) else None
        ),
        local_config=local_config, task_id=task_id, host_cwd=host_cwd,
    )


def _cleanup_env(env: Any, *, force_remove: Optional[bool] = None) -> None:
    """Tear down one environment via cleanup()/stop()/terminate(), whichever it has.

    ``force_remove`` is forwarded to ``cleanup()`` only when given and the backend's
    signature accepts it (``DockerEnvironment``, issue #20561; other backends don't).
    Shared by ``cleanup_vm``, the idle reaper and the prompt-time backend probe so
    the signature check lives in one place.
    """
    if hasattr(env, 'cleanup'):
        if force_remove is not None and "force_remove" in inspect.signature(env.cleanup).parameters:
            env.cleanup(force_remove=force_remove)
        else:
            env.cleanup()
    elif hasattr(env, 'stop'):
        env.stop()
    elif hasattr(env, 'terminate'):
        env.terminate()


def _teardown_env(env: Any, task_id: str, *, force_remove: Optional[bool] = None, done_msg: str = "Cleaned up inactive environment for task: %s") -> None:
    """``_cleanup_env`` plus outcome logging. A 404/"not found" error means the
    sandbox is already gone — logged at info."""
    try:
        _cleanup_env(env, force_remove=force_remove)
        logger.info(done_msg, task_id)
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _clear_file_ops_cache(task_id: str) -> None:
    """Invalidate the file_ops cache entry so ShellFileOperations can't reference a dead sandbox."""
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(task_id)
    except ImportError:
        pass


def _unregister_env(task_id: str):
    """Pop *task_id* from the env cache, activity map and creation locks; return
    the env (or None). Callers run the (slow) teardown OUTSIDE the lock —
    Modal/Docker teardown can block 10-15s and would stall every concurrent
    terminal/file tool call."""
    from tools.terminal_tool import (
        _active_environments, _creation_locks, _creation_locks_lock, _env_lock,
        _last_activity,
    )
    with _env_lock:
        env = _active_environments.pop(task_id, None)
        _last_activity.pop(task_id, None)
    with _creation_locks_lock:
        _creation_locks.pop(task_id, None)
    return env


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    from tools.terminal_tool import (
        _active_environments, _creation_locks, _creation_locks_lock, _env_lock,
        _last_activity,
    )
    current_time = time.time()

    # Sandboxes with active background processes stay alive (refresh activity).
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            if process_registry.has_active_processes(task_id):
                _last_activity[task_id] = current_time
    except ImportError:
        pass

    # Phase 1: unregister stale entries atomically under the lock; phase 2:
    # stop them outside it (see _unregister_env for why).
    with _env_lock:
        stale = [t for t, last in list(_last_activity.items()) if current_time - last > lifetime_seconds]
        envs_to_stop = [(t, _active_environments.pop(t, None)) for t in stale]
        for t in stale:
            _last_activity.pop(t, None)
        with _creation_locks_lock:
            for t in stale:
                _creation_locks.pop(t, None)
    for task_id, env in envs_to_stop:
        if env is not None:
            _clear_file_ops_cache(task_id)
            _teardown_env(env, task_id)


def get_active_env(task_id: str):
    """Return the active BaseEnvironment for *task_id*, or None."""
    from tools.terminal_tool import _active_environments, _env_lock, _resolve_container_task_id
    lookup = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(lookup) or _active_environments.get(task_id)


def ensure_task_env(task_id: Optional[str] = None):
    """Lazily create and cache the sandbox env for *task_id* if none is active.

    Lets non-terminal callers (``tools.image_source`` reading container-only
    paths) bring the sandbox up on demand with the same machinery as the
    terminal tool. No-op on local. Returns the env, or ``None`` when local or
    when creation fails (best-effort; the caller's fail-closed path stays intact).

    :func:`terminal_tool` creates the environment on the first terminal command, but nothing else did — so
    under a non-local backend (ssh, docker, …) a session whose first action is ``vision_analyze`` on a
    container-only path hit "no active sandbox session" because the SSH/Docker handshake never ran (issue
    #62825). vision reads such paths inside the sandbox (see ``tools.image_source``), so it calls this to
    bring the env up on demand, reusing the same creation machinery as the terminal tool.
    """
    from tools.terminal_tool import (
        _active_environments, _creation_locks, _creation_locks_lock, _env_lock,
        _get_env_config, _last_activity, _resolve_container_task_id,
        _resolve_task_host_cwd, _select_image, _start_cleanup_thread, resolve_task_overrides,
    )
    config = _get_env_config()
    env_type = config["env_type"]
    if env_type == "local":
        return None

    effective_task_id = _resolve_container_task_id(task_id)

    existing = get_active_env(effective_task_id)
    if existing is not None:
        with _env_lock:
            _last_activity[effective_task_id] = time.time()
        return existing

    image = _select_image(env_type, resolve_task_overrides(task_id), config)

    _start_cleanup_thread()

    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(effective_task_id, threading.Lock())

    with task_lock:
        existing = get_active_env(effective_task_id)
        if existing is not None:
            return existing
        try:
            new_env = _create_configured_env(
                config, env_type, image=image, cwd=config["cwd"],
                timeout=config["timeout"], task_id=effective_task_id,
                host_cwd=_resolve_task_host_cwd(config, task_id),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort bring-up
            logger.warning(
                "Lazy %s environment init failed for task %s: %s",
                env_type, effective_task_id[:8], exc,
            )
            return None

        with _env_lock:
            _active_environments[effective_task_id] = new_env
            _last_activity[effective_task_id] = time.time()
        logger.info(
            "%s environment lazily initialized for task %s",
            env_type, effective_task_id[:8],
        )
        return new_env


def is_persistent_env(task_id: str) -> bool:
    """True if *task_id*'s active env persists across turns.

    The agent loop skips per-turn teardown for these (persistent docker,
    daytona, modal, …); non-persistent backends are torn down at end of turn
    to prevent leakage, and the idle reaper handles the rest. Session-scoped
    docker containers count as persistent HERE: their lifetime is the session
    (removed by ``AIAgent.close()`` → ``cleanup_vm`` and the idle reaper).
    """
    env = get_active_env(task_id)
    if env is None:
        return False
    return bool(getattr(env, "_session_scoped", False) or getattr(env, "_persistent", False))


def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    from tools.terminal_tool import _active_environments
    cleaned = 0
    for task_id in list(_active_environments.keys()):
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)

    # Also clean any orphaned directories
    for path in _scratch_paths():
        with _quiet("Failed to remove orphaned path %s", path, exc=OSError):
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)

    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(task_id: str, *, force_remove: bool = False):
    """Manually clean up a specific environment by task_id.

    *force_remove* is forwarded to backends that accept it (currently only
    ``DockerEnvironment``). Default False matches session-lifecycle semantics:
    callers (``AIAgent.close()`` on TUI/gateway session teardown, the per-turn
    cleanup of non-persistent envs) must honor the user's persist-mode
    preference — stopping the container here would break the "ONE long-lived
    container shared across sessions" contract. Pass ``force_remove=True``
    only for user-initiated teardown. The idle reaper calls ``env.cleanup()``
    directly, so persist-mode idle envs are likewise no-op'd; only the orphan
    reaper at next startup reclaims them.
    """
    env = _unregister_env(task_id)
    _clear_file_ops_cache(task_id)
    if env is None:
        return
    _teardown_env(
        env, task_id, force_remove=force_remove,
        done_msg="Manually cleaned up environment for task: %s",
    )


def _evict_environment_for_task(task_id: Optional[str]) -> None:
    """Drop any cached env for *task_id* (and its collapsed key) after an
    infrastructure failure, so later calls don't reuse a dead connection."""
    from tools.terminal_tool import (
        _active_environments, _env_lock, _last_activity, _resolve_container_task_id,
    )
    keys = {_resolve_container_task_id(task_id)}
    if task_id:
        keys.add(task_id)
    evicted = []
    with _env_lock:
        for key in keys:
            env = _active_environments.pop(key, None)
            _last_activity.pop(key, None)
            if env is not None:
                evicted.append(env)
    for env in evicted:
        with _quiet("cleanup of degraded environment failed"):
            env.cleanup()
