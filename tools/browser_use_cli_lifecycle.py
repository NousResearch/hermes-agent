"""Process-local harness leases: calls reuse a daemon until its last owner releases it.

This does not reap daemons after SIGKILL/OOM: those require an independent reaper.
"""
import contextlib
import logging
import os
import subprocess
import shutil
import threading
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
_lock = threading.RLock()


@dataclass
class _Daemon:
    cmd: list
    env: dict
    owners: dict = field(default_factory=dict)
    calls: int = 0


_daemons: dict = {}


_runtimes: dict = {}


def prepare_runtime(env, profile_home):
    """Give this process/profile an exclusive IPC namespace, including the default name.

    A BU_NAME in the harness default directory is shared with unrelated Hermes processes.
    Process-local leases cannot authorize stopping it. Keep configuration/workspace paths,
    but put IPC under a private short directory so --reload only reaches our own daemons.
    """
    from tools.browser_tool import _socket_safe_tmpdir
    # Unnamed managed tasks can resolve different browsers. A default daemon latches
    # its first CDP endpoint, so those browsers also need distinct IPC namespaces.
    endpoint = "named" if env.get("BU_NAME") else (env.get("BU_CDP_WS") or env.get("BU_CDP_URL") or "default")
    scope = (os.getpid(), str(profile_home), endpoint)
    with _lock:
        if scope not in _runtimes:
            _runtimes[scope] = tempfile.mkdtemp(prefix="hbu-", dir=_socket_safe_tmpdir())
        env["BH_RUNTIME_DIR"] = _runtimes[scope]
        env["BH_RUNTIME_DIR_SHARED"] = "1"


def _endpoint_key(env):
    """Mirror v0.1.13 IPC path precedence without importing its env-bound globals."""
    runtime = env.get("BH_RUNTIME_DIR") or env.get("BH_TMP_DIR")
    if runtime:
        root = Path(runtime).expanduser().resolve()
    else:
        home = env.get("BH_HOME") or env.get("BROWSER_HARNESS_HOME")
        base = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        root = (Path(home).expanduser() if home else base / "browser-harness") / "runtime"
        root = root.resolve()
    name = env.get("BU_NAME") or "default"
    stem = "bu" if runtime and env.get("BH_RUNTIME_DIR_SHARED") != "1" else f"bu-{name}"
    return os.path.normcase(str(root / stem))


def _stop_released(run_cli):
    # Lock spans --reload so a new lease cannot reuse a generation being stopped.
    for key, daemon in list(_daemons.items()):
        if daemon.owners or daemon.calls:
            continue
        try:
            result = run_cli([*daemon.cmd, "--reload"], "", daemon.env, 60)
            if result.returncode:
                logger.debug("Harness stop failed for %s: exit %s", key, result.returncode)
                continue
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("Harness stop failed for %s: %s", key, exc)
            continue
        del _daemons[key]


@contextlib.contextmanager
def drive_daemon(cmd, env, owner, cache_key, run_cli):
    key = _endpoint_key(env)
    with _lock:
        daemon = _daemons.setdefault(key, _Daemon(list(cmd), dict(env)))
        daemon.cmd, daemon.env = list(cmd), dict(env)
        daemon.owners.setdefault(owner, set()).add(cache_key)
        daemon.calls += 1
    try:
        yield
    finally:
        with _lock:
            daemon.calls -= 1
            _stop_released(run_cli)


def release_daemons(run_cli, *, owner=None, cache_key=None):
    """Release one task, a browser cache entry (including idle reap), or all owners.

    In-flight CLI calls delay stop; completing a call alone never releases its task.
    Owners sharing an IPC endpoint protect it even across served profiles.
    """
    with _lock:
        for daemon in _daemons.values():
            if owner is not None:
                daemon.owners.pop(owner, None)
            elif cache_key is not None:
                daemon.owners = {task: keys - {cache_key} for task, keys in daemon.owners.items()
                                 if keys - {cache_key}}
            else:
                daemon.owners.clear()
        _stop_released(run_cli)


def has_active_calls(cache_key):
    with _lock:
        return any(daemon.calls and any(cache_key in keys for keys in daemon.owners.values())
                   for daemon in _daemons.values())


def cleanup_runtime_dirs():
    """Exit only: retain directories while calls can still prepare a future lease."""
    with _lock:
        for scope, runtime in list(_runtimes.items()):
            root = os.path.normcase(str(Path(runtime).resolve())) + os.sep
            if not any(key.startswith(root) for key in _daemons):
                shutil.rmtree(runtime, ignore_errors=True)
                del _runtimes[scope]
