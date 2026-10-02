"""Helpers shared by the non-local terminal backends (docker, ssh, singularity, cloud SDKs)."""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import subprocess
import threading
import time
from typing import Callable, Iterable

from tools.environments.base_session_env import _SHELL_ENV_NAME_RE
from tools.environments.local_env_policy import _is_hermes_internal_secret, _is_provider_env_blocklisted

logger = logging.getLogger(__name__)


def load_hermes_env_vars() -> dict[str, str]:
    """``~/.hermes/.env`` values, or ``{}`` — a broken .env must not fail command execution."""
    try:
        from hermes_cli.config import load_env
        return load_env() or {}
    except Exception:
        return {}


def resolve_passthrough_env(explicit_forward: Iterable[str] = (),
                            hermes_env_loader: Callable[[], dict[str, str]] = load_hermes_env_vars,
                            ) -> tuple[dict[str, str], set[str]]:
    """Values to forward into a remote shell plus the scoped names that must be unset there.

    Implicit passthrough (skill ``required_environment_variables`` + ``terminal.env_passthrough``)
    is filtered through the Hermes provider-credential blocklist and the dynamic internal-secret
    check; ``explicit_forward`` entries (docker_forward_env) are an operator opt-in that bypasses
    both. Each value is the routed profile's secret when multiplex is active; a name the active
    scope lacks is returned in the unset set so a shared sandbox cannot leak another profile's
    value.
    """
    passthrough_keys: set[str] = set()
    resolve_passthrough_value = None
    multiplex_active = False
    is_global_env = lambda _name: False  # noqa: E731
    try:
        from tools.env_passthrough import get_all_passthrough, resolve_passthrough_value
        from agent.secret_scope import _is_global_env as is_global_env, is_multiplex_active
        multiplex_active = is_multiplex_active()
        passthrough_keys = set(get_all_passthrough())
    except Exception:
        pass
    implicit_forward = {k for k in passthrough_keys if not _is_hermes_internal_secret(k)}
    forward_keys = set(explicit_forward) | {
        k for k in implicit_forward if not _is_provider_env_blocklisted(k)}
    hermes_env = hermes_env_loader() if forward_keys else {}
    exec_env: dict[str, str] = {}
    unset_names: set[str] = set()
    for key in sorted(forward_keys):
        value = os.getenv(key) or hermes_env.get(key)
        if resolve_passthrough_value is not None:
            value = resolve_passthrough_value(key, value)
        if value is not None:
            exec_env[key] = value
        elif multiplex_active and not is_global_env(key) and _SHELL_ENV_NAME_RE.fullmatch(key):
            unset_names.add(key)
    return exec_env, unset_names


def prepend_unset(cmd_string: str, names: Iterable[str]) -> str:
    """Prefix ``cmd_string`` with ``unset`` of the profile-scoped names the remote shell must not see."""
    names = sorted(names)
    if not names:
        return cmd_string
    return f"unset {' '.join(shlex.quote(n) for n in names)} 2>/dev/null || true\n{cmd_string}"


def client_env_with(values: dict[str, str]) -> dict[str, str] | None:
    """Env for the docker/ssh CLIENT subprocess: forwarded values travel here (owner-readable
    /proc/*/environ) while the argv carries names only; ``None`` = inherit when nothing to add."""
    return {**os.environ, **values} if values else None


def run_capture(cmd: list[str], *, timeout: float, check: bool = False, env: dict | None = None,
                ) -> subprocess.CompletedProcess:
    """``subprocess.run`` with the backend-standard capture settings: text mode with utf-8/replace
    decoding and stdin closed (DEVNULL) so a CLI that unexpectedly prompts cannot hang the agent."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, check=check, stdin=subprocess.DEVNULL, env=env)


def bash_argv(cmd_string: str, login: bool = False) -> list[str]:
    """``bash [-l] -c <cmd>`` argv tail used by every spawn-per-call backend."""
    return ["bash", "-l", "-c", cmd_string] if login else ["bash", "-c", cmd_string]


# Killing the local client of a spawn-per-call backend (``ssh``) does not signal the command it
# started on the other side, so a timed-out or interrupted command kept running there. The remote
# shell records its PID; the kill runs remotely over a separate connection against that shell's
# process group (``ps`` resolves the group when the shell is not its leader; without ``ps`` the
# PID's own group, then the PID alone). TERM, up to 1s grace, then KILL (the local backend's shape).
# The script runs in a subshell so the record is removed even when it sets its own EXIT trap or
# ``exec``s; a killed group leaves the removal to the kill.
# A kill can land before the shell has recorded its PID (the session is still starting). The kill
# first leaves a ``.stop`` marker, then reads the PID; the shell first records its PID, then checks
# the marker. Whichever runs second sees the other's write, so either the kill finds the PID or the
# shell exits before running the command.
def record_exec_group(cmd_string: str, pidfile: str) -> str:
    """``cmd_string`` wrapped to record its shell's PID in ``pidfile`` for :func:`exec_group_kill_script`."""
    q = shlex.quote(pidfile)
    return (f"{{ echo $$ > {q}; }} 2>/dev/null\n"
            f"if [ -e {q}.stop ]; then rm -f {q} {q}.stop; exit 130; fi\n(\n{cmd_string}\n)\n"
            f"__hermes_exec_rc=$?\nrm -f {q}\nexit $__hermes_exec_rc")


def exec_group_kill_script(pidfile: str, *, force: bool = False) -> str:
    """Bash script that kills the process group :func:`record_exec_group` recorded in ``pidfile``:
    TERM, up to 1s grace, then KILL; ``force`` (a host about to hard-exit) sends KILL at once."""
    q = shlex.quote(pidfile)
    target = (f': 2>/dev/null > {q}.stop; p=$(cat {q} 2>/dev/null); [ -n "$p" ] || exit 0; rm -f {q} {q}.stop; '
              'g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d " "); t=-${g:-$p}; ')
    if force:
        return target + 'kill -KILL -- "$t" 2>/dev/null || kill -KILL "$p" 2>/dev/null; exit 0'
    return target + (
        'kill -TERM -- "$t" 2>/dev/null || { t=$p; kill -TERM "$t" 2>/dev/null; } || exit 0; '
        'for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 -- "$t" 2>/dev/null || exit 0; sleep 0.1; done; '
        'kill -KILL -- "$t" 2>/dev/null; exit 0')


# Bound on one remote kill round trip (the script's own TERM grace is 1s). Enforced by a reaper,
# never by the caller: see launch_remote_kill.
REMOTE_KILL_TIMEOUT_S = 5.0


def launch_remote_kill(argv: list[str], *, timeout: float = REMOTE_KILL_TIMEOUT_S,
                       label: str = "remote-kill") -> subprocess.Popen | None:
    """Start ``argv`` (a kill that runs on the far side: ``ssh … bash -c``, ``docker exec …``) and
    return at once. ``_kill_process`` runs inside the timeout path of ``_wait_for_process``, under
    the ``run_bounded_sync`` backstop that allows only ``_EXECUTE_WAIT_BOUND_GRACE_S`` past the
    command's timeout, and ``kill_live_foreground_processes`` calls it once per live command on exit
    paths, so a kill that waited for its round trip made timeouts late and serialized shutdown.
    A daemon thread reaps the client and kills it after ``timeout`` so a dead link cannot leave it
    behind; its own session lets it finish if this host exits first."""
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("%s could not start: %s", label, e)
        return None

    def _reap() -> None:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.debug("%s did not finish within %.1fs; abandoning it", label, timeout)
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                proc.wait(timeout=1)

    threading.Thread(target=_reap, name=f"hermes-{label}", daemon=True).start()
    return proc


def wait_remote_kills(procs: Iterable[subprocess.Popen], budget: float = REMOTE_KILL_TIMEOUT_S) -> None:
    """Wait, at most ``budget`` seconds in total, for kills from :func:`launch_remote_kill` to end.
    For a teardown that is about to close the transport those kills travel over (ssh ``-O exit``)."""
    deadline = time.monotonic() + budget
    for proc in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))


def ensure_lazy_dep(extra: str) -> None:
    """Lazy-install an optional SDK's pm extra (idempotent). Install failures
    surface as ``ImportError``."""
    import pm

    try:
        pm.ensure_import(extra)
    except Exception as e:
        raise ImportError(str(e))
