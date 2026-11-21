"""The user's zsh login PATH for the terminal snapshot (#129103).

The snapshot runs ``bash -l``, which never reads ``~/.zprofile``. On macOS zsh is
the default login shell and Homebrew, pyenv and conda put their PATH entries
there, so without this the snapshot's ``python3`` is ``/usr/bin/python3``.

The PATH is captured from ``$SHELL -lc``: login but not interactive, so
``~/.zshrc`` prompt plugins (Powerlevel10k's gitstatusd, #107109) never start.
The probe gets a fixed environment (the process's HOME, user and ZDOTDIR and a
system PATH), never a session's: one gateway serves many sessions, and a
session's client toolchain (``agent/session_client_env``) must not reach
another session through the cache. It runs once per shell, HOME and ZDOTDIR per
process, with a timeout, closed stdin and a process-group kill; any failure
leaves the snapshot as it was before this module existed and is logged once.

The captured entries are put in front at the start of the snapshot command:
after ``/etc/profile``'s ``path_helper`` (which moves the spawn PATH behind the
system dirs, so changing the spawn env would not survive) and before the
configured init files, so directories those files add (nvm from a listed
``~/.zshrc``) still come first, as they do in the user's own terminal.
"""

import contextlib
import logging
import os
import platform
import shlex
import signal
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# Login shells whose init files bash never reads and that accept the POSIX probe
# below. bash-family shells are already covered by ``bash -l``; fish rejects
# ``${PATH}`` (#107845) and keeps the old behaviour.
_LOGIN_PATH_PROBE_SHELLS = frozenset({"zsh"})

# The same markers as the Desktop probe (apps/desktop/electron/shell-path.ts), so
# profile output printed around the value is ignored.
_PATH_START = "__HERMES_LOGIN_PATH_START__"
_PATH_END = "__HERMES_LOGIN_PATH_END__"
_PROBE_COMMAND = "printf '%s' \"" + _PATH_START + "${PATH}" + _PATH_END + '"'
_PROBE_TIMEOUT_SECONDS = 5.0
_PIPE_DRAIN_SECONDS = 1.0

# The probe's starting PATH: only system dirs, so the captured value holds what
# the login files add and nothing inherited from this process or a session.
_PROBE_BASE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
# Process variables the login files may read; everything else is left out.
_PROBE_ENV_KEYS = ("HOME", "USER", "LOGNAME", "ZDOTDIR", "LANG", "LC_ALL", "TMPDIR")

# (shell, HOME, ZDOTDIR) -> captured PATH, or None when the probe failed. A slow
# or broken profile costs at most one timeout per process.
_probe_results: dict[tuple[str, str, str], str | None] = {}
_probe_lock = threading.Lock()


def _probe_shell() -> str | None:
    """``$SHELL`` when it names an executable shell this module probes, else None."""
    shell = os.environ.get("SHELL", "")
    if (shell and Path(shell).name in _LOGIN_PATH_PROBE_SHELLS
            and os.path.isfile(shell) and os.access(shell, os.X_OK)):
        return shell
    return None


def _probe_env(shell: str) -> dict[str, str]:
    """The fixed probe environment: no session or inherited PATH entries."""
    env = {k: os.environ[k] for k in _PROBE_ENV_KEYS if os.environ.get(k)}
    env.update(SHELL=shell, PATH=_PROBE_BASE_PATH)
    return env


def _extract_path(output: str) -> str | None:
    """The value between the last start marker and the end marker after it."""
    start = output.rfind(_PATH_START)
    if start < 0:
        return None
    start += len(_PATH_START)
    end = output.find(_PATH_END, start)
    if end < 0:
        return None
    value = output[start:end].strip()
    return value or None


def _run_probe(shell: str, env: dict[str, str]) -> str | None:
    """Run ``shell -lc <probe>`` and return the PATH it printed, or None."""
    try:
        proc = subprocess.Popen(
            [shell, "-lc", _PROBE_COMMAND], env=env, text=True,
            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            # detached-spawn: ok - short-lived probe, its group is killed below on timeout
            start_new_session=True, cwd=env.get("HOME") or None)
    except (OSError, ValueError) as exc:
        logger.debug("login-shell PATH probe could not start %s: %s", shell, exc)
        return None
    try:
        stdout, _ = proc.communicate(timeout=_PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # Kill the whole group: a daemon the profile started can hold stdout open.
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok - POSIX only (login_shell_path returns on Windows)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.communicate(timeout=_PIPE_DRAIN_SECONDS)
        if proc.stdout is not None:
            with contextlib.suppress(OSError):
                proc.stdout.close()
        logger.debug("login-shell PATH probe of %s timed out after %ss",
                     shell, _PROBE_TIMEOUT_SECONDS)
        return None
    value = _extract_path(stdout or "")
    if value is None:
        logger.debug("login-shell PATH probe of %s printed no PATH (exit %s)",
                     shell, proc.returncode)
    return value


def login_shell_path() -> str | None:
    """The PATH the user's zsh login files set up; None when ``$SHELL`` is not
    zsh, on Windows, or when the probe failed (logged once at warning level)."""
    if _IS_WINDOWS:
        return None
    shell = _probe_shell()
    if shell is None:
        return None
    env = _probe_env(shell)
    key = (shell, env.get("HOME", ""), env.get("ZDOTDIR", ""))
    with _probe_lock:
        if key not in _probe_results:
            value = _run_probe(shell, env)
            _probe_results[key] = value
            if value is None:
                logger.warning(
                    "Could not read the PATH from the zsh login shell %s; terminal "
                    "commands use the bash login PATH until Hermes restarts. Keep "
                    "~/.zprofile fast and quiet, or list PATH files in "
                    "terminal.shell_init_files.", shell)
        return _probe_results[key]


def prepend_login_shell_path(cmd_string: str, login_path: str | None) -> str:
    """Prepend a line that puts *login_path*'s dirs first on PATH, then the dirs
    the shell already had that *login_path* lacks, each once, in order."""
    if not login_path:
        return cmd_string
    entries = [d for d in dict.fromkeys(login_path.split(":")) if d]
    if not entries:
        return cmd_string
    # Split $PATH by slicing, not word splitting: no IFS change, no glob expansion.
    merge = (f"__hermes_np={shlex.quote(':'.join(entries))}; __hermes_rest=\"${{PATH}}:\"; "
             'while [ -n "$__hermes_rest" ]; do __hermes_p=${__hermes_rest%%:*}; '
             '__hermes_rest=${__hermes_rest#*:}; [ -n "$__hermes_p" ] || continue; '
             'case ":$__hermes_np:" in *":$__hermes_p:"*) ;; '
             '*) __hermes_np="$__hermes_np:$__hermes_p" ;; esac; done; '
             'PATH=$__hermes_np; export PATH; unset __hermes_np __hermes_rest __hermes_p')
    return merge + "\n" + cmd_string
