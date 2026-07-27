"""Local-environment toolchain probe for the system prompt.

When the terminal backend is local (the agent's tools run on the same
machine as Hermes itself), we surface a single deterministic line about
Python tooling state so models don't have to discover it by hitting
walls.  Common failure modes this addresses:

* Hermes ships under one Python (e.g. 3.11 in a bundled venv) while the
  user's login shell has a different one (e.g. 3.12 system).  ``pip``
  resolved from PATH may not match ``python3 -m pip``.
* The bundled-venv Python has no pip module installed → ``python3 -m
  pip`` returns ``No module named pip``.
* The system Python is PEP-668 externally-managed → naive
  ``pip install`` fails with ``error: externally-managed-environment``.

The probe is cheap (a handful of subprocess calls, ~50ms total),
cached for the lifetime of the process, and emits **at most one
short line** when something non-default is detected.  When the
environment looks normal (python3+pip both present and matched, no
PEP 668), it emits nothing — no token cost.

Remote terminal backends (docker, modal, ssh, …) are skipped: the
host's Python state is irrelevant when tools run inside a sandbox.
The sandbox has its own existing probe (``_probe_remote_backend``)
in ``agent/prompt_builder.py``.

Toggle via ``agent.environment_probe`` in config.yaml (default True).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)

# Module-level cache.  The probe result is deterministic for the
# lifetime of the process — Python install state doesn't change
# mid-session in any way that would matter for the system prompt.
#
# Concurrency model (#67964): the probe runs in exactly ONE background
# worker thread; ``_PROBE_DONE`` signals completion.  Callers never
# execute the probe themselves and never wait unboundedly — they block
# at most ``_PROBE_WAIT_TIMEOUT`` seconds on the event and then fail
# open with "".  This guarantees a stuck probe (e.g. a Windows pipe
# wedged open by an orphaned pip descendant) can degrade at most the
# probe line itself, never system-prompt construction.
_CACHE_LOCK = threading.Lock()
_CACHED_LINE: Optional[str] = None  # None = not probed yet; "" = probed, nothing to say.
_PROBE_DONE = threading.Event()
_PROBE_THREAD: Optional[threading.Thread] = None
# Generation counter — bumped on every reset so a stale worker (started
# before a test reset) can't publish its result into the fresh generation.
_PROBE_GEN = 0

# Upper bound a prompt build will wait for the probe.  Generous vs the
# ~0.5s healthy runtime (6 subprocesses × 3s timeout ≈ 18s pathological
# worst case), but finite: prompt construction must always proceed.
_PROBE_WAIT_TIMEOUT = 10.0
# Once one caller has burned the full wait and given up, later callers
# stop paying it too — they just peek at the event.  If the stuck worker
# ever finishes, the published line resumes appearing in new prompts.
_WAIT_ALREADY_TIMED_OUT = False

# Remote backends — keep in sync with agent/prompt_builder.py:_REMOTE_TERMINAL_BACKENDS.
# Duplicated rather than imported to avoid a circular import (prompt_builder
# imports nothing from tools).
_REMOTE_BACKENDS = frozenset({
    "docker", "singularity", "modal", "daytona", "ssh", "managed_modal",
})


def _run(cmd: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    """Run a short subprocess.  Returns (returncode, stdout, stderr).

    Failures (binary missing, timeout, OSError) return (-1, "", "<reason>").

    Output is captured through temporary files rather than ``capture_output``
    pipes so ``timeout`` bounds the *whole* call — even on native Windows.  A
    console-script launcher (e.g. ``pip.exe``) can spawn a descendant that
    inherits the captured stdout/stderr handles and outlives its parent.  With
    OS pipes, the reader threads inside ``subprocess.communicate()`` then block
    until that descendant closes the write end — which the timeout does *not*
    cover, because killing the direct child leaves the grandchild holding the
    pipe.  A whole warm probe could hang for ~28 min this way while holding
    ``_CACHE_LOCK``, wedging every new session's system-prompt build.

    Temp files have no reader threads, so ``wait()`` only ever waits on the
    direct child; a lingering grandchild holding the handle can't block us, and
    the probe genuinely fails open on timeout.
    """
    try:
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=err_f,
                    timeout=timeout,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    # CREATE_NO_WINDOW (0 on POSIX): the probe runs in
                    # windowless processes (pythonw gateway / kanban workers)
                    # where a console child would otherwise flash a visible
                    # window per probe — ~5 flashes at every worker startup.
                    creationflags=windows_hide_flags(),
                )
            except subprocess.TimeoutExpired:
                return -1, "", "timeout"
            out_f.seek(0)
            err_f.seek(0)
            out = out_f.read().decode("utf-8", "replace").strip()
            err = err_f.read().decode("utf-8", "replace").strip()
            return result.returncode, out, err
    except FileNotFoundError:
        return -1, "", "not found"
    except OSError as exc:
        return -1, "", f"oserror: {exc}"


def _python_version_of(binary: str) -> Optional[str]:
    """Return a short version string like ``3.12.4`` for ``binary``, or None."""
    if not shutil.which(binary):
        return None
    rc, out, err = _run([binary, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"])
    if rc == 0 and out:
        return out
    return None


def _has_pip_module(binary: str) -> bool:
    """True if ``<binary> -m pip --version`` succeeds."""
    if not shutil.which(binary):
        return False
    rc, _out, _err = _run([binary, "-m", "pip", "--version"])
    return rc == 0


def _detect_pep668(binary: str) -> bool:
    """True when ``<binary>``'s install location is PEP-668 externally-managed.

    Looks for ``EXTERNALLY-MANAGED`` next to the stdlib (the marker file
    Debian/Ubuntu drop in to gate naive ``pip install``).
    """
    if not shutil.which(binary):
        return False
    code = (
        "import sys, os;"
        "stdlib = os.path.dirname(os.__file__);"
        "marker = os.path.join(stdlib, 'EXTERNALLY-MANAGED');"
        "print('yes' if os.path.exists(marker) else 'no')"
    )
    rc, out, _err = _run([binary, "-c", code])
    return rc == 0 and out.strip() == "yes"


def _pip_python_version() -> Optional[str]:
    """If ``pip`` is on PATH, return the Python version it's bound to.

    ``pip --version`` output looks like::

        pip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)

    Returns the parenthesised version (e.g. ``"3.12"``) or None.
    """
    if not shutil.which("pip"):
        return None
    rc, out, _err = _run(["pip", "--version"])
    if rc != 0 or not out:
        return None
    # Parse trailing "(python X.Y)".
    if "(python " in out and out.endswith(")"):
        try:
            tail = out.rsplit("(python ", 1)[1]
            return tail[:-1].strip()
        except (IndexError, AttributeError):
            return None
    return None


def _build_probe_line() -> str:
    """Build the one-liner.  Returns "" when nothing notable is detected.

    Emit only when SOMETHING is off — the goal is to save the model from
    hitting an avoidable wall, not to narrate a healthy environment.
    """
    # Bail out if a remote terminal backend is configured; the host's
    # Python state isn't where the agent's tools run.
    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    if backend in _REMOTE_BACKENDS:
        return ""

    py3_ver = _python_version_of("python3")
    py_ver = _python_version_of("python")  # for systems with a `python` alias
    py3_has_pip = _has_pip_module("python3") if py3_ver else False
    pip_bound_to = _pip_python_version()
    py3_pep668 = _detect_pep668("python3") if py3_ver else False
    has_uv = shutil.which("uv") is not None

    # If python3 exists, has pip, has uv (or no PEP 668), and there's no
    # version mismatch between `pip` and `python3` → environment is
    # clean enough to stay silent.  The model can discover details by
    # running commands if it cares.
    mismatch = bool(pip_bound_to and py3_ver and not py3_ver.startswith(pip_bound_to))
    silent_conditions = (
        py3_ver is not None
        and py3_has_pip
        and not mismatch
        and (not py3_pep668 or has_uv)
    )
    if silent_conditions:
        return ""

    # Build a compact factual summary.  Keep it ONE line so it doesn't
    # dominate the prompt; the model is good at parsing dense info.
    bits: list[str] = []
    if py3_ver:
        py3_bit = f"python3={py3_ver}"
        if not py3_has_pip:
            py3_bit += " (no pip module)"
        bits.append(py3_bit)
    else:
        bits.append("python3=missing")

    if py_ver and py_ver != py3_ver:
        bits.append(f"python={py_ver}")
    elif not py_ver and py3_ver:
        # Common on Debian/Ubuntu — call it out so the model doesn't
        # type `python` and hit "command not found".
        bits.append("python=missing (use python3)")

    if pip_bound_to:
        if mismatch:
            bits.append(f"pip→python{pip_bound_to} (mismatch)")
        elif not py3_has_pip:
            # pip exists but `python3 -m pip` doesn't — the script
            # works but the module path doesn't.
            bits.append(f"pip→python{pip_bound_to}")
    elif py3_has_pip:
        # `pip` not on PATH but `python3 -m pip` works.
        pass
    else:
        bits.append("pip=missing")

    if py3_pep668:
        bits.append("PEP 668=yes (use venv or uv)")

    if has_uv:
        bits.append("uv=installed")

    if not bits:
        return ""

    return "Python toolchain: " + ", ".join(bits) + "."


def get_environment_probe_line(*, force_refresh: bool = False) -> str:
    """Return the cached probe line (building it on first call).

    Returns "" when the environment is clean — the system prompt
    assembler should drop the section in that case rather than
    emit an empty heading.

    The probe itself always runs in a single background worker thread;
    this function waits on its completion event for at most
    ``_PROBE_WAIT_TIMEOUT`` seconds and then fails open with "".  A
    wedged probe subprocess (#67964) therefore can never block
    system-prompt construction — at worst the toolchain line is absent
    from prompts built while the probe is stuck.

    ``force_refresh`` is for tests; real callers should never need it.
    """
    global _CACHED_LINE, _PROBE_THREAD, _PROBE_GEN, _WAIT_ALREADY_TIMED_OUT
    if force_refresh:
        with _CACHE_LOCK:
            _CACHED_LINE = None
            _PROBE_DONE.clear()
            _PROBE_THREAD = None
            _PROBE_GEN += 1
            _WAIT_ALREADY_TIMED_OUT = False

    if _PROBE_DONE.is_set():
        return _CACHED_LINE or ""

    _ensure_probe_started()
    wait_timeout = 0.05 if _WAIT_ALREADY_TIMED_OUT else _PROBE_WAIT_TIMEOUT
    if not _PROBE_DONE.wait(timeout=wait_timeout):
        # Probe stuck or pathologically slow.  The line is a nice-to-have;
        # blocking prompt construction is an outage.  Fail open — if the
        # worker eventually finishes, sessions started later get the line.
        if not _WAIT_ALREADY_TIMED_OUT:
            _WAIT_ALREADY_TIMED_OUT = True
            logger.warning(
                "env_probe did not finish within %.0fs; building the system "
                "prompt without the Python toolchain line",
                _PROBE_WAIT_TIMEOUT,
            )
        return ""
    return _CACHED_LINE or ""


def _probe_worker(gen: int) -> None:
    """Body of the single probe thread — computes and publishes the line."""
    global _CACHED_LINE
    try:
        line = _build_probe_line()
    except Exception as exc:  # never let probe failure propagate
        logger.debug("env_probe failed: %s", exc)
        line = ""
    with _CACHE_LOCK:
        if gen != _PROBE_GEN:
            return  # superseded by a reset (tests) — discard stale result
        _CACHED_LINE = line
        _PROBE_DONE.set()


def _ensure_probe_started() -> None:
    """Start the probe worker if it isn't running and hasn't finished."""
    global _PROBE_THREAD
    with _CACHE_LOCK:
        if _PROBE_DONE.is_set():
            return
        if _PROBE_THREAD is not None and _PROBE_THREAD.is_alive():
            return
        _PROBE_THREAD = threading.Thread(
            target=_probe_worker,
            args=(_PROBE_GEN,),
            name="env-probe",
            daemon=True,
        )
        _PROBE_THREAD.start()


def warm_environment_probe_async() -> None:
    """Kick off the probe in a background thread so the first
    system-prompt build doesn't pay the ~0.5s of subprocess calls
    (python3/pip/PEP-668 version checks) on the time-to-first-token
    critical path.

    Idempotent and fail-safe.  The prompt-build call to
    ``get_environment_probe_line`` waits (bounded) on the same worker's
    completion event instead of recomputing.  Called from agent init
    (all platforms); safe to call from anywhere.
    """
    _ensure_probe_started()


def _reset_cache_for_tests() -> None:
    """Test helper — clear the cache between probe scenarios."""
    global _CACHED_LINE, _PROBE_THREAD, _PROBE_GEN, _WAIT_ALREADY_TIMED_OUT
    with _CACHE_LOCK:
        _CACHED_LINE = None
        _PROBE_DONE.clear()
        _PROBE_THREAD = None
        _PROBE_GEN += 1
        _WAIT_ALREADY_TIMED_OUT = False
