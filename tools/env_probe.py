"""Local-environment toolchain probe for the system prompt: when the terminal backend
is local, one deterministic line about Python tooling (python3/python versions, missing
pip, pip bound to another Python, PEP 668) so models don't discover it by hitting
walls. Cached per process; "" when clean. Remote backends are skipped (the sandbox has
its own probe in agent/prompt_builder). Toggle: ``agent.environment_probe`` in config.yaml."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
from typing import Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)

# Concurrency model: exactly ONE background worker runs the probe; ``_PROBE_DONE``
# signals completion. Callers block at most ``_PROBE_WAIT_TIMEOUT`` s then fail open
# with "" — a stuck probe (e.g. a Windows pipe wedged by an orphaned pip descendant)
# can degrade only the probe line, never system-prompt construction.
# Module-level cache. The probe result is deterministic for the lifetime of the process — Python install
# state doesn't change mid-session in any way that would matter for the system prompt. See #67964.
_CACHE_LOCK = threading.Lock()
_CACHED_LINE: Optional[str] = None  # None = not probed yet; "" = probed, nothing to say.
_PROBE_DONE = threading.Event()
_PROBE_THREAD: Optional[threading.Thread] = None
_PROBE_GEN = 0  # bumped on reset so a stale worker can't publish into the fresh generation
_PROBE_WAIT_TIMEOUT = 10.0  # healthy runtime ~0.5s
_WAIT_ALREADY_TIMED_OUT = False  # after one full wait, later callers only peek

# Keep in sync with agent/prompt_builder.py:_REMOTE_TERMINAL_BACKENDS.
# Duplicated rather than imported to avoid a circular import.
_REMOTE_BACKENDS = frozenset({
    "docker", "singularity", "modal", "daytona", "ssh", "managed_modal",
    "vercel_sandbox",
})


def _plugin_backend_is_remote(backend: str) -> bool:
    """Whether a plugin-registered terminal backend is remote (fail-soft)."""
    if not backend or backend in _REMOTE_BACKENDS or backend == "local":
        return False
    try:
        from agent.terminal_env_registry import provider_flag

        return bool(provider_flag(backend, "is_remote", False))
    except Exception:
        return False


def _run(cmd: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    """Run a short subprocess -> (returncode, stdout, stderr); failures (binary
    missing, timeout, OSError) return (-1, "", "<reason>"). Output goes through temp
    files, not pipes, so ``timeout`` bounds the *whole* call even on native Windows: a
    console-script launcher (``pip.exe``) can spawn a descendant that inherits the
    captured handles and outlives its parent; with OS pipes ``communicate()``'s reader
    threads block until that grandchild closes the write end (a warm probe could hang
    ~28 min holding ``_CACHE_LOCK``). Temp files make ``wait()`` cover only the child."""
    try:
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            try:
                result = subprocess.run(
                    cmd, stdout=out_f, stderr=err_f, timeout=timeout, check=False,
                    # CREATE_NO_WINDOW (0 on POSIX): pythonw hosts would flash a console
                    stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
            except subprocess.TimeoutExpired:
                return -1, "", "timeout"
            out_f.seek(0)
            err_f.seek(0)
            return (result.returncode, out_f.read().decode("utf-8", "replace").strip(),
                    err_f.read().decode("utf-8", "replace").strip())
    except FileNotFoundError:
        return -1, "", "not found"
    except OSError as exc:
        return -1, "", f"oserror: {exc}"


def _py_out(binary: str, *args: str) -> Optional[str]:
    """stdout of ``<binary> *args`` when the binary is on PATH and exits 0, else None."""
    if not shutil.which(binary):
        return None
    rc, out, _err = _run([binary, *args])
    return out if rc == 0 else None


def _python_version_of(binary: str) -> Optional[str]:
    """Return a short version string like ``3.12.4`` for ``binary``, or None."""
    code = "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    return _py_out(binary, "-c", code) or None


def _has_pip_module(binary: str) -> bool:
    """True if ``<binary> -m pip --version`` succeeds."""
    return _py_out(binary, "-m", "pip", "--version") is not None


def _detect_pep668(binary: str) -> bool:
    """True when ``<binary>`` is PEP-668 externally-managed (``EXTERNALLY-MANAGED``
    marker next to the stdlib, as Debian/Ubuntu ship)."""
    code = ("import os; print('yes' if os.path.exists(os.path.join("
            "os.path.dirname(os.__file__), 'EXTERNALLY-MANAGED')) else 'no')")
    return (_py_out(binary, "-c", code) or "").strip() == "yes"


def _pip_python_version() -> Optional[str]:
    """If ``pip`` is on PATH, the Python version it's bound to — the trailing
    ``(python X.Y)`` of ``pip --version`` (e.g. ``"3.12"``), else None."""
    out = _py_out("pip", "--version") or ""
    if "(python " in out and out.endswith(")"):
        return out.rsplit("(python ", 1)[1][:-1].strip()
    return None


def _resolve_terminal_backend() -> str:
    """Scope-aware terminal backend name (``local`` when unresolvable)."""
    try:
        from tools.terminal_scope import terminal_env

        return (terminal_env("TERMINAL_ENV") or "local").strip().lower()
    except Exception:  # never let policy resolution break prompt building
        logger.debug("terminal backend resolution failed", exc_info=True)
        return "local"


def _build_probe_line() -> str:
    """Build the one-liner; "" when nothing notable is detected — the goal is to
    save the model from an avoidable wall, not narrate a healthy environment."""
    py3_ver = _python_version_of("python3")
    py_ver = _python_version_of("python")  # for systems with a `python` alias
    py3_has_pip = _has_pip_module("python3") if py3_ver else False
    pip_bound_to = _pip_python_version()
    py3_pep668 = _detect_pep668("python3") if py3_ver else False
    # Bare which() is correct here (unlike Hermes's own uv call sites): this reports
    # the environment *the model will see* in the terminal tool, whose PATH includes
    # the Hermes-managed $HERMES_HOME/bin via local.py.
    has_uv = shutil.which("uv") is not None

    mismatch = bool(pip_bound_to and py3_ver and not py3_ver.startswith(pip_bound_to))
    if py3_ver is not None and py3_has_pip and not mismatch and (not py3_pep668 or has_uv):
        return ""
    # Compact factual summary; ONE line so it doesn't dominate the prompt.
    bits: list[str] = []
    if py3_ver:
        bits.append(f"python3={py3_ver}" + ("" if py3_has_pip else " (no pip module)"))
    else:
        bits.append("python3=missing")
    if py_ver and py_ver != py3_ver:
        bits.append(f"python={py_ver}")
    elif not py_ver and py3_ver:
        # Common on Debian/Ubuntu — stop the model typing `python`.
        bits.append("python=missing (use python3)")
    if pip_bound_to:
        if mismatch:
            bits.append(f"pip→python{pip_bound_to} (mismatch)")
        elif not py3_has_pip:
            bits.append(f"pip→python{pip_bound_to}")  # pip script works, `-m pip` doesn't
    elif not py3_has_pip:
        # (when `pip` is off PATH but `python3 -m pip` works, say nothing)
        bits.append("pip=missing")
    if py3_pep668:
        bits.append("PEP 668=yes (use venv or uv)")
    if has_uv:
        bits.append("uv=installed")
    return "Python toolchain: " + ", ".join(bits) + "."


def get_environment_probe_line(*, force_refresh: bool = False) -> str:
    """Return the cached probe line (building it on first call); "" when the
    environment is clean, so the prompt assembler drops the section. Waits at most
    ``_PROBE_WAIT_TIMEOUT`` on the single worker, then fails open with "".
    ``force_refresh`` is for tests.

    A wedged probe subprocess (#67964) therefore can never block system-prompt construction — at worst the
    toolchain line is absent from prompts built while the probe is stuck.
    """
    global _WAIT_ALREADY_TIMED_OUT
    if force_refresh:
        _reset_cache_for_tests()
    # Resolve the backend HERE, in the caller's context: under gateway multiplexing the
    # routed profile's backend lives in the per-turn terminal scope, which the worker
    # thread does not inherit. Remote backends answer "" without consulting the cache
    # — the cached line describes the HOST toolchain.
    # See #68559.
    backend = _resolve_terminal_backend()
    if backend in _REMOTE_BACKENDS or _plugin_backend_is_remote(backend):
        return ""
    if _PROBE_DONE.is_set():
        return _CACHED_LINE or ""
    _ensure_probe_started()
    wait_timeout = 0.05 if _WAIT_ALREADY_TIMED_OUT else _PROBE_WAIT_TIMEOUT
    if not _PROBE_DONE.wait(timeout=wait_timeout):
        # Probe stuck or pathologically slow: the line is a nice-to-have,
        # blocking prompt construction is an outage. Fail open.
        if not _WAIT_ALREADY_TIMED_OUT:
            _WAIT_ALREADY_TIMED_OUT = True
            logger.warning(
                "env_probe did not finish within %.0fs; building the system "
                "prompt without the Python toolchain line", _PROBE_WAIT_TIMEOUT)
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
        if _PROBE_DONE.is_set() or (_PROBE_THREAD is not None and _PROBE_THREAD.is_alive()):
            return
        _PROBE_THREAD = threading.Thread(
            target=_probe_worker, args=(_PROBE_GEN,), name="env-probe", daemon=True)
        _PROBE_THREAD.start()


def warm_environment_probe_async() -> None:
    """Start the probe in the background so the first system-prompt build doesn't
    pay the ~0.5s of subprocess calls on the time-to-first-token path. Idempotent;
    ``get_environment_probe_line`` waits (bounded) on the same worker."""
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
