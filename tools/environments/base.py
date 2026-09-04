"""Base class for all Hermes execution environment backends.

Unified spawn-per-call model: every command spawns a fresh ``bash -c`` process.
A session snapshot (env vars, functions, aliases) is captured once at init and
re-sourced before each command. CWD persists via in-band stdout markers (remote)
or a temp file (local). Cohesive pieces live in sibling modules (``base_output``,
``base_session_env``, ``base_wait``, ``path_utils``).
"""

import json
import logging
import os
import shlex
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable

from hermes_constants import get_hermes_home
from tools.interrupt import is_interrupted, is_thread_interrupted
from tools.environments.base_output import (
    ProcessHandle, _finalize_wait_result, _new_output_collector, _start_drain_thread,
)
from tools.environments.base_session_env import (
    _SHELL_ENV_NAME_RE, _SNAP_TMP_SUFFIX, _cwd_marker, _snapshot_bootstrap_script, _split_cwd_marker,
    _wrap_command_script,
)
from tools.environments.base_wait import _WaitTrace

logger = logging.getLogger(__name__)

# Opt-in debug tracing for the interrupt/activity/poll machinery
# (HERMES_DEBUG_INTERRUPT=1). Off by default to avoid flooding gateway logs.
_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))

# Extra seconds the ``run_bounded_sync`` backstop waits past the inner ``_wait_for_process``
# deadline: the inner loop returns partial output + 124; the outer bound only fires when that
# loop never returns. Keep small so a healthy timeout still comes from the inner path.
# The inner poll loop is what returns partial output + returncode 124; this outer bound only exists for when
# that loop itself never returns (family A of #94285: a blocked wait that silently disables asyncio timers).
_EXECUTE_WAIT_BOUND_GRACE_S = 2.0

if _DEBUG_INTERRUPT:
    # quiet_mode forces the `tools` logger to ERROR on CLI startup, which would
    # swallow every trace; force this logger back to INFO in the opt-in case.
    logger.setLevel(logging.INFO)

# Thread-local activity callback: the agent sets it before a tool call so
# long-running _wait_for_process loops can report liveness to the gateway.
_activity_callback_local = threading.local()


class EnvironmentConnectionError(RuntimeError):
    """Infrastructure/connection-class failure of a terminal backend (SSH host down, Docker
    daemon not running, remote sync on a dead link) — never a command that merely exited
    nonzero. Subclassing RuntimeError keeps every ``except RuntimeError`` catcher working.
    ``terminal_tool`` turns this into a structured ``status: "degraded"`` result; the failed
    backend is never cached, so a later call retries from scratch."""

    def __init__(self, reason: str, *, retry_hint: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.retry_hint = retry_hint or (
            "This is an infrastructure failure, not a command failure. "
            "Verify the backend is reachable (network, service running, "
            "credentials), then retry the same command — recovery is "
            "automatic once the backend is back.")


def set_activity_callback(cb: Callable[[str], None] | None) -> None:
    """Register a callback that _wait_for_process fires periodically."""
    _activity_callback_local.callback = cb


def get_activity_callback() -> Callable[[str], None] | None:
    """Thread-local activity callback; capture it before handing work to another thread.

    Public accessor for callers outside this module that need to capture the calling thread's callback
    before handing work to another thread (the callback is thread-local, so a freshly spawned thread cannot
    read it back) — e.g. the manual cron-run heartbeat (#76502).
    """
    return getattr(_activity_callback_local, "callback", None)


def touch_activity_if_due(state: dict, label: str) -> None:
    """Fire the activity callback at most once every ``state['interval']`` (default 10 s).
    *state* holds ``last_touch``/``start`` monotonic timestamps. Swallows all exceptions."""
    now = time.monotonic()
    if now - state["last_touch"] < state.get("interval", 10.0):
        return
    state["last_touch"] = now
    try:
        cb = get_activity_callback()
        if cb:
            cb(f"{label} ({int(now - state['start'])}s elapsed)")
    except Exception:
        pass


def get_sandbox_dir() -> Path:
    """Host-side root for all sandbox storage (Docker workspaces, Singularity
    overlays/SIF cache). ``TERMINAL_SANDBOX_DIR`` overrides ``{HERMES_HOME}/sandboxes``."""
    custom = os.getenv("TERMINAL_SANDBOX_DIR")
    p = Path(custom) if custom else get_hermes_home() / "sandboxes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_json_store(path: Path) -> dict:
    """Load a JSON file as a dict, returning ``{}`` on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json_store(path: Path, data: dict) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _file_mtime_key(host_path: str) -> tuple[float, int] | None:
    """Return ``(mtime, size)`` for cache comparison, or ``None`` if unreadable."""
    try:
        st = Path(host_path).stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


class BaseEnvironment(ABC):
    """Common interface and unified execution flow for all Hermes backends. Subclasses
    implement ``_run_bash()`` and ``cleanup()``; the base provides ``execute()`` with
    snapshot sourcing, CWD tracking, interrupt handling and timeout enforcement."""

    # Subclasses that embed stdin as a heredoc (Modal, Daytona) set this.
    _stdin_mode: str = "pipe"  # "pipe" or "heredoc"

    # True only when commands execute on the SAME host as the Hermes process
    # (LocalEnvironment); controller-host facts then describe the execution target.
    is_local: bool = False

    # Snapshot creation timeout (override for slow cold-starts).
    _snapshot_timeout: int = 30

    # Local and Docker override this because they resolve allowlisted values
    # through the active profile scope; other backends keep plain snapshots.
    _profile_scoped_passthrough: bool = False

    def get_temp_dir(self) -> str:
        """Backend temp directory for session artifacts (``/tmp`` in sandboxes;
        LocalEnvironment overrides for Termux where only ``TMPDIR`` is writable)."""
        return "/tmp"

    def __init__(self, cwd: str, timeout: int, env: dict = None):
        self.cwd = cwd
        self.timeout = timeout
        self.env = env or {}

        self._session_id = uuid.uuid4().hex[:12]
        temp_dir = self.get_temp_dir().rstrip("/") or "/"
        self._snapshot_path = f"{temp_dir}/hermes-snap-{self._session_id}.sh"
        self._cwd_file = f"{temp_dir}/hermes-cwd-{self._session_id}.txt"
        self._cwd_marker = _cwd_marker(self._session_id)
        self._snapshot_ready = False
        self._snapshot_passthrough_names: set[str] = set()
        # True when login bash is unusable (e.g. broken Git-for-Windows startup)
        # so execute() must fall back to non-login ``bash -c``, not ``bash -l``.
        self._prefer_nonlogin = False

    # --- Abstract methods ---
    def _run_bash(
        self, cmd_string: str, *, login: bool = False, timeout: int = 120, stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Spawn a bash process to run *cmd_string*; every backend overrides this."""
        raise NotImplementedError(f"{type(self).__name__} must implement _run_bash()")

    @abstractmethod
    def cleanup(self):
        """Release backend resources (container, instance, connection)."""
        ...

    # --- Session snapshot (init_session) ---
    def _additional_profile_scoped_passthrough_names(self) -> Iterable[str]:
        """Return backend-specific names that must not persist in snapshots."""
        return ()

    def _snapshot_excluded_passthrough_names(self) -> tuple[str, ...]:
        """Profile-scoped names that must not persist in the snapshot. Monotonic for the
        environment lifetime: an allowlist can be cleared after a value was captured, and
        retaining the exclusion keeps that old value from leaking to a later profile."""
        if not self._profile_scoped_passthrough:
            return ()
        try:
            from agent.secret_scope import is_multiplex_active
            if is_multiplex_active():
                from tools.env_passthrough import get_all_passthrough
                names = (*get_all_passthrough(), *self._additional_profile_scoped_passthrough_names())
                self._snapshot_passthrough_names.update(
                    name for name in names if isinstance(name, str) and _SHELL_ENV_NAME_RE.fullmatch(name))
        except Exception:
            logger.debug("Could not refresh profile-scoped snapshot exclusions", exc_info=True)
        return tuple(sorted(self._snapshot_passthrough_names))

    def _snapshot_script_kwargs(self, cwd: str) -> dict:
        """Quoting inputs shared by the bootstrap and per-command wrapper scripts.
        ``_quote_cwd_for_cd`` / ``_quote_shell_path`` (not bare shlex.quote) let the Windows
        subclass rewrite ``C:\\...`` to ``/c/...`` so ``cd`` resolves and MSYS doesn't choke."""
        return dict(
            quoted_cwd=self._quote_cwd_for_cd(cwd),
            quoted_snap=self._quote_shell_path(self._snapshot_path),
            snap_tmp_template=self._quote_shell_path(self._snapshot_path + _SNAP_TMP_SUFFIX),
            cwd_marker=self._cwd_marker)

    def init_session(self):
        """Capture the login shell environment into the snapshot file (once, after construction).
        On success ``_snapshot_ready`` is set so commands source the snapshot instead of running
        under ``bash -l``. On failure, fall back to ``bash -l`` per command — unless a non-login
        probe shows login bash itself is dead, in which case prefer ``bash -c``."""
        bootstrap = _snapshot_bootstrap_script(
            excluded_names=self._snapshot_excluded_passthrough_names(), **self._snapshot_script_kwargs(self.cwd))
        try:
            proc = self._run_bash(bootstrap, login=True, timeout=self._snapshot_timeout)
            result = self._wait_for_process(proc, timeout=self._snapshot_timeout)
            if int(result.get("returncode") or 0) != 0:
                raise RuntimeError(f"snapshot bootstrap failed with exit code {result.get('returncode')}")
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info("Session snapshot created (session=%s, cwd=%s)", self._session_id, self.cwd)
        except Exception as exc:
            self._snapshot_ready = False
            self._prefer_nonlogin, detail = self._probe_nonlogin_fallback(str(exc))
            if self._prefer_nonlogin:
                logger.warning(
                    "init_session failed (session=%s): %s — "
                    "login bash unusable; falling back to non-login bash -c",
                    self._session_id, exc)
            else:
                logger.warning(
                    "init_session failed (session=%s): %s — falling back to bash -l per command",
                    self._session_id, detail)

    def _probe_nonlogin_fallback(self, detail: str) -> tuple[bool, str]:
        """Run ``true`` under non-login bash; return ``(prefer_nonlogin, detail)``."""
        probe_timeout = min(15, self._snapshot_timeout)
        try:
            probe = self._run_bash("true", login=False, timeout=probe_timeout)
            probe_result = self._wait_for_process(probe, timeout=probe_timeout)
            prefer_nonlogin = int(probe_result.get("returncode") or 0) == 0
            if not prefer_nonlogin:
                detail = (probe_result.get("stdout") or detail).strip() or detail
            return prefer_nonlogin, detail
        except Exception as probe_exc:
            return False, f"{detail}; non-login probe: {probe_exc}"

    # --- Command wrapping ---
    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        """Quote a ``cd`` target while preserving ``~`` expansion (``~/...``
        goes through ``$HOME`` so suffixes with spaces stay one word)."""
        if cwd == "~":
            return cwd
        if cwd == "~/":
            return "$HOME"
        if cwd.startswith("~/"):
            return f"$HOME/{shlex.quote(cwd[2:])}"
        return shlex.quote(cwd)

    def _quote_shell_path(self, path: str) -> str:
        """Quote *path* for a bash script. LocalEnvironment overrides this to
        rewrite native/mixed Windows paths to ``/c/...``; remote backends are POSIX."""
        return shlex.quote(path)

    def _wrap_command(self, command: str, cwd: str) -> str:
        """Full bash script: source snapshot, cd, run, re-dump env, emit CWD markers."""
        return _wrap_command_script(
            command,
            passthrough_names=self._snapshot_excluded_passthrough_names(),
            snapshot_ready=self._snapshot_ready,
            **self._snapshot_script_kwargs(cwd))

    @staticmethod
    def _embed_stdin_heredoc(command: str, stdin_data: str) -> str:
        """Append stdin_data as a shell heredoc to the command string (SDK backends)."""
        delimiter = f"HERMES_STDIN_{uuid.uuid4().hex[:12]}"
        return f"{command} << '{delimiter}'\n{stdin_data}\n{delimiter}"

    # --- Process lifecycle ---
    def _wait_for_process(
        self, proc: ProcessHandle, timeout: int = 120, *,
        bounded_capture: bool = False, watch_interrupt_tid: int | None = None) -> dict:
        """Poll-based wait with interrupt checking and stdout draining (shared, not overridden).
        ``bounded_capture=True`` (foreground terminal-tool path only) retains at most
        ``tool_output.max_bytes`` in a head/tail window so a verbose subprocess cannot OOM the
        process; the default keeps full fidelity for internal consumers. Fires the activity
        callback every 10s so the gateway's inactivity timeout doesn't kill long commands.
        ``watch_interrupt_tid`` is the tool-worker thread that submitted this wait: ``execute()``
        may move the wait onto a ``run_bounded_sync`` worker while ``/stop`` still interrupts the
        original tid, so both bits are honored. ``KeyboardInterrupt``/``SystemExit`` mid-poll
        kills the process first — the local backend spawns into its own process group, so an
        unkilled child would be orphaned.

        The default (False) preserves full-fidelity capture for internal consumers — file-operation ``cat``
        reads feeding the patch engine, code-execution RPC reads, log reads — where truncation would corrupt
        data. See #64435.
        """
        output = _new_output_collector(proc, bounded_capture)
        drain_thread = _start_drain_thread(proc, output)
        _now = time.monotonic()
        deadline = _now + timeout
        _activity_state = {"last_touch": _now, "start": _now}
        trace = _WaitTrace(proc, timeout, enabled=_DEBUG_INTERRUPT, logger=logger)
        trace.enter()

        def _kill_and_join():
            self._kill_process(proc)
            drain_thread.join(timeout=2)

        try:
            # Adaptive poll: start at 5ms so fast commands return in ~6ms, back
            # off exponentially toward 200ms so long builds don't pay poll CPU.
            _poll_sleep = 0.005
            while proc.poll() is None:
                trace.iterations += 1
                if is_interrupted() or is_thread_interrupted(watch_interrupt_tid):
                    trace.interrupted()
                    _kill_and_join()
                    return self._finalize_wait_result(output, output.render(suffix="\n[Command interrupted]"), 130)
                if time.monotonic() > deadline:
                    trace.timed_out()
                    _kill_and_join()
                    rendered = output.render(suffix=f"\n[Command timed out after {timeout}s]")
                    if output.total_chars == 0:
                        rendered = rendered.lstrip()
                    return self._finalize_wait_result(output, rendered, 124)
                touch_activity_if_due(_activity_state, "terminal command running")
                trace.heartbeat()
                time.sleep(_poll_sleep)
                if _poll_sleep < 0.2:
                    _poll_sleep = min(_poll_sleep * 1.5, 0.2)
        except (KeyboardInterrupt, SystemExit):
            trace.exception_exit()
            try:
                _kill_and_join()
            except Exception:
                pass  # cleanup is best-effort
            raise

        # The drain thread exits promptly after bash does (~300ms idle check);
        # a long join here would itself indicate a bug in the drain loop.
        drain_thread.join(timeout=2)
        try:
            proc.stdout.close()
        except Exception:
            pass
        trace.natural_exit(proc.returncode)

        # Join the stdin writer before reading its error list: a child that exits without
        # reading stdin can otherwise race ahead of a recorded encode failure. The timeout
        # is a pure safety net (write raises BrokenPipeError once the pipe closes).
        stdin_thread = getattr(proc, "_hermes_stdin_thread", None)
        if stdin_thread is not None:
            stdin_thread.join(timeout=5)
        rendered = output.render()
        result = self._finalize_wait_result(output, rendered, proc.returncode)
        if stdin_errors := getattr(proc, "_hermes_stdin_errors", None):
            result["stdin_error"] = err = str(stdin_errors[0])
            result["output"] = rendered + f"\n[stdin write failed: {err}]"
        return result

    _finalize_wait_result = staticmethod(_finalize_wait_result)

    def _kill_process(self, proc: ProcessHandle):
        """Terminate a process. Subclasses may override for process-group kill."""
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # --- CWD extraction ---
    def _update_cwd(self, result: dict):
        """Extract CWD from command output. Override for local file-based read."""
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """Parse the ``__HERMES_CWD_{session}__`` marker from ``result["output"]``, update
        ``self.cwd`` and strip the marker line. ``result["cwd_observed"]``/``["cwd"]`` are set
        only when THIS command emitted a marker: a killed/timed-out command emits none and
        ``self.cwd`` keeps the previous value. The environment is shared across sessions, so
        concurrent callers must read ``result["cwd"]`` rather than ``self.cwd``."""
        split = _split_cwd_marker(result.get("output", ""), self._cwd_marker)
        if split is None:
            return
        cwd_path, cleaned = split
        if cwd_path:
            self.cwd = cwd_path
            result["cwd_observed"] = True
            result["cwd"] = cwd_path
        result["output"] = cleaned

    # --- Hooks ---
    def _before_execute(self) -> None:
        """Hook before each command. Remote backends (SSH, Modal, Daytona)
        trigger their FileSyncManager here; bind-mount backends and Local don't."""
        pass

    # --- Unified execute() ---
    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
        rewrite_compound_background: bool = True,
        bounded_capture: bool = False) -> dict:
        """Execute a command, return {"output": str, "returncode": int}. ``bounded_capture=True``
        caps retention at ``tool_output.max_bytes`` WHILE draining; only the foreground terminal
        tool may set it — internal full-fidelity consumers (file-op ``cat`` reads feeding the
        patch engine, RPC reads, log reads) MUST leave it False or data is corrupted. The wait is
        bounded by ``agent.deadline.run_bounded_sync`` so a wedged poll loop cannot hang past
        ``timeout`` and silently disable every asyncio timer in the process.

        ``bounded_capture=True`` caps stdout/stderr retention at ``tool_output.max_bytes`` WHILE the stream
        is drained (head/tail window) instead of holding the full output in memory (#64435).
        See #94285.
        """
        self._before_execute()

        exec_command, sudo_stdin = self._prepare_command(command)
        # Guard against the `A && B &` subshell-wait trap by default; callers
        # that already produce shell-safe wrappers (spawn_via_env) pass False.
        if rewrite_compound_background:
            from tools.terminal_tool_sudo import _rewrite_compound_background
            exec_command = _rewrite_compound_background(exec_command)
        effective_timeout = timeout or self.timeout
        effective_cwd = cwd or self.cwd

        # Merge sudo stdin with caller stdin.
        effective_stdin = sudo_stdin + (stdin_data or "") if sudo_stdin is not None else stdin_data
        if effective_stdin and self._stdin_mode == "heredoc":
            exec_command = self._embed_stdin_heredoc(exec_command, effective_stdin)
            effective_stdin = None

        wrapped = self._wrap_command(exec_command, effective_cwd)

        # Login shell if the snapshot failed (so the user's profile still
        # loads), unless login itself is broken — then non-login is the only path.
        login = not self._snapshot_ready and not self._prefer_nonlogin

        parent_tid = threading.current_thread().ident
        # The activity callback is thread-local and the wait runs on the
        # deadline worker, so copy it across or long commands look idle.
        parent_activity_cb = get_activity_callback()
        proc_holder: list = []

        def _spawn_and_wait() -> dict:
            if parent_activity_cb is not None:
                set_activity_callback(parent_activity_cb)
            spawned = self._run_bash(wrapped, login=login, timeout=effective_timeout, stdin_data=effective_stdin)
            proc_holder.append(spawned)
            return self._wait_for_process(
                spawned, timeout=effective_timeout, bounded_capture=bounded_capture, watch_interrupt_tid=parent_tid)

        def _on_timeout() -> None:
            if proc_holder:
                self._kill_spawned_tree(proc_holder[0])

        # Hard wall-clock backstop: ``_wait_for_process`` polls to ``effective_timeout`` on the
        # tool thread; if that is the event-loop thread, or the wait never returns (Windows
        # pipe/poll hang), every asyncio timer is silently disabled. ``run_bounded_sync`` drives
        # expiry from a daemon worker + ``Event.wait`` so a blocked loop cannot disable it; the
        # grace lets the inner loop return the partial-output 124 path.
        # See #94285.
        from agent.deadline import run_bounded_sync

        try:
            bound_s = float(effective_timeout)
        except (TypeError, ValueError):
            bound_s = 120.0  # a non-numeric timeout must not disable the backstop
        bound_s += _EXECUTE_WAIT_BOUND_GRACE_S

        try:
            bounded = run_bounded_sync(
                _spawn_and_wait, bound_s, label=f"terminal.wait:{type(self).__name__}", on_timeout=_on_timeout)
        except (KeyboardInterrupt, SystemExit):
            _on_timeout()
            raise

        result = (
            {"output": f"[Command timed out after {effective_timeout}s]", "returncode": 124}
            if bounded.timed_out else bounded.value)
        self._update_cwd(result)
        return result

    def _kill_spawned_tree(self, spawned) -> None:
        """Best-effort kill of a wedged spawned process and its tree (backstop path)."""
        try:
            self._kill_process(spawned)
        except Exception:
            logger.debug("terminal wait-bound kill_process failed", exc_info=True)
        pid = getattr(spawned, "pid", None)
        if not pid:
            return
        try:
            from agent.deadline import kill_process_tree
            kill_process_tree(int(pid))
        except Exception:
            logger.debug("terminal wait-bound kill_process_tree failed", exc_info=True)

    # --- Shared helpers ---
    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _prepare_command(self, command: str) -> tuple[str, str | None]:
        """Transform sudo commands if SUDO_PASSWORD is available."""
        from tools.terminal_tool_sudo import _transform_sudo_command
        return _transform_sudo_command(command)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import IO  # noqa: F401,E402
from typing import Protocol  # noqa: F401,E402
import codecs  # noqa: F401,E402
from collections import deque  # noqa: F401,E402
import re  # noqa: F401,E402
import select  # noqa: F401,E402
import subprocess  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'sanitize_task_id_for_path': ('tools.environments.path_utils', 'sanitize_task_id_for_path'),
    'windows_hide_flags': ('hermes_cli._subprocess_compat', 'windows_hide_flags'),
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
