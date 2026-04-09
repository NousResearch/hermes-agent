"""Base class for all Hermes execution environment backends.

Unified spawn-per-call model: every command spawns a fresh ``bash -c`` process.
A session snapshot (env vars, functions, aliases) is captured once at init and
re-sourced before each command. CWD persists via in-band stdout markers (remote)
or a temp file (local).
"""

import json
import logging
import os
import shlex
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Callable, Protocol

from hermes_constants import get_hermes_home
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)


def get_sandbox_dir() -> Path:
    """Return the host-side root for all sandbox storage (Docker workspaces,
    Singularity overlays/SIF cache, etc.).

    Configurable via TERMINAL_SANDBOX_DIR. Defaults to {HERMES_HOME}/sandboxes/.
    """
    custom = os.getenv("TERMINAL_SANDBOX_DIR")
    if custom:
        p = Path(custom)
    else:
        p = get_hermes_home() / "sandboxes"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Shared constants and utilities
# ---------------------------------------------------------------------------

_SYNC_INTERVAL_SECONDS = 5.0


def _pipe_stdin(proc: subprocess.Popen, data: str) -> None:
    """Write *data* to proc.stdin on a daemon thread to avoid pipe-buffer deadlocks."""

    def _write():
        try:
            proc.stdin.write(data)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    threading.Thread(target=_write, daemon=True).start()


def _popen_bash(
    cmd: list[str], stdin_data: str | None = None, **kwargs
) -> subprocess.Popen:
    """Spawn a subprocess with standard stdout/stderr/stdin setup.

    If *stdin_data* is provided, writes it asynchronously via :func:`_pipe_stdin`.
    Backends with special Popen needs (e.g. local's ``preexec_fn``) can bypass
    this and call :func:`_pipe_stdin` directly.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        text=True,
        **kwargs,
    )
    if stdin_data is not None:
        _pipe_stdin(proc, stdin_data)
    return proc


def _load_json_store(path: Path) -> dict:
    """Load a JSON file as a dict, returning ``{}`` on any error."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save_json_store(path: Path, data: dict) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _file_mtime_key(host_path: str) -> tuple[float, int] | None:
    """Return ``(mtime, size)`` for cache comparison, or ``None`` if unreadable."""
    try:
        st = Path(host_path).stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# ProcessHandle protocol
# ---------------------------------------------------------------------------


class ProcessHandle(Protocol):
    """Duck type that every backend's _run_bash() must return.

    subprocess.Popen satisfies this natively.  SDK backends (Modal, Daytona)
    return _ThreadedProcessHandle which adapts their blocking calls.
    """

    def poll(self) -> int | None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...

    @property
    def stdout(self) -> IO[str] | None: ...

    @property
    def returncode(self) -> int | None: ...


class _ThreadedProcessHandle:
    """Adapter for SDK backends (Modal, Daytona) that have no real subprocess.

    Wraps a blocking ``exec_fn() -> (output_str, exit_code)`` in a background
    thread and exposes a ProcessHandle-compatible interface.  An optional
    ``cancel_fn`` is invoked on ``kill()`` for backend-specific cancellation
    (e.g. Modal sandbox.terminate, Daytona sandbox.stop).
    """

    def __init__(
        self,
        exec_fn: Callable[[], tuple[str, int]],
        cancel_fn: Callable[[], None] | None = None,
    ):
        self._cancel_fn = cancel_fn
        self._done = threading.Event()
        self._returncode: int | None = None
        self._error: Exception | None = None

        # Pipe for stdout — drain thread in _wait_for_process reads the read end.
        read_fd, write_fd = os.pipe()
        self._stdout = os.fdopen(read_fd, "r", encoding="utf-8", errors="replace")
        self._write_fd = write_fd

        def _worker():
            try:
                output, exit_code = exec_fn()
                self._returncode = exit_code
                # Write output into the pipe so drain thread picks it up.
                try:
                    os.write(self._write_fd, output.encode("utf-8", errors="replace"))
                except OSError:
                    pass
            except Exception as exc:
                self._error = exc
                self._returncode = 1
            finally:
                try:
                    os.close(self._write_fd)
                except OSError:
                    pass
                self._done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    @property
    def stdout(self):
        return self._stdout

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        return self._returncode if self._done.is_set() else None

    def kill(self):
        if self._cancel_fn:
            try:
                self._cancel_fn()
            except Exception:
                pass

    def wait(self, timeout: float | None = None) -> int:
        self._done.wait(timeout=timeout)
        return self._returncode


# ---------------------------------------------------------------------------
# CWD marker for remote backends
# ---------------------------------------------------------------------------


def _cwd_marker(session_id: str) -> str:
    return f"__HERMES_CWD_{session_id}__"


# ---------------------------------------------------------------------------
# BaseEnvironment
# ---------------------------------------------------------------------------


class BaseEnvironment(ABC):
    """Common interface and unified execution flow for all Hermes backends.

    Subclasses implement ``_run_bash()`` and ``cleanup()``.  The base class
    provides ``execute()`` with session snapshot sourcing, CWD tracking,
    interrupt handling, and timeout enforcement.
    """

    # Subclasses that embed stdin as a heredoc (Modal, Daytona) set this.
    _stdin_mode: str = "pipe"  # "pipe" or "heredoc"

    # Snapshot creation timeout (override for slow cold-starts).
    _snapshot_timeout: int = 30

    def get_temp_dir(self) -> str:
        """Return the backend temp directory used for session artifacts.

        Most sandboxed backends use ``/tmp`` inside the target environment.
        LocalEnvironment overrides this on platforms like Termux where ``/tmp``
        may be missing and ``TMPDIR`` is the portable writable location.
        """
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
        self._last_sync_time: float | None = (
            None  # set to 0 by backends that need file sync
        )

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Spawn a bash process to run *cmd_string*.

        Returns a ProcessHandle (subprocess.Popen or _ThreadedProcessHandle).
        Must be overridden by every backend.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _run_bash()")

    @abstractmethod
    def cleanup(self):
        """Release backend resources (container, instance, connection)."""
        ...

    # ------------------------------------------------------------------
    # Session snapshot (init_session)
    # ------------------------------------------------------------------

    def init_session(self):
        """Capture login shell environment into a snapshot file.

        Called once after backend construction.  On success, sets
        ``_snapshot_ready = True`` so subsequent commands source the snapshot
        instead of running with ``bash -l``.
        """
        # Full capture: env vars, functions (filtered), aliases, shell options.
        bootstrap = (
            f"export -p > {self._snapshot_path}\n"
            f"declare -f | grep -vE '^_[^_]' >> {self._snapshot_path}\n"
            f"alias -p >> {self._snapshot_path}\n"
            f"echo 'shopt -s expand_aliases' >> {self._snapshot_path}\n"
            f"echo 'set +e' >> {self._snapshot_path}\n"
            f"echo 'set +u' >> {self._snapshot_path}\n"
            f"pwd -P > {self._cwd_file} 2>/dev/null || true\n"
            f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' \"$(pwd -P)\"\n"
        )
        try:
            proc = self._run_bash(bootstrap, login=True, timeout=self._snapshot_timeout)
            result = self._wait_for_process(proc, timeout=self._snapshot_timeout)
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info(
                "Session snapshot created (session=%s, cwd=%s)",
                self._session_id,
                self.cwd,
            )
        except Exception as exc:
            logger.warning(
                "init_session failed (session=%s): %s — "
                "falling back to bash -l per command",
                self._session_id,
                exc,
            )
            self._snapshot_ready = False

    # ------------------------------------------------------------------
    # Command wrapping
    # ------------------------------------------------------------------

    def _wrap_command(self, command: str, cwd: str) -> str:
        """Build the full bash script that sources snapshot, cd's, runs command,
        re-dumps env vars, and emits CWD markers."""
        escaped = command.replace("'", "'\\''")

        parts = []

        # Source snapshot (env vars from previous commands)
        if self._snapshot_ready:
            parts.append(f"source {self._snapshot_path} 2>/dev/null || true")

        # cd to working directory — let bash expand ~ natively
        quoted_cwd = (
            shlex.quote(cwd) if cwd != "~" and not cwd.startswith("~/") else cwd
        )
        parts.append(f"cd {quoted_cwd} || exit 126")

        # Run the actual command
        parts.append(f"eval '{escaped}'")
        parts.append("__hermes_ec=$?")

        # Re-dump env vars to snapshot (last-writer-wins for concurrent calls)
        if self._snapshot_ready:
            parts.append(f"export -p > {self._snapshot_path} 2>/dev/null || true")

        # Write CWD to file (local reads this) and stdout marker (remote parses this)
        parts.append(f"pwd -P > {self._cwd_file} 2>/dev/null || true")
        # Use a distinct line for the marker. The leading \n ensures
        # the marker starts on its own line even if the command doesn't
        # end with a newline (e.g. printf 'exact'). We'll strip this
        # injected newline in _extract_cwd_from_output.
        parts.append(
            f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' \"$(pwd -P)\""
        )
        parts.append("exit $__hermes_ec")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Stdin heredoc embedding (for SDK backends)
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_stdin_heredoc(command: str, stdin_data: str) -> str:
        """Append stdin_data as a shell heredoc to the command string."""
        delimiter = f"HERMES_STDIN_{uuid.uuid4().hex[:12]}"
        return f"{command} << '{delimiter}'\n{stdin_data}\n{delimiter}"

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def _wait_for_process(self, proc: ProcessHandle, timeout: int = 120) -> dict:
        """Poll-based wait with interrupt checking and stdout draining.

        Shared across all backends — not overridden.
        """
        output_chunks: list[str] = []

        def _drain():
            try:
                for line in proc.stdout:
                    output_chunks.append(line)
            except UnicodeDecodeError:
                output_chunks.clear()
                output_chunks.append(
                    "[binary output detected — raw bytes not displayable]"
                )
            except (ValueError, OSError):
                pass

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()
        deadline = time.monotonic() + timeout

        while proc.poll() is None:
            if is_interrupted():
                self._kill_process(proc)
                drain_thread.join(timeout=2)
                return {
                    "output": "".join(output_chunks) + "\n[Command interrupted]",
                    "returncode": 130,
                }
            if time.monotonic() > deadline:
                self._kill_process(proc)
                drain_thread.join(timeout=2)
                partial = "".join(output_chunks)
                timeout_msg = f"\n[Command timed out after {timeout}s]"
                return {
                    "output": partial + timeout_msg
                    if partial
                    else timeout_msg.lstrip(),
                    "returncode": 124,
                }
            time.sleep(0.2)

        drain_thread.join(timeout=5)

        try:
            proc.stdout.close()
        except Exception:
            pass

        return {"output": "".join(output_chunks), "returncode": proc.returncode}

    def _kill_process(self, proc: ProcessHandle):
        """Terminate a process. Subclasses may override for process-group kill."""
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # ------------------------------------------------------------------
    # CWD extraction
    # ------------------------------------------------------------------

    def _update_cwd(self, result: dict):
        """Extract CWD from command output. Override for local file-based read."""
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """Parse the __HERMES_CWD_{session}__ marker from stdout output.

        Updates self.cwd and strips the marker from result["output"].
        Used by remote backends (Docker, SSH, Modal, Daytona, Singularity).
        """
        output = result.get("output", "")
        marker = self._cwd_marker
        last = output.rfind(marker)
        if last == -1:
            return

        # Find the opening marker before this closing one
        search_start = max(0, last - 4096)  # CWD path won't be >4KB
        first = output.rfind(marker, search_start, last)
        if first == -1 or first == last:
            return

        cwd_path = output[first + len(marker) : last].strip()
        if cwd_path:
            self.cwd = cwd_path

        # Strip the marker line AND the \n we injected before it.
        # The wrapper emits: printf '\n__MARKER__%s__MARKER__\n'
        # So the output looks like: <cmd output>\n__MARKER__path__MARKER__\n
        # We want to remove everything from the injected \n onwards.
        line_start = output.rfind("\n", 0, first)
        if line_start == -1:
            line_start = first
        line_end = output.find("\n", last + len(marker))
        line_end = line_end + 1 if line_end != -1 else len(output)

        result["output"] = output[:line_start] + output[line_end:]

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _before_execute(self):
        """Rate-limited file sync before each command.

        Backends that need pre-command sync set ``self._last_sync_time = 0``
        in ``__init__`` and override :meth:`_sync_files`.  Backends needing
        extra pre-exec logic (e.g. Daytona sandbox restart check) override
        this method and call ``super()._before_execute()``.
        """
        if self._last_sync_time is not None:
            now = time.monotonic()
            if now - self._last_sync_time >= _SYNC_INTERVAL_SECONDS:
                self._sync_files()
                self._last_sync_time = now

    def _sync_files(self):
        """Push files to remote environment. Called rate-limited by _before_execute."""
        pass

    # ------------------------------------------------------------------
    # Unified execute()
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> dict:
        """Execute a command, return {"output": str, "returncode": int}."""
        self._before_execute()

        exec_command, sudo_stdin = self._prepare_command(command)
        effective_timeout = timeout or self.timeout
        effective_cwd = cwd or self.cwd

        # Merge sudo stdin with caller stdin
        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data

        # Embed stdin as heredoc for backends that need it
        if effective_stdin and self._stdin_mode == "heredoc":
            exec_command = self._embed_stdin_heredoc(exec_command, effective_stdin)
            effective_stdin = None

        wrapped = self._wrap_command(exec_command, effective_cwd)

        # Use login shell if snapshot failed (so user's profile still loads)
        login = not self._snapshot_ready

        proc = self._run_bash(
            wrapped, login=login, timeout=effective_timeout, stdin_data=effective_stdin
        )
        result = self._wait_for_process(proc, timeout=effective_timeout)
        self._update_cwd(result)

        return result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def stop(self):
        """Alias for cleanup (compat with older callers)."""
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _prepare_command(self, command: str) -> tuple[str, str | None]:
        """Transform sudo commands if SUDO_PASSWORD is available."""
        from tools.terminal_tool import _transform_sudo_command

        return _transform_sudo_command(command)

    def _timeout_result(self, timeout: int | None) -> dict:
        """Standard return dict when a command times out."""
        return {
            "output": f"Command timed out after {timeout or self.timeout}s",
            "returncode": 124,
        }
