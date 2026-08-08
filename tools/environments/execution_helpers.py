"""Dependency-light execution helpers shared by environment backends.

Backends import these helpers directly instead of through ``base`` so a
long-lived process with an older cached ``tools.environments.base`` module can
still lazy-import newly updated backend modules.  ``base`` re-exports the same
objects for backward compatibility.
"""

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable, cast

from hermes_cli._subprocess_compat import windows_hide_flags


class EnvironmentConnectionError(RuntimeError):
    """Infrastructure/connection-class failure of a terminal backend.

    Raised when the backend itself is unreachable (SSH host down, Docker
    daemon not running, remote file sync failing on a dead link) — never
    for a command that merely exited nonzero.  Subclassing RuntimeError
    keeps every existing ``except RuntimeError`` catcher working.

    ``terminal_tool`` turns this into a structured ``status: "degraded"``
    tool result (config gate ``terminal.degraded_mode: warn|fail``) so the
    model gets an actionable reason + retry hint instead of a traceback.
    The failed backend is never cached, so a later call retries from
    scratch and simply works once the backend is reachable again.
    """

    def __init__(self, reason: str, *, retry_hint: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.retry_hint = retry_hint or (
            "This is an infrastructure failure, not a command failure. "
            "Verify the backend is reachable (network, service running, "
            "credentials), then retry the same command — recovery is "
            "automatic once the backend is back."
        )


def _pipe_stdin(proc: subprocess.Popen, data: str) -> None:
    """Write *data* to proc.stdin on a daemon thread to avoid pipe-buffer deadlocks.

    On Windows, text-mode stdin (``text=True`` / ``encoding="utf-8"``)
    translates ``\\n`` → ``\\r\\n`` as the data flows through the pipe —
    which corrupts every write_file / patch call because the bytes that
    land on disk include injected carriage returns.  The file IS created,
    but every subsequent byte-count / content compare against the
    caller's ``\\n``-only string fails.

    Workaround: write through ``proc.stdin.buffer`` (the underlying byte
    buffer), encoding to UTF-8 ourselves.  That bypasses Python's
    newline translation entirely on every platform.  No behaviour change
    on POSIX — the byte sequence is identical to what text-mode would
    produce there.
    """

    def _write():
        try:
            if proc.stdin is None:
                return
            # proc.stdin is a TextIOWrapper when text=True was set on the
            # Popen.  Its ``.buffer`` attribute is the raw BufferedWriter
            # that bypasses newline translation.  When Popen was created
            # in byte mode, proc.stdin is already a BufferedWriter with
            # no ``.buffer`` attribute — fall back to .write() directly.
            raw = data.encode("utf-8") if isinstance(data, str) else data
            target = getattr(proc.stdin, "buffer", proc.stdin)
            target.write(raw)
            target.close()
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
    kwargs.setdefault("creationflags", windows_hide_flags())
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
        **kwargs,
    )
    if stdin_data is not None:
        _pipe_stdin(proc, stdin_data)
    return proc


def _load_json_store(path: Path) -> dict:
    """Load a JSON file as a dict, returning ``{}`` on any error."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
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
        return cast(int, self._returncode)
