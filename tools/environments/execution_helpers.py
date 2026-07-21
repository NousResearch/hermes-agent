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


def _pipe_stdin(proc: subprocess.Popen, data: str) -> None:
    """Write *data* to proc.stdin on a daemon thread without newline conversion.

    On Windows, text-mode stdin translates ``\n`` to ``\r\n``. Writing encoded
    bytes through the underlying buffer preserves caller-provided content on
    every platform.
    """

    def _write():
        try:
            if proc.stdin is None:
                return
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
    """Spawn a subprocess with the shared stdout/stderr/stdin contract."""
    kwargs.setdefault("creationflags", windows_hide_flags())
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
    """ProcessHandle adapter for SDK backends without a real subprocess."""

    def __init__(
        self,
        exec_fn: Callable[[], tuple[str, int]],
        cancel_fn: Callable[[], None] | None = None,
    ):
        self._cancel_fn = cancel_fn
        self._done = threading.Event()
        self._returncode: int | None = None
        self._error: Exception | None = None

        read_fd, write_fd = os.pipe()
        self._stdout = os.fdopen(read_fd, "r", encoding="utf-8", errors="replace")
        self._write_fd = write_fd

        def _worker():
            try:
                output, exit_code = exec_fn()
                self._returncode = exit_code
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

        threading.Thread(target=_worker, daemon=True).start()

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
