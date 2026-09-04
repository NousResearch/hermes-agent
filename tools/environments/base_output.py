"""Process-handle plumbing for ``tools.environments.base``.

Bounded output capture (head/tail window + disk spill), the ``ProcessHandle``
duck type, the SDK adapter ``_ThreadedProcessHandle``, stdin piping, and the
stdout drain thread used by ``BaseEnvironment._wait_for_process``.
"""

import codecs
import logging
import os
import selectors
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import IO, Callable, Protocol

from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)

# Sentinel capacity for full-fidelity capture: large enough that the collector
# never evicts, so bounded and unbounded modes share one code path.
_UNBOUNDED_CAPTURE_CHARS = 2**63 - 1

_SPILL_MAX_AGE_S = 7 * 86400


class _BoundedOutputCollector:
    """Retain a bounded 40/60 head-tail window of streamed text. When ``spill_path`` is set,
    the FULL stream is also teed to that file once eviction begins (up to ``_SPILL_CAP_CHARS``)
    so a truncated foreground result is recoverable without re-running."""

    # Hard ceiling on spill file size; protects disk from runaway output.
    _SPILL_CAP_CHARS = 5_000_000

    def __init__(self, max_chars: int, spill_path: "Path | None" = None):
        self.max_chars = max(1, int(max_chars))
        self._head_limit = int(self.max_chars * 0.4)
        self._tail_limit = self.max_chars - self._head_limit
        self._head: list[str] = []
        self._tail: deque[str] = deque()
        self._head_chars = 0
        self._tail_chars = 0
        self._total_chars = 0
        self._lock = threading.Lock()
        self._spill_path = spill_path
        self._spill_fh: IO[str] | None = None
        self._spill_chars = 0
        self._spill_capped = False

    def _maybe_spill(self, text: str) -> None:
        """Tee ``text`` to the spill file (opened lazily on first overflow)."""
        if self._spill_path is None or self._spill_capped:
            return
        try:
            if self._spill_fh is None:
                from tools.spill_safety import ensure_spill_dir, open_exclusive
                # Raw pre-redaction output: private perms + symlink-refusing
                # exclusive create (a planted link must fail, never redirect).
                ensure_spill_dir(self._spill_path.parent, private=True)
                self._spill_fh = open_exclusive(self._spill_path, private=True, errors="replace")
                # Backfill what's retained so the file holds the stream from byte 0.
                backlog = "".join(self._head) + "".join(self._tail)
                self._spill_fh.write(backlog)
                self._spill_chars = len(backlog)
            budget = self._SPILL_CAP_CHARS - self._spill_chars
            if budget <= 0 or len(text) > budget:
                self._spill_fh.write(text[:max(0, budget)])
                self._spill_fh.write("\n... [spill capped at 5,000,000 chars] ...\n")
                self._spill_capped = True
            else:
                self._spill_fh.write(text)
            self._spill_chars += len(text)
        except OSError:
            # Disk trouble must never break command execution.
            self._spill_capped = True

    def close_spill(self) -> "str | None":
        """Close the spill file and return its path if it was used."""
        with self._lock:
            if self._spill_fh is None:
                return None
            try:
                self._spill_fh.close()
            except OSError:
                pass
            self._spill_fh = None
            return str(self._spill_path)

    @property
    def buffered_chars(self) -> int:
        with self._lock:
            return self._head_chars + self._tail_chars

    @property
    def total_chars(self) -> int:
        with self._lock:
            return self._total_chars

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            text_len = len(text)
            # Spill tee activates at the first overflow, then mirrors every chunk.
            if self._spill_path is not None and (
                self._spill_fh is not None or self._total_chars + text_len > self.max_chars):
                self._maybe_spill(text)
            self._total_chars += text_len
            start = 0

            if self._head_chars < self._head_limit:
                take = min(self._head_limit - self._head_chars, text_len)
                if take:
                    self._head.append(text[:take])
                    self._head_chars += take
                    start = take

            remaining = text_len - start
            if remaining <= 0 or self._tail_limit <= 0:
                return
            if remaining >= self._tail_limit:
                self._tail.clear()
                self._tail.append(text[-self._tail_limit :])
                self._tail_chars = self._tail_limit
                return

            chunk = text[start:]
            self._tail.append(chunk)
            self._tail_chars += len(chunk)
            while (excess := self._tail_chars - self._tail_limit) > 0:
                first = self._tail[0]
                if len(first) <= excess:
                    self._tail.popleft()
                    self._tail_chars -= len(first)
                else:
                    self._tail[0] = first[excess:]
                    self._tail_chars -= excess

    def render(self, *, suffix: str = "") -> str:
        """Render within ``max_chars``, preserving a required status suffix."""
        with self._lock:
            if len(suffix) >= self.max_chars:
                return suffix[-self.max_chars :]

            head = "".join(self._head)
            tail = "".join(self._tail)
            available = self.max_chars - len(suffix)
            if self._total_chars <= available:
                return head + tail + suffix

            # The notice length depends on the omitted count, which depends on
            # the notice length; iterate to a fixed point (converges quickly).
            notice = ""
            for _ in range(4):
                omitted = max(0, self._total_chars - max(0, available - len(notice)))
                updated = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
                    f"out of {self._total_chars:,} total] ...\n\n")
                if updated == notice:
                    break
                notice = updated

            content_budget = max(0, available - len(notice))
            head_chars = int(content_budget * 0.4)
            tail_chars = content_budget - head_chars
            rendered_tail = tail[-tail_chars:] if tail_chars else ""
            return head[:head_chars] + notice[:available] + rendered_tail + suffix


def _new_output_collector(proc, bounded_capture: bool) -> _BoundedOutputCollector:
    """Build the collector for one ``_wait_for_process`` call. ``bounded_capture`` (foreground
    terminal path only) caps retention at ``tool_output.max_bytes`` and tees overflow to a
    spill file under ``$HERMES_HOME/cache/terminal-output`` (created only on actual overflow;
    spills older than 7 days are pruned opportunistically). Otherwise the collector is
    effectively unbounded so internal consumers keep full-fidelity output."""
    if not bounded_capture:
        return _BoundedOutputCollector(_UNBOUNDED_CAPTURE_CHARS)
    try:
        from tools.tool_output_limits import get_max_bytes
        capture_limit = get_max_bytes()
    except Exception:
        capture_limit = 50_000
    spill_path = None
    try:
        spill_dir = get_hermes_home() / "cache" / "terminal-output"
        spill_path = spill_dir / f"out-{int(time.time())}-{os.getpid()}-{id(proc) & 0xffff:x}.log"
        if spill_dir.is_dir():
            cutoff = time.time() - _SPILL_MAX_AGE_S
            for old in spill_dir.glob("out-*.log"):
                try:
                    if old.stat().st_mtime < cutoff:
                        old.unlink()
                except OSError:
                    pass
    except Exception:
        spill_path = None
    return _BoundedOutputCollector(capture_limit, spill_path=spill_path)


def _finalize_wait_result(collector: _BoundedOutputCollector, rendered: str, returncode: int | None) -> dict:
    """Assemble a wait result, attaching spill metadata when overflow occurred."""
    result = {"output": rendered, "returncode": returncode}
    spill = collector.close_spill()
    if spill:
        result["output_total_chars"] = collector.total_chars
        result["full_output_path"] = spill
    return result


# --- Stdin / spawn helpers ---
def _pipe_stdin(proc: subprocess.Popen, data: str) -> None:
    """Write *data* to proc.stdin on a daemon thread to avoid pipe-buffer deadlocks.
    Writes go through ``proc.stdin.buffer`` as UTF-8 bytes we encode ourselves: Windows
    text-mode stdin would translate ``\\n`` -> ``\\r\\n`` and corrupt every write_file/patch
    payload. Encoding uses ``surrogateescape`` (exact inverse of the read-side decode);
    surrogates outside U+DC80-U+DCFF raise, the error is recorded on
    ``proc._hermes_stdin_errors`` (surfaced by ``_wait_for_process`` as ``stdin_error``) and
    stdin is still closed in ``finally`` so the child sees EOF instead of hanging.
    """
    errors: list[BaseException] = []
    proc._hermes_stdin_errors = errors

    def _write():
        if proc.stdin is None:
            errors.append(RuntimeError("process stdin unavailable"))
            return
        # Resolve the target BEFORE encoding: a failed encode must still
        # reach the finally-close, or the child hangs on EOF forever.
        target = getattr(proc.stdin, "buffer", proc.stdin)
        try:
            raw = data.encode("utf-8", "surrogateescape") if isinstance(data, str) else data
            written = target.write(raw)
            if written != len(raw):
                # A short write from a buffered writer is a real failure.
                raise RuntimeError(f"short stdin write: {written} of {len(raw)} bytes")
        except (BrokenPipeError, OSError):
            pass  # child closed stdin early — normal
        except Exception as exc:
            # Only reachable with out-of-range surrogates (e.g. literal U+D800).
            errors.append(exc)
        finally:
            try:
                target.close()
            except Exception:
                pass

    thread = threading.Thread(target=_write, daemon=True)
    proc._hermes_stdin_thread = thread
    thread.start()


def _popen_bash(cmd: list[str], stdin_data: str | None = None, **kwargs) -> subprocess.Popen:
    """Spawn a subprocess with standard stdout/stderr/stdin setup; *stdin_data* is written
    asynchronously via :func:`_pipe_stdin`. Backends with special Popen needs (e.g. local's
    ``preexec_fn``) can bypass this and call :func:`_pipe_stdin` directly."""
    kwargs.setdefault("creationflags", windows_hide_flags())
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace",
        **kwargs)
    if stdin_data is not None:
        _pipe_stdin(proc, stdin_data)
    return proc


# --- ProcessHandle protocol ---
class ProcessHandle(Protocol):
    """Duck type every backend's _run_bash() must return. subprocess.Popen satisfies this
    natively; SDK backends (Modal, Daytona) return _ThreadedProcessHandle."""

    def poll(self) -> int | None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...

    @property
    def stdout(self) -> IO[str] | None: ...

    @property
    def returncode(self) -> int | None: ...


class _ThreadedProcessHandle:
    """Adapter for SDK backends (Modal, Daytona) that have no real subprocess: runs a blocking
    ``exec_fn() -> (output_str, exit_code)`` on a background thread behind a ProcessHandle
    interface. ``cancel_fn`` is invoked on ``kill()`` for backend-specific cancellation."""

    def __init__(self, exec_fn: Callable[[], tuple[str, int]], cancel_fn: Callable[[], None] | None = None):
        self._cancel_fn = cancel_fn
        self._done = threading.Event()
        self._returncode: int | None = None

        # Pipe for stdout — the drain thread in _wait_for_process reads the read end.
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
            except Exception:
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
        return self._returncode


# --- Stdout drain thread ---
def _drain_stdout(proc: ProcessHandle, output: _BoundedOutputCollector) -> None:
    """Drain ``proc.stdout`` into *output* until EOF or shortly after exit.
    ``for line in proc.stdout`` would block on ``readline()`` until EOF, and a backgrounded
    grandchild (``cmd &``, ``setsid cmd & disown``) inherits the pipe's write end — so the
    tool would hang for the grandchild's lifetime. Instead we ``select()`` with a short poll
    and stop ~300ms after bash exits even if the pipe has not EOF'd. Raw 4096-byte ``os.read``
    chunks can split a multibyte UTF-8 sequence, so an incremental decoder with
    ``errors="replace"`` buffers partial sequences across chunks. Streams without a real
    integer ``fileno()`` (mocks, in-memory adapters) are iterated to EOF instead — otherwise
    the thread would die silently and lose all output. ``select()`` does not work on pipe fds
    on Windows, so a blocking ``os.read`` loop is used there.
    """
    stream = proc.stdout
    if stream is None:
        return
    # Non-blocking drain via select(). The old pattern — ``for line in proc.stdout`` — blocks on
    # ``readline()`` until the pipe reaches EOF. When the user's command backgrounds a process (``cmd &``,
    # ``setsid cmd & disown``, etc.), that backgrounded grandchild inherits the write-end of our stdout pipe
    # via ``fork()``. Even after ``bash`` itself exits, the pipe stays open because the grandchild still
    # holds it — so the drain thread never returns and the tool hangs for the full lifetime of the
    # grandchild (issue #8340: users reported indefinite hangs when restarting uvicorn with ``setsid ... &
    # disown``). The fix: select() with a short poll interval, and stop draining shortly after ``bash``
    # exits even if the pipe hasn't EOF'd yet. Any output the grandchild writes after that point goes to an
    # orphaned pipe (harmless — the kernel reaps it when our end closes). Decoding: we ``os.read()`` raw
    # bytes in fixed-size chunks (4096) so a single multibyte UTF-8 character can split across reads. An
    # incremental decoder buffers partial sequences across chunks, and ``errors="replace"`` mirrors the
    # baseline ``TextIOWrapper`` (which was constructed with ``encoding="utf-8", errors="replace"`` on
    # ``Popen``) so binary or mis-encoded output is preserved with U+FFFD substitution rather than
    # clobbering the whole buffer.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        fd = stream.fileno()
    except Exception:  # mocks / in-memory adapters without a real descriptor
        fd = None
    try:
        if not isinstance(fd, int) or fd < 0:
            for piece in stream:
                if piece is not None:
                    output.append(decoder.decode(piece) if isinstance(piece, bytes) else str(piece))
        elif os.name == "nt":
            while chunk := os.read(fd, 4096):
                output.append(decoder.decode(chunk))
        else:
            _drain_fd_select(proc, fd, output, decoder)
    except Exception:
        pass  # closed fd / broken stream: keep what was captured
    finally:
        # With errors="replace" this emits U+FFFD for a final incomplete sequence.
        try:
            tail = decoder.decode(b"", final=True)
            if tail:
                output.append(tail)
        except Exception:
            pass


def _drain_fd_select(proc, fd: int, output: _BoundedOutputCollector, decoder) -> None:
    """POSIX drain: selectors-based poll, stopping ~300ms after bash exits with the pipe idle.

    select.select() only accepts fds < FD_SETSIZE (1024), so a pipe fd >= 1024
    raised ValueError — misread as EOF, silently discarding all output
    (#94928). selectors.DefaultSelector() has no fd limit and logs a genuine
    readiness error instead. A non-pollable fd (regular file -> EPERM from
    epoll, or an already-closed one) falls back to direct reads rather than
    dropping output.
    """
    idle_after_exit = 0
    _selector = selectors.DefaultSelector()
    try:
        try:
            _selector.register(fd, selectors.EVENT_READ)
        except (ValueError, OSError) as _exc:
            # register() fails for a non-pollable fd (e.g. a regular file,
            # which epoll rejects with EPERM) or an already-closed one. Drain
            # with direct reads instead of silently dropping the output. Switch
            # the fd to non-blocking and reuse the idle-after-exit bound so a
            # still-live pipe (whose register failure is pathological) cannot
            # hold the drain for the full timeout: regular files read to EOF
            # without ever blocking, and the bound mirrors the selector path.
            logger.warning(
                "drain: could not register fd %s with selector (%r); falling back to polling read",
                fd, _exc,
            )
            try:
                os.set_blocking(fd, False)
            except (ValueError, OSError):
                pass
            try:
                while True:
                    try:
                        chunk = os.read(fd, 4096)
                    except BlockingIOError:
                        chunk = None
                    except (ValueError, OSError):
                        break
                    if chunk:
                        output.append(decoder.decode(chunk))
                        idle_after_exit = 0
                        continue
                    if chunk == b"":
                        break  # true EOF — all writers closed
                    # No data right now; idle bound like the selector path.
                    if proc.poll() is not None:
                        idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
                    time.sleep(0.1)
            finally:
                try:
                    os.set_blocking(fd, True)
                except (ValueError, OSError):
                    pass
        else:
            while True:
                try:
                    ready = _selector.select(timeout=0.1)
                except (ValueError, OSError) as _exc:
                    logger.warning(
                        "drain: readiness poll failed for fd %s (%r); "
                        "stopping drain — output may be truncated",
                        fd, _exc,
                    )
                    break
                if ready:
                    try:
                        chunk = os.read(fd, 4096)
                    except (ValueError, OSError):
                        break
                    if not chunk:
                        break  # true EOF — all writers closed
                    output.append(decoder.decode(chunk))
                    idle_after_exit = 0
                elif proc.poll() is not None:
                    # bash is gone and the pipe was idle ~100ms; allow two more
                    # cycles for a buffered tail, then stop (a grandchild may
                    # hold the pipe).
                    idle_after_exit += 1
                    if idle_after_exit >= 3:
                        break
    finally:
        _selector.close()


def _start_drain_thread(proc: ProcessHandle, output: _BoundedOutputCollector) -> threading.Thread:
    """Start the daemon thread running :func:`_drain_stdout`."""
    thread = threading.Thread(target=_drain_stdout, args=(proc, output), daemon=True)
    thread.start()
    return thread
