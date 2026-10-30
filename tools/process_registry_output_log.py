"""File-backed output for background processes a registered worker starts.

A managed worker is a per-turn interpreter. A stdout pipe whose read end that worker's reader
thread owns closes when the worker exits, so the child's next write raises SIGPIPE
(``broken_pipe``) and everything it printed is gone. Under a worker the child writes to a
profile-scoped log instead and its shell records the exit status beside it (the contract the
env-backend wrapper already uses), so whichever interpreter adopts the checkpoint entry next
(the session's next worker, the owner at startup) tails the same file and collects the exit.
"""
import codecs
from contextlib import suppress
import logging
import os
from pathlib import Path
import time
from typing import Optional
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home

logger = logging.getLogger("tools.process_registry")

_TAIL_SECONDS = 0.2
# After the process ended: an orphaned grandchild may keep appending; stop draining after this.
_FINAL_DRAIN_MAX_BYTES = 1 << 20


def iter_pty_output(pty):
    """Drain the direct child's buffered tail without waiting on inherited slaves."""
    from tools.process_registry import _IS_WINDOWS
    from tools.environments.fd_wait import open_fd_waiter
    fd = getattr(pty, "fd", None)
    waiter = open_fd_waiter(fd) if not _IS_WINDOWS and isinstance(fd, int) else None
    if not _IS_WINDOWS and isinstance(fd, int) and waiter is None:
        raise OSError("PTY readiness setup failed")
    drained = 0
    try:
        while True:
            if waiter is not None:
                exited = not pty.isalive()
                if not waiter.wait(0 if exited else _TAIL_SECONDS):
                    if exited:
                        return
                    continue
                try:
                    chunk = os.read(fd, 4096)
                except OSError as exc:
                    import errno
                    if exc.errno == errno.EIO:
                        return  # POSIX PTY slave closure.
                    raise
                if exited:
                    drained += len(chunk)
            else:
                chunk = pty.read(4096)
            if chunk:
                yield chunk
            elif not pty.isalive():
                return
            if drained >= _FINAL_DRAIN_MAX_BYTES:
                return
    except EOFError:
        return
    finally:
        if waiter is not None:
            waiter.close()


def output_dir() -> Path:
    return get_hermes_home() / "logs" / "process-output"


def worker_output_log(session_id: str) -> Optional[Path]:
    """The log a worker-spawned process writes to; None outside a registered worker."""
    from agent.runtime_session_store import is_worker_process
    if not is_worker_process():
        return None
    directory = output_dir()
    make_user_dir(directory, mode=0o700, parents=True, exist_ok=True)
    from tools.process_registry_results import RESULT_RETENTION_SECONDS
    # Only a recorded exit retires a log: a quiet process can run longer than the retention.
    cutoff = time.time() - RESULT_RETENTION_SECONDS
    for exited in directory.glob("proc_*.exit"):
        with suppress(FileNotFoundError):
            if exited.stat().st_mtime < cutoff:
                exited.with_suffix(".log").unlink(missing_ok=True)
                exited.unlink()
    return directory / f"{session_id}.log"


def adoptable_log(path) -> str:
    """A checkpointed log path, only when it names this profile's output directory."""
    log = Path(str(path or ""))
    return str(log) if path and log.parent == output_dir() and log.suffix == ".log" else ""


def open_output_log(log: Path) -> int:
    """Owner-only append descriptor for the child's stdout+stderr (the parent closes its copy)."""
    return os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o600)


def _exit_path(log) -> Path:
    return Path(log).with_suffix(".exit")


def record_exit_command(log: Path, command: str) -> str:
    """Prefix ``command`` so the shell writes its exit status beside the log: an interpreter
    that adopts the process after the spawner exited has no Popen to wait on."""
    import shlex
    target = shlex.quote(_exit_path(log).as_posix())
    return f"__hermes_exit_file={target}; trap 'printf \"%s\\n\" \"$?\" > \"$__hermes_exit_file\"' EXIT\n{command}"


def recorded_exit_code(log) -> Optional[int]:
    try:
        return int(_exit_path(log).read_text(encoding="utf-8-sig").strip())
    except (OSError, ValueError):
        return None


class ProcessOutputLogMixin:
    """The subclass supplies ``_ingest_output``, ``_clean_shell_noise``, ``_finish_reader``,
    ``_finish_exited`` and ``_detached_host_fate``. See ProcessRegistry."""

    def _start_adopted_reader(self, session, collector, finalize_capture):
        import threading
        ready = threading.Event()

        def read_after_transfer(owned):
            ready.wait()
            self._reader_loop(owned)

        from contextvars import copy_context
        # Start the reader before closing the collector, but do not expose this session
        # to ANY completion owner until finalization has transferred capture state.
        reader = threading.Thread(target=copy_context().run, args=(read_after_transfer, session),
                                  daemon=True, name=f"proc-reader-{session.id}")
        session._reader_thread = reader
        reader.start()  # Failure here leaves foreground ownership and collector intact.
        try:
            if finalize_capture is not None:
                finalize_capture()
        except Exception:
            logger.debug("adopted capture finalization failed", exc_info=True)
            session.capture_failed("handoff finalization")
        finally:
            if collector is not None and collector.capture_error:
                session.capture_error = session.capture_error or collector.capture_error
            with self._lock:
                self._prune_if_needed()
                self._running[session.id] = session
                ready.set()
        try:
            self._write_checkpoint()
        except Exception:
            # Ownership is committed: never let checkpoint failure resume foreground reads.
            logger.debug("adopted reader checkpoint failed", exc_info=True)

    @staticmethod
    def _exit_fields(session) -> dict:
        return {
            "exit_code": session.exit_code,
            "completion_reason": session.completion_reason,
            "termination_source": session.termination_source,
            **session.capture_fields(),
        }

    def _finish_reader(self, session, decoder, append, label, wait, exit_code) -> None:
        """Flush captured bytes and reap the child without inventing an exit status."""
        try:
            tail = decoder.decode(b"", final=True) if decoder is not None else ""
            if tail:
                append(tail)
        except Exception:  # health: allow BLE001 -- decoder failure must not skip child reaping
            session.capture_failed("background decoder finalization")
        try:
            wait()
        except Exception as e:  # health: allow BLE001 -- third-party PTY wait exceptions must retain known status or leave tracking intact
            # A PTY child reaped by isalive() already has its exitstatus; only an
            # unknown status must stay tracked for later reconciliation.
            if exit_code() is None:
                logger.warning("%s wait failed; leaving process tracked: %s", label, e)
                return
            logger.warning("%s wait failed; recording known exit status: %s", label, e)
        self._finish_exited(session, exit_code())

    def _log_reader_loop(self, session) -> None:
        """Tail ``session.output_log`` until its process ends. The spawner decides the exit from
        its Popen; an adopter (no Popen) from the recorded status, else ``lost`` (-1)."""
        decoder = None
        proc = session.process
        # poll()/wait() reconcile asks this reader to finish instead of reading a pipe itself.
        session._reader_selectable = True

        head_noise = True

        def ingest(text: str) -> None:
            nonlocal head_noise
            if head_noise:  # ``bash -lic`` startup warnings, as the pipe reader strips them
                text = self._clean_shell_noise(text)
                head_noise = not text.strip()
            self._ingest_output(session, text)

        def running() -> bool:
            if session._reader_finish_requested.is_set():
                return False
            if proc is not None:
                return proc.poll() is None
            return self._detached_host_fate(session.pid, session.host_start_time) == "running"
        try:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            with open(session.output_log, "rb") as stream:
                drained = 0
                while True:
                    # Decided BEFORE the read: once the process is gone every byte it wrote is readable.
                    alive = running()
                    chunk = stream.read(65536)
                    if chunk:
                        if text := decoder.decode(chunk):
                            ingest(text)
                        drained += 0 if alive else len(chunk)
                        if drained < _FINAL_DRAIN_MAX_BYTES:
                            continue
                    if not alive:
                        break
                    time.sleep(_TAIL_SECONDS)
        except Exception:  # health: allow BLE001 -- log/decoder boundary preserves lifecycle and reports capture loss
            session.capture_failed("background log read")
            self._write_checkpoint()
            # Keep monitoring an adopted child even though its log is unreadable.
            while proc is None and running():
                time.sleep(_TAIL_SECONDS)
        if proc is not None:
            self._finish_reader(session, decoder, ingest, "Log", proc.wait, lambda: proc.returncode)
            return
        try:
            if decoder is not None and (tail := decoder.decode(b"", final=True)):
                ingest(tail)
        except Exception:  # health: allow BLE001 -- capture finalization cannot skip adopted child status recording
            session.capture_failed("background decoder finalization")
        code = recorded_exit_code(session.output_log)
        if code is None:
            # SIGKILL / exec past the trap: the spawner is gone and no status was recorded.
            self._finish_exited(session, -1, reason="lost", source="backend_lost")
        else:
            self._finish_exited(session, code)
