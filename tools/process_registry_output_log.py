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
from typing import Optional, TYPE_CHECKING
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession

logger = logging.getLogger("tools.process_registry")

_TAIL_SECONDS = 0.2
# After the process ended: an orphaned grandchild may keep appending; stop draining after this.
_FINAL_DRAIN_MAX_BYTES = 1 << 20


def output_dir() -> Path:
    return get_hermes_home() / "logs" / "process-output"


def worker_output_log(session_id: str, *, durable: bool = False) -> Optional[Path]:
    """Worker output, or an explicitly enrolled work producer's durable transport."""
    from agent.runtime_session_store import is_worker_process
    if not durable and not is_worker_process():
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
    """Shared ingestion/finalization and durable log transport for ProcessRegistry."""

    def _ingest_output(self, session: "ProcessSession", text: str, *, unless_exited: bool = False) -> None:
        """Buffer a freshly-read chunk, then scan watch patterns and stream it live.
        ``unless_exited`` drops the chunk once the session has exited (atomically with a kill)."""
        watch_text = text
        if session.completion_work_home and not (unless_exited and session.exited):
            from tools.process_work_results import ingest
            watch_text = ingest(session, text)
        if not unless_exited:
            session.append_output(text)
        elif not session.append_output_if_running(text):
            return
        self._check_watch_patterns(session, watch_text)
        self._emit_output(session, text)

    def _finish_exited(self, session: "ProcessSession", exit_code, reason: str = "exited", source: str = "") -> None:
        """Mark a reader-observed exit (a raced kill keeps its own code/reason) and finish.
        Under the session lock, like kill_process's commit: otherwise the kill can land between
        mark_exited's ``killed`` check and its writes, and the reader overwrites the kill."""
        with session._lock:
            session.mark_exited(exit_code, reason, source)
        if session.completion_work_home:
            from tools.process_work_results import finish_without_result
            finish_without_result(session)
        self._move_to_finished(session)

    def _finish_reader(self, session, decoder, append, label, wait, exit_code) -> None:
        """Reader-thread teardown: flush the decoder (a truncated multibyte tail becomes
        one U+FFFD instead of vanishing), reap the child (no zombies), record the exit.

        A process may close stdout long before it exits.  The reader owns a dedicated
        daemon thread, so it must keep waiting rather than publish a false completion
        and discard the only ``Popen`` handle that can reap the child.
        """
        with suppress(Exception):
            tail = decoder.decode(b"", final=True)
            if tail:
                append(tail)
        try:
            wait()
        except Exception as e:
            logger.warning("%s wait failed: %s", label, e, exc_info=True)
            # A PTY child reaped by isalive() already has its exitstatus; only an
            # unknown status must stay tracked for later reconciliation.
            if exit_code() is None:
                return
        self._finish_exited(session, exit_code())

    def _log_reader_loop(self, session) -> None:
        """Tail ``session.output_log`` until its process ends. The spawner decides the exit from
        its Popen; an adopter (no Popen) from the recorded status, else ``lost`` (-1)."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
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
                        if chunk and session.completion_work_home and stream.read(1):
                            # The bounded final drain left bytes unread. Absence
                            # of a frame in this prefix is not missing evidence.
                            session.work_result_error = session.work_result_error or 'output_log_incomplete'
                        break
                    time.sleep(_TAIL_SECONDS)
        except OSError as exc:
            if session.completion_work_home:
                session.work_result_error = 'output_log_unreadable'
            logger.warning("Process output log %s unreadable: %s", session.output_log, exc)
        if proc is not None:
            self._finish_reader(session, decoder, ingest, "Log", proc.wait, lambda: proc.returncode)
            return
        if tail := decoder.decode(b"", final=True):
            ingest(tail)
        code = recorded_exit_code(session.output_log)
        if code is None:
            # SIGKILL / exec past the trap: the spawner is gone and no status was recorded.
            self._finish_exited(session, -1, reason="lost", source="backend_lost")
        else:
            self._finish_exited(session, code)
