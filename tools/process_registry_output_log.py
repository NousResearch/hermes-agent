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
from typing import TYPE_CHECKING, Optional
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home

if TYPE_CHECKING:
    from tools.process_registry import ProcessRegistry

logger = logging.getLogger("tools.process_registry")

_TAIL_SECONDS = 0.2
# After the process ended: an orphaned grandchild may keep appending; stop draining after this.
_FINAL_DRAIN_MAX_BYTES = 1 << 20


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


def finish_reader(registry: "ProcessRegistry", session, decoder, append, label, wait, exit_code, proc=None) -> None:
    """Reader-thread teardown: flush the decoder (a truncated multibyte tail becomes
    one U+FFFD instead of vanishing), reap the child (no zombies), record the exit.

    A process may close stdout long before it exits.  The reader owns a dedicated
    daemon thread, so it must keep waiting rather than publish a false completion
    and discard the only ``Popen`` handle that can reap the child.
    For pipe-backed readers (``proc`` given), EOF on the capture pipe is NOT the
    same as process exit: the child may have closed its stdout/stderr while still
    alive (issue #86416), so ``wait`` blocks until the real child exit. When
    observation itself fails (``wait`` raised and ``poll`` reports nothing), leave
    the session in _running and let poll()/wait() reconcile via
    _reconcile_local_exit when it actually terminates. An unknown exit code is
    never finalized as a completion.
    """
    with suppress(Exception):
        tail = decoder.decode(b"", final=True)
        if tail:
            append(tail)
    rc = None
    try:
        rc = wait()
    except Exception as e:
        # A PTY child reaped by isalive() already has its exitstatus; only an
        # unknown status must stay tracked for later reconciliation.
        if proc is None and exit_code() is None:
            logger.warning("%s wait failed; leaving process tracked: %s", label, e, exc_info=True)
            return
        if proc is None:
            logger.warning("%s wait failed; recording known exit status: %s", label, e, exc_info=True)
        else:
            logger.debug("%s wait timed out or failed: %s", label, e, exc_info=True)
    if rc is None and proc is not None:
        rc = proc.poll()
    if rc is None and proc is not None:
        # The child state cannot be observed — keep the session running and
        # do not emit a completion.
        logger.info(
            "Process %s reached EOF on capture pipe but the direct child "
            "is still running (pid=%s); not marking as exited.",
            session.id,
            session.pid,
        )
        return
    code = exit_code()
    if code is None:
        # Never fabricate a completion from an unknown exit (issue #86416).
        return
    registry._finish_exited(session, code)


class ProcessOutputLogMixin:
    """The subclass supplies ``_ingest_output``, ``_clean_shell_noise``, ``_finish_reader``,
    ``_finish_exited`` and ``_detached_host_fate``. See ProcessRegistry."""

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
                        break
                    time.sleep(_TAIL_SECONDS)
        except OSError as exc:
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
