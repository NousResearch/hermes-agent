"""File-backed output for background processes a registered worker starts.

A managed worker is a per-turn interpreter. A stdout pipe whose read end that worker's reader
thread owns closes when the worker exits, so the child's next write raises SIGPIPE
(``broken_pipe``) and everything it printed is gone. Under a worker the child writes to a
profile-scoped log instead and its shell records the exit status beside it (the contract the
env-backend wrapper already uses), so whichever interpreter adopts the checkpoint entry next
(the session's next worker, the owner at startup) tails the same file and collects the exit.
"""
import codecs
import json
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
            if exited.name.endswith('.scope.exit'):
                try:
                    if json.loads(exited.read_text(encoding='utf-8-sig')).get('quiescent') is not True:
                        continue
                except (OSError, ValueError, AttributeError):
                    continue  # Unknown native scope evidence never expires on a timer.
            if exited.stat().st_mtime < cutoff:
                if exited.name.endswith('.scope.exit'):
                    from tools.process_registry_job import checkpoint_retains_receipt
                    if checkpoint_retains_receipt(exited):
                        continue
                    log = exited.with_name(exited.name.removesuffix('.scope.exit') + '.log')
                else:
                    log = exited.with_suffix('.log')
                log.unlink(missing_ok=True)
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


def complete_log_source(session, stream):
    """A generic helper binds final regular output to its actual inherited fd."""
    from tools.process_registry_job import read_job_receipt
    row = read_job_receipt(session)
    actual = os.fstat(stream.fileno())
    if stream.tell() != actual.st_size:
        return False  # The source shrank or grew beyond the bytes actually collected.
    if not session.native_job_scope or (row and row.get('unit')):
        return True  # Existing unit/live readers still require successful original-source EOF.
    output = row.get('stdout') if row and row.get('quiescent') else None
    return (isinstance(output, dict)
            and all(type(output.get(key)) is int and output[key] >= 0 for key in ('device', 'inode', 'bytes'))
            and (actual.st_dev, actual.st_ino, actual.st_size) ==
                (output['device'], output['inode'], output['bytes']))


class ProcessOutputLogMixin:
    """The subclass supplies ``_ingest_output``, ``_clean_shell_noise``,
    ``_finish_exited`` and ``_detached_host_fate``. See ProcessRegistry."""

    def _finish_reader(self, session, decoder, append, label, wait, exit_code, *, output_complete=True) -> None:
        """Reader-thread teardown: flush the decoder (a truncated multibyte tail becomes
        one U+FFFD instead of vanishing), reap the child (no zombies), record the exit.

        A process may close stdout long before it exits.  The reader owns a dedicated
        daemon thread, so it must keep waiting rather than publish a false completion
        and discard the only ``Popen`` handle that can reap the child.
        """
        try:
            tail = decoder.decode(b"", final=True)
            if tail:
                append(tail)
        except Exception:
            output_complete = False
            logger.debug("%s output finalization failed", label, exc_info=True)
        try:
            wait()
        except Exception as e:
            # A PTY child reaped by isalive() already has its exitstatus; only an
            # unknown status must stay tracked for later reconciliation.
            if exit_code() is None:
                logger.warning("%s wait failed; leaving process tracked: %s", label, e, exc_info=True)
                return
            logger.warning("%s wait failed; recording known exit status: %s", label, e, exc_info=True)
        self._finish_exited(session, exit_code(), output_complete=output_complete)

    def _log_reader_loop(self, session) -> None:
        """Tail ``session.output_log`` until its process ends. The spawner decides the exit from
        its Popen; an adopter (no Popen) from the recorded status, else ``lost`` (-1)."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        proc = session.process
        output_complete = True
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
                        from tools.process_registry_job import job_quiescent
                        if drained < _FINAL_DRAIN_MAX_BYTES or job_quiescent(session):
                            continue
                        output_complete = False  # A bounded drain is not original-source EOF.
                    if not alive:
                        break
                    time.sleep(_TAIL_SECONDS)
                output_complete &= complete_log_source(session, stream)
        except Exception as exc:
            output_complete = False
            logger.warning("Process output log %s unreadable: %s", session.output_log, exc, exc_info=True)
        if proc is not None:
            self._finish_reader(session, decoder, ingest, "Log", proc.wait, lambda: proc.returncode,
                                output_complete=output_complete)
            return
        try:
            if tail := decoder.decode(b"", final=True):
                ingest(tail)
        except Exception:
            output_complete = False
            logger.debug("Process output log decoder flush failed", exc_info=True)
        code = recorded_exit_code(session.output_log)
        if code is None:
            # SIGKILL / exec past the trap: the spawner is gone and no status was recorded.
            self._finish_exited(session, -1, reason="lost", source="backend_lost", output_complete=output_complete)
        else:
            self._finish_exited(session, code, output_complete=output_complete)
