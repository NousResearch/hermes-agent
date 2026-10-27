"""File-backed output for background processes a registered worker starts.

A managed worker is a per-turn interpreter. A stdout pipe whose read end that worker's reader
thread owns closes when the worker exits, so the child's next write raises SIGPIPE
(``broken_pipe``) and everything it printed is gone. Under a worker the child writes to a
profile-scoped log instead and its shell records the exit status beside it (the contract the
env-backend wrapper already uses), so whichever interpreter adopts the checkpoint entry next
(the session's next worker, the owner at startup) tails the same file and collects the exit.
"""
import codecs
import hashlib
import json
from contextlib import suppress
import logging
import os
import stat
import threading
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
                    from tools.process_registry_job import checkpoint_retains_receipt
                    if checkpoint_retains_receipt(exited):
                        continue
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


def pty_exit_code(pty):
    """A reaped PTY reports either an exit status or an actual terminating signal."""
    code, terminated = getattr(pty, 'exitstatus', None), getattr(pty, 'signalstatus', None)
    if type(code) is int:
        return code
    return -terminated if type(terminated) is int and terminated > 0 else None


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


def bind_unit_log(session, descriptor):
    """Bind the actual launcher descriptor, never a later lookup by pathname."""
    actual = os.fstat(descriptor)
    if not stat.S_ISREG(actual.st_mode):
        raise ValueError('native unit stdout is not a regular file')
    session.output_log_state = dict(unit=session.systemd_unit, device=actual.st_dev, inode=actual.st_ino,
                                   bytes=0, sha256=hashlib.sha256(b'').hexdigest(), final=False)


def valid_unit_log_state(state):
    return (isinstance(state, dict) and isinstance(state.get('unit'), str)
            and state['unit'].startswith('hermes-worker-') and type(state.get('final')) is bool
            and all(type(state.get(k)) is int and state[k] >= 0 for k in ('device', 'inode', 'bytes'))
            and isinstance(state.get('sha256'), str) and len(state['sha256']) == 64
            and all(c in '0123456789abcdef' for c in state['sha256']))


def verify_unit_log(stream, state):
    """An unchanged collected prefix binds subsequent reads to the original source."""
    if not valid_unit_log_state(state):
        raise ValueError('original native unit log witness unavailable')
    actual = os.fstat(stream.fileno())
    if (not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (state['device'], state['inode'])
            or actual.st_size < state['bytes']):
        raise ValueError('original native unit log changed')
    digest, remaining = hashlib.sha256(), state['bytes']
    while remaining:
        chunk = stream.read(min(65536, remaining))
        if not chunk:
            raise ValueError('original native unit log truncated')
        digest.update(chunk)
        remaining -= len(chunk)
    if digest.hexdigest() != state['sha256']:
        raise ValueError('collected native unit output changed')
    stream.seek(0)


def unit_log_pending(value):
    get = value.get if isinstance(value, dict) else lambda k, default=None: getattr(value, k, default)
    state = get('output_log_state')
    return bool(get('systemd_unit') and get('native_job_scope') and get('output_log')
                and not (valid_unit_log_state(state) and state['final']))


def inherit_unit_log(entries, previous):
    """Cached writers cannot erase the original source or its finalized collection."""
    changed = set()
    for entry in entries:
        sid = entry.get('session_id', entry.get('id'))
        for old in previous:
            if old.get('session_id', old.get('id')) != sid or not valid_unit_log_state(old.get('output_log_state')):
                continue
            if any(entry.get(k) != old.get(k) for k in ('pid', 'host_start_time', 'started_at')):
                raise ValueError('native unit output identity mismatch')
            state, before = old['output_log_state'], entry.get('output_log_state')
            if (valid_unit_log_state(before)
                    and any(state[k] != before[k] for k in ('unit', 'device', 'inode'))):
                raise ValueError('native unit original source identity mismatch')
            if (not valid_unit_log_state(before) or (state['final'] and not before['final'])
                    or (state['final'] == before['final'] and state['bytes'] > before['bytes'])):
                entry['output_log_state'] = dict(state)
                entry['output_log'] = old.get('output_log', entry.get('output_log', ''))
                if not state['final']:
                    entry['systemd_unit'] = old.get('systemd_unit') or state['unit']
                changed.add(sid)
    return changed


def merge_unit_log_result(record, previous):
    """Only an actual finalized retained payload may promote a stale unit snapshot."""
    state = previous.get('output_log_state')
    if (valid_unit_log_state(state) and type(previous.get('exit_code')) is int
            and previous.get('completion_reason') != 'lost'):
        if any(record.get(k) != previous.get(k) for k in ('id', 'pid', 'host_start_time', 'started_at')):
            raise ValueError('native unit result identity mismatch')
        for key in ('exit_code', 'completion_reason', 'termination_source'):
            record[key] = previous[key]  # Scope collection does not change the primary's actual status.
    if (previous.get('result_complete') is True and type(previous.get('exit_code')) is int
            and previous.get('completion_reason') != 'lost' and valid_unit_log_state(state) and state['final']):
        if any(record.get(k) != previous.get(k) for k in ('id', 'pid', 'host_start_time', 'started_at')):
            raise ValueError('final native unit output identity mismatch')
        for key in ('output', 'exit_code', 'completion_reason', 'termination_source', 'result_complete',
                    'output_log', 'output_log_state'):
            record[key] = previous[key]
        return True
    if unit_log_pending(record):
        record['result_complete'] = False
    return False


def finalized_unit_result(entry, receipt):
    """An old writer cannot resurrect a discharged unit whose actual final payload is durable."""
    from tools.process_registry_handoff import OWNERSHIP_FIELDS
    from tools.process_registry_job import job_quiescent, read_job_receipt
    return bool(receipt and receipt.get('result_complete') is True and type(receipt.get('exit_code')) is int
                and receipt.get('completion_reason') != 'lost'
                and valid_unit_log_state(receipt.get('output_log_state')) and receipt['output_log_state']['final']
                and entry.get('systemd_unit', '') in ('', receipt['output_log_state']['unit'])
                and (read_job_receipt(entry) or {}).get('unit') == receipt['output_log_state']['unit']
                and all(entry.get(k) == receipt.get(k) for k in (*OWNERSHIP_FIELDS, 'pid', 'host_start_time'))
                and job_quiescent(entry))


class ProcessOutputLogMixin:
    """The subclass supplies ``_ingest_output``, ``_clean_shell_noise``,
    ``_finish_exited`` and ``_detached_host_fate``. See ProcessRegistry."""

    def _finalize_unit_publication(self, session):
        """This original publisher has finished reading; commit final scoped bytes before its event."""
        from tools.process_registry_job import unit_quiescent
        if session.systemd_unit and session.output_log and unit_quiescent(session):
            self._collect_unit_log(session, _original_publisher=True)

    def _collect_unit_log(self, session, *, _original_publisher=False):
        """Collect final original bytes only after native unit emptiness and reader completion."""
        from tools.process_registry_job import job_quiescent, read_job_receipt
        from tools.process_registry_results import save_completed_result
        reader = session._reader_thread
        publisher = _original_publisher and (reader is None or reader is threading.current_thread())
        if not job_quiescent(session) or (reader is not None and not publisher and not session._completion_event.wait(1.0)):
            return False
        save_completed_result(session)  # May import an actual same-job finalized retained payload.
        state = session.output_log_state
        if valid_unit_log_state(state) and state['unit'] != (read_job_receipt(session) or {}).get('unit'):
            return False
        if valid_unit_log_state(state) and state['final'] and session._result_complete:
            return session._result_persisted_generation == session.ownership_generation
        with session._lock:
            session._result_complete, session._result_persisted_generation = False, -1
        if type(session.exit_code) is not int or session.completion_reason == 'lost':
            code = recorded_exit_code(session.output_log)
            if code is None:
                return False  # Unit emptiness cannot reconstruct the primary's missing status.
            with session._lock:
                session.exit_code, session.completion_reason, session.termination_source = code, 'exited', ''
        try:
            decoder, digest = codecs.getincrementaldecoder('utf-8')(errors='replace'), hashlib.sha256()
            prefix, remaining = hashlib.sha256(), state['bytes'] if valid_unit_log_state(state) else 0
            output, total, size, head = '', 0, 0, True
            with open(session.output_log, 'rb') as stream:
                verify_unit_log(stream, state)
                for chunk in iter(lambda: stream.read(65536), b''):
                    digest.update(chunk)
                    prefix.update(chunk[:remaining])
                    remaining = max(0, remaining - len(chunk))
                    size += len(chunk)
                    text = decoder.decode(chunk)
                    if head:
                        text = self._clean_shell_noise(text)
                        head = not text.strip()
                    total += len(text)
                    output = (output + text)[-session.max_output_chars:]
                tail = decoder.decode(b'', final=True)
                total += len(tail)
                output = (output + tail)[-session.max_output_chars:]
                actual = os.fstat(stream.fileno())
                if (actual.st_size != size or prefix.hexdigest() != state['sha256']
                        or (actual.st_dev, actual.st_ino) != (state['device'], state['inode'])):
                    raise ValueError('native unit final source changed during collection')
            with session._lock:
                session.output_buffer, session.total_output_chars = output, total
                session.output_log_state = {**state, 'bytes': size, 'sha256': digest.hexdigest(), 'final': True}
                session._result_complete = True
            return save_completed_result(session)
        except (OSError, ValueError, TypeError):
            logger.warning('Native unit final output unavailable: %s', session.id, exc_info=True)
            return False

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
                state, digest = session.output_log_state, hashlib.sha256()
                if valid_unit_log_state(state):
                    verify_unit_log(stream, state)
                drained = 0
                while True:
                    # Decided BEFORE the read: once the process is gone every byte it wrote is readable.
                    alive = running()
                    chunk = stream.read(65536)
                    if chunk:
                        digest.update(chunk)
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
                if valid_unit_log_state(state):
                    from tools.process_registry_job import read_job_receipt
                    receipt = read_job_receipt(session)
                    session.output_log_state = {**state, 'bytes': stream.tell(), 'sha256': digest.hexdigest(),
                                                'final': bool(output_complete and receipt and receipt.get('unit') == state['unit']
                                                              and receipt['quiescent'])}
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
            self._finish_exited(session, -1, reason="lost", source="backend_lost",
                                output_complete=output_complete and not session.systemd_unit)
        else:
            self._finish_exited(session, code, output_complete=output_complete)
