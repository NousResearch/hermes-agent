"""Bounded, profile-local receipts for completed terminal processes.

Receipts are read through process_manage, never replayed as notifications or
adopted as live PIDs. Each producer writes its own file so independent one-shot
parents cannot overwrite each other's results in the running-PID checkpoint.
"""

import json
import logging
import re
import sqlite3
import time
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger("tools.process_registry")

RESULT_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_RETAINED_RESULTS = 64
_RESULT_FIELDS = (
    "id", "command", "cwd", "task_id", "owner_task_id", "session_key",
    "parent_session_id", "started_at", "exit_code", "completion_reason",
    "termination_source", "notify_on_complete",
)


def _result_paths():
    """Prune by completion time, not start time (jobs can take days)."""
    directory = get_hermes_home() / "logs" / "process-results"
    cutoff = time.time() - RESULT_RETENTION_SECONDS
    retained = []
    for path in directory.glob("proc_*.json"):
        try:
            modified = path.stat().st_mtime
            if modified < cutoff:
                path.unlink(missing_ok=True)
            else:
                retained.append((modified, path))
        except FileNotFoundError:
            continue  # Another producer pruned it.
    retained.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    for _, path in retained[MAX_RETAINED_RESULTS:]:
        path.unlink(missing_ok=True)
    return [path for _, path in retained[:MAX_RETAINED_RESULTS]]


def save_completed_result(session) -> bool:
    from agent.redact import redact_sensitive_text, redact_terminal_output
    from tools.process_registry import MAX_OUTPUT_CHARS

    with session._lock:
        record = {key: getattr(session, key) for key in _RESULT_FIELDS}
        record["completion_consumed"] = session.completion_consumed
        record.update(pid=session.pid, host_start_time=session.host_start_time, pid_scope=session.pid_scope,
                      native_job_scope=session.native_job_scope,
                      ownership_generation=session.ownership_generation, handoff_note=session.handoff_note)
        record["output"] = session.output_buffer[-MAX_OUTPUT_CHARS:]
    # Live-output opt-out must not persist raw credentials in durable receipts.
    record["output"] = redact_terminal_output(record["output"], record["command"], force=True)
    record["command"] = redact_sensitive_text(record["command"], code_file=True, force=True)
    record['handoff_note'] = redact_sensitive_text(record['handoff_note'], code_file=True, force=True)
    directory = get_hermes_home() / "logs" / "process-results"
    try:
        from hermes_constants import assert_named_profile_home_live
        assert_named_profile_home_live(directory)
        make_user_dir(directory, mode=0o700, parents=True, exist_ok=True)
        from hermes_cli.active_sessions import _FileLock
        from tools.process_registry import _checkpoint_path
        from tools.process_registry_handoff import merge_ownership, OWNERSHIP_FIELDS
        from tools.process_registry_job import inherit_job_scope
        checkpoint = _checkpoint_path()
        # Match the checkpoint writer's lock order. A retired/cached reader can
        # finish after handoff; merge its immutable process's committed identity
        # before it publishes either the retained receipt or completion event.
        with _FileLock(checkpoint.with_name(checkpoint.name + '.lock')):
            try:
                previous = json.loads(checkpoint.read_text(encoding='utf-8-sig'))
            except FileNotFoundError:
                previous = []
            merge_ownership([record], previous)
            inherit_job_scope([record], previous)
            with _FileLock(directory / '.lock'):
                path = directory / f"{session.id}.json"
                receipts = completed_ownership([record])
                merge_ownership([record], receipts)
                inherit_job_scope([record], receipts)
                # A stale result writer cannot unconsume a completed Stop.
                record['completion_consumed'] |= completed_result_consumed(session.id)
                atomic_json_write(path, record, mode=0o600)
                _result_paths()
        with session._lock:
            session.native_job_scope |= record['native_job_scope']
            if session.ownership_generation <= record['ownership_generation']:
                for field in OWNERSHIP_FIELDS:
                    setattr(session, field, record[field])
            session.completion_consumed |= record['completion_consumed']
            session._result_persisted_generation = record['ownership_generation']
        return True
    except (OSError, ValueError, TypeError):
        # Preserve live delivery on disk failure, but never silently claim durability.
        logger.warning("Could not retain completed process result %s", session.id, exc_info=True)
        return False


def completed_result_consumed(session_id: str) -> bool:
    """Read the existing native result's monotonic consumption verdict after reload."""
    if not isinstance(session_id, str) or not re.fullmatch(r'proc_[\w]+', session_id):
        return False
    path = get_hermes_home() / 'logs' / 'process-results' / f'{session_id}.json'
    try:
        record = json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return False
    if not isinstance(record, dict) or record.get('id') != session_id:
        raise ValueError('invalid retained process result')
    return record.get('completion_consumed') is True


def completed_ownership(entries):
    """Same native process receipts only; caller holds its profile checkpoint lock."""
    records = []
    for entry in entries:
        sid = entry.get('session_id', entry.get('id'))
        if not isinstance(sid, str) or not re.fullmatch(r'proc_[\w]+', sid):
            continue
        path = get_hermes_home() / 'logs' / 'process-results' / f'{sid}.json'
        try:
            record = json.loads(path.read_text(encoding='utf-8-sig'))
        except FileNotFoundError:
            continue
        if not isinstance(record, dict) or record.get('id') != sid:
            raise ValueError('invalid retained process ownership')
        records.append(record)
    return records


def _owns_result(owner: str, parent: str | None) -> bool:
    if not parent:
        return False
    if owner == parent:
        return True
    from hermes_state import SessionDB

    # Pure lineage read on the hot path of every retained-result load; a writable open here
    # was one more writer handle per call inside the gateway (#100896).
    db = SessionDB(read_only=True)
    try:
        return db.get_compression_tip(parent) == owner
    finally:
        db.close()


def load_completed_results(prefix: str = "", *, owns_entry=None, strict=False) -> dict:
    """Restore read-only snapshots; no process handles, watchers, or queue events."""
    from tools.process_registry import ProcessSession, _checkpoint_path

    from gateway.session_context import get_session_env

    owner = get_session_env("HERMES_SESSION_ID", "")
    if not owner and owns_entry is None:
        return {}
    results = {}
    try:
        try:
            checkpoint = json.loads(_checkpoint_path().read_text(encoding='utf-8-sig'))
        except FileNotFoundError:
            checkpoint = []
        if not isinstance(checkpoint, list) or any(not isinstance(row, dict) for row in checkpoint):
            raise ValueError('invalid process checkpoint')
        scopes = {row.get('session_id'): row.get('pid_scope', 'host') for row in checkpoint}
        native_jobs = {row.get('session_id'): row.get('native_job_scope', False) for row in checkpoint}
        paths = _result_paths()
    except (OSError, ValueError):
        if strict:
            raise
        logger.warning("Could not read retained process results", exc_info=True)
        return results
    for path in paths:
        if not path.stem.startswith(prefix):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if record["id"] != path.stem or not re.fullmatch(r"proc_[\w]+", record["id"]):
                continue
            if not (owns_entry(record) if owns_entry is not None else _owns_result(owner, record.get("parent_session_id"))):
                continue
            session = ProcessSession(
                **{key: record[key] for key in _RESULT_FIELDS},
                exited=True, output_buffer=record["output"], completion_consumed=record.get('completion_consumed', False),
                pid=record.get('pid'), host_start_time=record.get('host_start_time'),
                pid_scope=scopes.get(record['id'], record.get('pid_scope', 'host')),
                native_job_scope=native_jobs.get(record['id'], record.get('native_job_scope', False)),
                ownership_generation=record.get('ownership_generation', 0), handoff_note=record.get('handoff_note', ''),
            )
            session._completion_event.set()
            session._result_persisted_generation = session.ownership_generation
            results[session.id] = session
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            if strict:
                raise
            logger.debug("Skipping unreadable process result %s", path.name, exc_info=True)
    return results
