"""Bounded, profile-local receipts for completed terminal processes.

Receipts are read through process_manage and never adopted as live PIDs. A receipt
whose gateway completion notification has not settled (``notify_pending``) is the
only durable trace of that notification: the completion queue and the watcher are
in-memory, so gateway startup re-arms a watcher from it. Each producer writes its
own file so independent one-shot parents cannot overwrite each other's results in
the running-PID checkpoint.
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
# Past this age a replayed notification would start a turn nobody is waiting on; the
# receipt stays readable through process_manage. Matches async delegation's replay cap.
MAX_NOTIFICATION_REPLAY_AGE_SECONDS = 48 * 60 * 60


def _route_fields() -> tuple:
    """The watcher route, so a restart can re-address the notification without its watcher."""
    from tools.process_registry import _WATCHER_ROUTE_KEYS
    return (*(f"watcher_{key}" for key in _WATCHER_ROUTE_KEYS), "watcher_interval")


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


def save_completed_result(session) -> None:
    from agent.redact import redact_sensitive_text, redact_terminal_output
    from tools.process_registry import MAX_OUTPUT_CHARS

    with session._lock:
        record = {key: getattr(session, key) for key in (*_RESULT_FIELDS, *_route_fields())}
        record["output"] = session.output_buffer[-MAX_OUTPUT_CHARS:]
    # A gateway watcher (watcher_interval) owes this completion a notification until it settles.
    record["notify_pending"] = bool(record["notify_on_complete"] and record["watcher_interval"])
    record["completed_at"] = time.time()
    # Live-output opt-out must not persist raw credentials in durable receipts.
    record["output"] = redact_terminal_output(record["output"], record["command"], force=True)
    record["command"] = redact_sensitive_text(record["command"], code_file=True, force=True)
    directory = get_hermes_home() / "logs" / "process-results"
    try:
        from hermes_constants import assert_named_profile_home_live
        assert_named_profile_home_live(directory)
        make_user_dir(directory, mode=0o700, parents=True, exist_ok=True)
        atomic_json_write(directory / f"{session.id}.json", record, mode=0o600)
        _result_paths()
    except OSError:
        # Preserve live delivery on disk failure, but never silently claim durability.
        logger.warning("Could not retain completed process result %s", session.id, exc_info=True)


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


def _session_from_record(record: dict):
    """Read-only exited snapshot: no process handle or PID, so nothing can be re-run or signalled."""
    from tools.process_registry import ProcessSession

    session = ProcessSession(
        **{key: record[key] for key in _RESULT_FIELDS},
        **{key: record[key] for key in _route_fields() if key in record},
        exited=True, output_buffer=record["output"],
    )
    session._completion_event.set()
    return session


def load_completed_results(prefix: str = "") -> dict:
    """Restore read-only snapshots; no process handles, watchers, or queue events."""
    from gateway.session_context import get_session_env

    owner = get_session_env("HERMES_SESSION_ID", "")
    if not owner:
        return {}
    results = {}
    try:
        paths = _result_paths()
    except OSError:
        logger.warning("Could not read retained process results", exc_info=True)
        return results
    for path in paths:
        if not path.stem.startswith(prefix):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if record["id"] != path.stem or not re.fullmatch(r"proc_[\w]+", record["id"]):
                continue
            if not _owns_result(owner, record.get("parent_session_id")):
                continue
            session = _session_from_record(record)
            results[session.id] = session
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            logger.debug("Skipping unreadable process result %s", path.name, exc_info=True)
    return results


def pending_notification_results() -> list:
    """This profile's exited sessions whose gateway notification never settled, oldest first.

    Startup-only, so not ownership-filtered: no session owns anything yet, and the receipt
    carries its own route. Over-age receipts are skipped, not replayed.
    """
    cutoff = time.time() - MAX_NOTIFICATION_REPLAY_AGE_SECONDS
    pending = []
    for path in _result_paths():
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if record["id"] != path.stem or not re.fullmatch(r"proc_[\w]+", record["id"]):
                continue
            if record.get("notify_pending") and record["completed_at"] >= cutoff:
                pending.append((record["completed_at"], record["id"], _session_from_record(record)))
        except (OSError, ValueError, KeyError, TypeError):
            logger.debug("Skipping unreadable process result %s", path.name, exc_info=True)
    return [session for *_, session in sorted(pending, key=lambda item: item[:2])]


def mark_result_notified(session_id: str) -> None:
    """Settle a receipt's pending notification so a later startup does not re-arm it."""
    path = get_hermes_home() / "logs" / "process-results" / f"{session_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        if record.get("notify_pending"):
            record["notify_pending"] = False
            atomic_json_write(path, record, mode=0o600)
    except FileNotFoundError:
        return  # No durable receipt (disk failure at exit, or a receipt-less test session).
    except (OSError, ValueError):
        # The next startup re-arms it: one duplicate notification, never a lost one.
        logger.warning("Could not settle the notification on process result %s", session_id, exc_info=True)
