"""Bounded, profile-local receipts for completed terminal processes.

Receipts are read through process_manage, never replayed as notifications or
adopted as live PIDs. Each producer writes its own file so independent one-shot
parents cannot overwrite each other's results in the running-PID checkpoint.

Exception — interruption notices: a process killed by Hermes lifecycle cleanup
(``agent_close`` / ``kill_all`` / ``gateway_shutdown``) or
found dead at boot carried a completion contract (``notify_on_complete`` or
``persist_on_release``), so its death must reach the agent instead of silently
disappearing with the killed watcher. Such receipts carry
``interrupt_notice_state="pending"`` until the death notice has been delivered,
and the gateway replays pending ones on startup (see
``ProcessRegistry.arm_interrupt_notices``). The receipt IS the durable notice
ledger — no second store to drift.
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
    "parent_session_id", "started_at", "exited_at", "exit_code", "completion_reason",
    "termination_source", "notify_on_complete", "persist_on_release",
    "watcher_platform", "watcher_chat_id", "watcher_user_id",
    "watcher_user_name", "watcher_thread_id", "watcher_message_id",
)

# Lifecycle kill sources whose kills owed the owner a death notice: the process was
# terminated by Hermes cleanup, not by the user or its own exit. ``gateway_turn_timeout``
# is deliberately absent — an abandoned-turn reap is a deliberate stop ("so a killed
# abandoned process can't revive stopped work") and must not wake anyone.
INTERRUPT_KILL_SOURCES = frozenset({"agent_close", "kill_all", "gateway_shutdown"})
# Exit kinds that interrupt a completion contract and owe a death notice.
_INTERRUPT_REASONS = {"killed": INTERRUPT_KILL_SOURCES, "lost": frozenset({"gateway_downtime"})}


def needs_interrupt_notice(session) -> bool:
    """True when the death of *session* must be announced to its owner.

    A notice is owed only to processes that carried a completion contract
    (``notify_on_complete`` or ``persist_on_release``) and whose exit was a
    lifecycle kill (``INTERRUPT_KILL_SOURCES``) or a boot-detected death during
    gateway downtime (``lost`` / ``gateway_downtime``)."""
    if not (getattr(session, "notify_on_complete", False) or getattr(session, "persist_on_release", False)):
        return False
    reason = getattr(session, "completion_reason", "") or ""
    return getattr(session, "termination_source", "") in _INTERRUPT_REASONS.get(reason, ())


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
    """Retain *session*'s completion receipt under this profile's logs. A hosted agent's process
    has none on this host: its command and output are the tenant's, kept in the live registry only."""
    if session.sandbox_key:
        return
    from agent.redact import redact_sensitive_text, redact_terminal_output
    from tools.process_registry import MAX_OUTPUT_CHARS

    with session._lock:
        record = {key: getattr(session, key) for key in _RESULT_FIELDS}
        record["output"] = session.output_buffer[-MAX_OUTPUT_CHARS:]
    if record.get("exited_at") is None and getattr(session, "exited_at", 0):
        record["exited_at"] = session.exited_at
    # A lifecycle-killed / downtime-lost contract holder owes its owner a death
    # notice; the receipt carries that obligation until the notice is delivered.
    record["interrupt_notice_state"] = "pending" if needs_interrupt_notice(session) else ""
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


def mark_interrupt_notice_delivered(session_id: str) -> bool:
    """Stamp a receipt's interruption notice as delivered (idempotent). True iff one was pending."""
    if not session_id:
        return False
    path = get_hermes_home() / "logs" / "process-results" / f"{session_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if record.get("interrupt_notice_state") != "pending":
        return False
    record["interrupt_notice_state"] = "delivered"
    try:
        atomic_json_write(path, record, mode=0o600)
    except OSError:
        logger.warning("Could not mark interruption notice delivered for %s", session_id, exc_info=True)
        return False
    return True


def pending_interrupt_notices() -> list:
    """Receipts whose death notice is still owed to the owner (``interrupt_notice_state="pending"``)."""
    pending = []
    try:
        paths = _result_paths()
    except OSError:
        logger.warning("Could not read retained process results", exc_info=True)
        return pending
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if record.get("interrupt_notice_state") == "pending" and record.get("id"):
            pending.append(record)
    return pending


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


def session_from_result_record(record: dict):
    """Build a finished ``ProcessSession`` snapshot from a receipt record.

    Receipts written before a field existed keep loading: missing keys take the
    dataclass default (a notice-scan rewrite must not orphan older results)."""
    from tools.process_registry import _CHECKPOINT_DEFAULTS, ProcessSession

    fields = {
        key: record.get(key, _CHECKPOINT_DEFAULTS.get(key))
        for key in _RESULT_FIELDS if key in record or key in _CHECKPOINT_DEFAULTS
    }
    # Receipts written before processes carried a sandbox are a local activity's.
    session = ProcessSession(**fields, sandbox_key=record.get("sandbox_key", ""),
                             exited=True, output_buffer=record.get("output") or "")
    session._completion_event.set()
    return session


def load_completed_results(prefix: str = "") -> dict:
    """Restore read-only snapshots; no process handles, watchers, or queue events."""
    from tools.process_registry import ProcessSession

    from gateway.session_context import get_session_env
    from tools.environments.sandbox_scope import hosted_scope

    owner = get_session_env("HERMES_SESSION_ID", "")
    # A hosted agent's processes leave no receipt here and it sees no local one: never read or prune them.
    if not owner or hosted_scope() is not None:
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
            session = session_from_result_record(record)
            results[session.id] = session
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            logger.debug("Skipping unreadable process result %s", path.name, exc_info=True)
    return results
