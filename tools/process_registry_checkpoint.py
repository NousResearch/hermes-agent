"""Running-process checkpoint persistence and PID-safe recovery.

Every Hermes process on a profile (gateway, CLI, TUI, cron, one-shots) checkpoints into the same
file, so each entry names its writer: a process rewrites only its own entries, and recovery adopts
only jobs whose writer is gone."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.process_registry")


def _entry_key(entry: Dict[str, Any]) -> tuple:
    return entry.get("session_id"), entry.get("owner_pid"), entry.get("owner_started_at")


def _written_by(entry: Dict[str, Any], writer: Dict[str, Any]) -> bool:
    """True when ``entry`` was written by the process incarnation ``writer`` describes."""
    from gateway.status import start_time_fingerprints_match

    recorded, current = entry.get("owner_started_at"), writer["owner_started_at"]
    return entry.get("owner_pid") == writer["owner_pid"] and (
        recorded is None or current is None or start_time_fingerprints_match(recorded, current))


class ProcessCheckpointMixin:
    # ----- Checkpoint (crash recovery) -----

    def _checkpoint_writer(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Identity of ``pid`` (default: this process) as stamped on its checkpoint entries."""
        pid = pid or os.getpid()
        return {"owner_pid": pid, "owner_started_at": self._safe_host_start_time(pid)}

    def _write_checkpoint(self, extra_entries: Optional[List[Dict[str, Any]]] = None, consumed=frozenset()):
        """Replace this process's entries in the checkpoint file atomically. Other writers'
        entries stay (a dead writer's wait for recovery) unless recovery ``consumed`` them."""
        from tools.process_registry import _checkpoint_path, _CHECKPOINT_FIELDS

        try:
            writer = self._checkpoint_writer()
            with self._lock:
                entries = []
                for s in self._running.values():
                    if s.exited:
                        continue
                    # Backfill the start time so recovery can detect PID recycling
                    # even for sessions spawned before this field existed.
                    if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                        s.host_start_time = self._safe_host_start_time(s.pid)
                    entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}, **writer}
                    # Redact inline credentials before persisting (~/.hermes/processes.json).
                    # Recovery uses command only for display (adoption re-validates the
                    # PID, never re-runs it), so masking is lossless.
                    # See #77484.
                    entry["command"] = redact_sensitive_text(s.command, code_file=True)
                    entry["owner_task_id"] = s.owner_task_id or s.task_id
                    entries.append(entry)
                if extra_entries:
                    tracked_ids = {item.get("session_id") for item in entries}
                    entries.extend(item for item in extra_entries if item.get("session_id") not in tracked_ids)
            from hermes_constants import mkdir_under_hermes_home
            from tools.skill_usage import skill_file_lock
            from utils import atomic_json_write
            path = _checkpoint_path()
            mkdir_under_hermes_home(path.parent)  # the lock file must not resurrect a deleted profile
            # Read-merge-write under a cross-process lock: rewriting the whole file with only this
            # process's jobs erased the gateway's crash-recovery entries whenever a CLI spawned.
            with skill_file_lock(path.with_name(path.name + ".lock")):
                try:
                    on_disk = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    on_disk = []
                kept = [e for e in on_disk if not _written_by(e, writer) and _entry_key(e) not in consumed]
                atomic_json_write(path, kept + entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def recover_from_checkpoint(self) -> int:
        """On gateway startup, probe PIDs from the checkpoint file; returns how many
        were recovered as detached sessions."""
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS, _checkpoint_path,
            _CHECKPOINT_DEFAULTS, _WATCHER_ROUTE_KEYS, _stop_systemd_unit,
        )

        checkpoint_path = _checkpoint_path()
        if not checkpoint_path.exists():
            return 0
        try:
            entries = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        recovered = 0
        unresolved_scope_entries: List[Dict[str, Any]] = []
        writer, consumed = self._checkpoint_writer(), set()
        for entry in entries:
            owner = entry.get("owner_pid")
            if owner is None:
                # Written before entries named their writer, which may be a CLI still running the
                # job: adopting it would hand that job to this process's shutdown kill_all.
                logger.info("Not recovering session %s: checkpoint entry has no writer identity",
                            entry.get("session_id", "?"))
                consumed.add(_entry_key(entry))
                continue
            # A live writer (CLI, TUI, cron worker) still owns its job; an entry carrying this
            # process's own identity predates an in-place exec and is ours to adopt.
            if not _written_by(entry, writer) and self._is_host_pid_alive(owner) and _written_by(
                    entry, self._checkpoint_writer(owner)):
                continue
            consumed.add(_entry_key(entry))
            pid, pid_scope = entry.get("pid"), entry.get("pid_scope", "host")
            if not pid:
                continue
            # The registry is process-global, so every profile's checkpoint carries every live
            # process; a multiplexer recovering several homes must adopt each session once.
            with self._lock:
                already_tracked = entry.get("session_id") in self._running
            if already_tracked:
                continue
            if pid_scope != "host":  # in-sandbox PIDs mean nothing once the env handle is gone
                logger.info(
                    "Skipping recovery for non-host process: %s (pid=%s, scope=%s)",
                    entry.get("command", "unknown")[:60], pid, pid_scope)
                continue
            # Alive AND the same process: across a restart the kernel may have
            # recycled the PID onto a stranger, and adopting it would let a later
            # kill tree-kill e.g. a browser.
            if not self._host_pid_is_ours(pid, entry.get("host_start_time")):
                if self._is_host_pid_alive(pid):
                    logger.info(
                        "Not recovering session %s: pid %d is alive but its "
                        "start time no longer matches — PID was recycled onto "
                        "an unrelated process; refusing to adopt it.",
                        entry.get("session_id", "?"), pid)
                systemd_unit = entry.get("systemd_unit", "")
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    logger.warning(
                        "Could not reap persisted scope %s for dead wrapper pid %s; "
                        "retaining checkpoint entry for the next startup",
                        systemd_unit, pid)
                    unresolved_scope_entries.append(entry)
                continue
            fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            fields.update(
                command=entry.get("command", "unknown"),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                started_at=entry.get("started_at", time.time()))
            # detached: can't read output, but can report status + kill
            session = ProcessSession(id=entry["session_id"], detached=True, **fields)
            with self._lock:
                self._running[session.id] = session
            recovered += 1
            logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)
            # Re-enqueue watcher so gateway can resume notifications
            if session.watcher_interval > 0:
                self.pending_watchers.append({
                    "session_id": session.id,
                    "check_interval": session.watcher_interval,
                    "session_key": session.session_key,
                    **{key: getattr(session, f"watcher_{key}") for key in _WATCHER_ROUTE_KEYS},
                    "notify_on_complete": session.notify_on_complete,
                    "parent_session_id": session.parent_session_id,
                })
        self._write_checkpoint(extra_entries=unresolved_scope_entries, consumed=consumed)
        return recovered
