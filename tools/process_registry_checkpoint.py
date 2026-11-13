"""Running-process checkpoint persistence and PID-safe recovery."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.process_registry")


class ProcessCheckpointMixin:
    def _detached_host_fate(self, pid: Optional[int], expected_start: Optional[int]) -> str:
        """Subclass supplies the live PID decision. See ProcessRegistry."""
        raise NotImplementedError

    def arm_interrupt_notices(self, *, rescan: bool = False) -> int:
        """Subclass supplies the pending-death-notice replay (``ProcessRegistry.arm_interrupt_notices``)."""
        raise NotImplementedError

    @staticmethod
    def _record_downtime_death(entry: dict[str, Any], ProcessSession, save_completed_result) -> None:
        """A checkpoint entry whose PID is gone at boot: the process died while the gateway was
        down. For a completion-contract holder that death owed its owner a notice — write a
        ``lost`` / ``gateway_downtime`` receipt carrying ``interrupt_notice_state="pending"`` so
        ``arm_interrupt_notices`` replays it through the completion wake chain. Unknown outcome
        (no collected exit), so the notice says exactly that."""
        if not (entry.get("notify_on_complete") or entry.get("persist_on_release")):
            return  # silent process: no completion contract, no death notice
        now = time.time()
        fields = {
            key: entry.get(key) for key in (
                "command", "task_id", "owner_task_id", "session_key", "parent_session_id",
                "started_at", *(f"watcher_{k}" for k in (
                    "platform", "chat_id", "user_id", "user_name", "thread_id", "message_id")),
            )
        }
        fields["id"] = entry.get("session_id", "")
        fields["command"] = entry.get("command", "unknown")
        session = ProcessSession(
            **{k: v for k, v in fields.items() if v is not None},
            exited=True, exit_code=None, completion_reason="lost",
            termination_source="gateway_downtime", exited_at=now,
            notify_on_complete=bool(entry.get("notify_on_complete")),
            persist_on_release=bool(entry.get("persist_on_release")),
        )
        try:
            save_completed_result(session)
            logger.warning(
                "Process %s died while the gateway was down (pid %s gone at recovery); "
                "queued an interruption notice for its owner", session.id, entry.get("pid"))
        except Exception:
            logger.warning("Could not record downtime death for %s", entry.get("session_id"), exc_info=True)

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(self, extra_entries: Optional[list[dict[str, Any]]] = None):
        """Write running process metadata to the checkpoint file atomically."""
        from tools.process_registry import _checkpoint_path, _CHECKPOINT_FIELDS
        from tools.environments.sandbox_scope import hosted_scope

        # A hosted caller only changes its own agent's records, which are never persisted here, so
        # it has no host change to write: it neither takes the lock nor rewrites this host's file.
        if hosted_scope() is not None:
            return
        try:
            with self._lock:
                entries = []
                for s in self._running.values():
                    # A hosted agent's process is not this host's to persist: its record (command,
                    # cwd, owner) stays in memory, whoever triggers the write (scoped or not).
                    if s.exited or s.sandbox_key:
                        continue
                    # Backfill the start time so recovery can detect PID recycling
                    # even for sessions spawned before this field existed.
                    if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                        s.host_start_time = self._safe_host_start_time(s.pid)
                    entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
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
                known = {*self._running, *self._finished}
            from hermes_cli.active_sessions import _FileLock
            from utils import atomic_json_write
            path = _checkpoint_path()
            # Every interpreter of the profile (owner, each managed worker) rewrites this file from
            # its own table; a worker's output-log handoff is its only path to the next one, so a
            # writer keeps the handoffs it does not track (read-merge-write under one file lock).
            with _FileLock(path.with_name(path.name + ".lock")):
                entries.extend(self._foreign_handoffs(path, known))
                atomic_json_write(path, entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def _foreign_handoffs(self, path, known) -> list[dict[str, Any]]:
        """On-disk worker handoffs this registry never tracked whose log still decides an outcome:
        the process is alive (same start time) or its recorded exit is waiting to be read."""
        from pathlib import Path
        from tools.process_registry_output_log import _exit_path, adoptable_log
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return []
        return [entry for entry in on_disk if isinstance(entry, dict) and entry.get("session_id") not in known
                and (log := adoptable_log(entry.get("output_log"))) and Path(log).exists()
                and (_exit_path(log).exists() or self._detached_host_fate(
                    entry.get("pid"), entry.get("host_start_time")) == "running")]

    def recover_from_checkpoint(self) -> int:
        """On gateway startup, probe PIDs from the checkpoint file; returns how many
        were recovered as detached sessions.

        Dead entries of completion-contract holders (``notify_on_complete`` /
        ``persist_on_release``) are not just dropped: the process died while the
        gateway was down (or was swept by a shutdown that never delivered), and its
        owner must learn it — a ``lost`` / ``gateway_downtime`` receipt with
        ``interrupt_notice_state="pending"`` is written and replayed to the owner by
        ``arm_interrupt_notices`` (death perceivable even when it cannot be kept
        alive)."""
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS, _checkpoint_path,
            _CHECKPOINT_DEFAULTS, _WATCHER_ROUTE_KEYS, _stop_systemd_unit,
        )
        from tools.process_registry_output_log import adoptable_log
        from tools.process_registry_results import save_completed_result

        # This host adopts what a crashed predecessor left running, so the jobs it starts must
        # survive its own crash for the next one: they stay out of the owner-death supervisor.
        self._jobs_outlive_host = True
        self.arm_interrupt_notices()
        checkpoint_path = _checkpoint_path()
        if not checkpoint_path.exists():
            return 0
        try:
            entries = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return 0
        recovered = 0
        unresolved_scope_entries: list[dict[str, Any]] = []
        for entry in entries:
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
            # Alive and still ours: re-attach. A start-time probe that cannot
            # be read is not proof the PID was reused — dropping it would leave
            # a live child unsupervised, and marking it exited would invent a
            # completion. A positive mismatch means the number was recycled:
            # do not adopt it and do not signal it.
            fate = self._detached_host_fate(pid, entry.get("host_start_time"))
            # A worker spawn's output log and recorded exit outlive both the spawner and the PID,
            # so its reader decides the outcome (a recycled PID is never signalled: fate != running).
            log = adoptable_log(entry.get("output_log"))
            if log:
                fate = "running"
            if fate == "reused":
                logger.info(
                    "Not recovering session %s: pid %d is alive but its "
                    "start time no longer matches — PID was recycled onto "
                    "an unrelated process; refusing to adopt or signal it.",
                    entry.get("session_id", "?"), pid)
                continue
            if fate != "running":
                systemd_unit = entry.get("systemd_unit", "")
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    logger.warning(
                        "Could not reap persisted scope %s for dead wrapper pid %s; "
                        "retaining checkpoint entry for the next startup",
                        systemd_unit, pid)
                    unresolved_scope_entries.append(entry)
                    continue
                self._record_downtime_death(entry, ProcessSession, save_completed_result)
                continue
            fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            fields.update(
                command=entry.get("command", "unknown"),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                started_at=entry.get("started_at", time.time()), sandbox_key=entry.get("sandbox_key", ""))
            # detached: no Popen handle, but can report status + kill; a worker spawn's output log
            # (and recorded exit) outlives its spawner, so tail it from the start.
            fields["output_log"] = log
            session = ProcessSession(id=entry["session_id"], detached=True, **fields)
            if session.output_log:
                self._track_started(session, self._log_reader_loop, f"proc-log-reader-{session.id}")
            else:
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
        self._write_checkpoint(extra_entries=unresolved_scope_entries)
        # Downtime deaths recorded above become pending notices; arm them too (idempotent).
        self.arm_interrupt_notices(rescan=True)
        return recovered
