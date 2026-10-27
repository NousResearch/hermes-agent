"""Running-process checkpoint persistence and PID-safe recovery."""

import json
import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.process_registry")


class ProcessCheckpointMixin:
    def _detached_host_fate(self, pid: Optional[int], expected_start: Optional[int]) -> str:
        """Subclass supplies the live PID decision. See ProcessRegistry."""
        raise NotImplementedError

    def _refresh_detached_session(self, session):
        """Re-attach, close, or prune a recovered host-PID session.

        A completion is not queued here: recovery has no waitable handle, so it
        never collected an exit status.
        """
        if session is None or session.exited or not session.detached or session.pid_scope != "host" or session.output_log:
            return session  # a log-backed adoption's reader collects the recorded exit itself
        fate = self._detached_host_fate(session.pid, session.host_start_time)
        if fate == "running":
            return session
        if not session.systemd_unit:
            from tools.process_registry_job import finish_detached_scope
            return finish_detached_scope(self, session)
        if fate == "gone":
            return self._prune_uncollected_detached(session)
        self._close_reused_detached(session)
        return session

    def _stop_scope(self, session) -> bool:
        """Stop the recorded native scope and persist its verdict, even after its wrapper exits."""
        from tools.process_registry_job import job_quiescent, record_scope_quiescence
        if not session.systemd_unit:
            stopped = job_quiescent(session)
            if stopped and session.exited:
                from tools.process_registry import save_completed_result
                stopped = ((session._reader_thread is None or session._completion_event.wait(1.0))
                           and save_completed_result(session))
            session._scope_stop_pending = not stopped
            self._write_checkpoint()
            return stopped
        from tools.process_registry import _stop_systemd_unit
        stopped = _stop_systemd_unit(session.systemd_unit)
        session._scope_stop_pending = not stopped
        if stopped:
            try:
                record_scope_quiescence(session)
                self._write_checkpoint()  # Publish capability before a concurrent old result writer.
                if session.exited and not self._reader_finalizing(session):
                    from tools.process_registry_results import save_completed_result
                    if not save_completed_result(session):
                        raise OSError('native scope result is not durable')
            except (OSError, ValueError):
                session._scope_stop_pending = True
                self._write_checkpoint()
                return False
            session.systemd_unit = ''  # The active binding is discharged only by affirmative Stop.
        self._write_checkpoint()
        return stopped

    def _refuse_backend_stop(self, session, consume_output):
        """The opaque wrapper API supplies no birth-bound job/tree cleanup proof."""
        from tools.process_registry import _completion_output
        from tools.process_registry_results import save_completed_result
        with session._lock:
            session._scope_stop_pending = True
            if consume_output:
                session.completion_consumed = True
                self._completion_consumed.add(session.id)
            result = {'status': 'error', 'error': 'process_stop_unknown',
                      'session_id': session.id, 'process_running': not session.exited,
                      **_completion_output(session)}
            if session.systemd_unit:
                result['scope_stop_failed'] = session.systemd_unit
        self._write_checkpoint()
        if session.exited:
            save_completed_result(session)
        return result

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(self, extra_entries: Optional[list[dict[str, Any]]] = None, *, exclude_ids=(), retain_ids=(), _lock_held=False):
        """Write running process metadata to the checkpoint file atomically."""
        from tools.process_registry import _checkpoint_path, _CHECKPOINT_FIELDS
        from tools.process_registry_job import job_quiescent, inherit_job_scope

        try:
            with nullcontext() if _lock_held else self._lock:
                entries = []
                retired = []
                retiring = set()
                persisted = {}
                for s in {**self._finished, **self._running}.values():
                    persisted[s.id] = s._result_persisted_generation
                    # Spawn captures identity from its fresh native handle.
                    # A later writer cannot bind an old PID to today's process.
                    entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
                    # Redact inline credentials before persisting (~/.hermes/processes.json).
                    # Recovery uses command only for display (adoption re-validates the
                    # PID, never re-runs it), so masking is lossless.
                    # See #77484.
                    entry["command"] = redact_sensitive_text(s.command, code_file=True)
                    entry['handoff_note'] = redact_sensitive_text(s.handoff_note, code_file=True)
                    entry["owner_task_id"] = s.owner_task_id or s.task_id
                    if (s.exited and s.pid_scope == 'host' and not s._scope_stop_pending
                            and not s.systemd_unit and not self._reader_finalizing(s) and s.id not in retain_ids):
                        retired.append(entry)
                    else:
                        entries.append(entry)
                if extra_entries:
                    unresolved = {item.get('session_id'): item for item in extra_entries}
                    # Recovery's unresolved native resource beats a completed
                    # output snapshot, which carries no scope authority.
                    entries = [unresolved.get(item.get('session_id'), item)
                               if item.get('session_id') not in self._running else item for item in entries]
                    tracked_ids = {item.get("session_id") for item in entries}
                    entries.extend(item for item in extra_entries if item.get("session_id") not in tracked_ids)
                known = {*self._running, *self._finished, *exclude_ids}
                known.update(item.get('session_id') for item in entries)
            from hermes_cli.active_sessions import _FileLock
            from utils import atomic_json_write
            path = _checkpoint_path()
            # Every interpreter of the profile rewrites this file from its own table. Keep
            # untracked resource evidence under the same file lock. An empty local
            # table is not a cleanup verdict for another interpreter's processes.
            with _FileLock(path.with_name(path.name + ".lock")):
                from tools.process_registry_handoff import merge_ownership, OWNERSHIP_FIELDS
                from tools.process_registry_results import completed_ownership
                try:
                    previous = json.loads(path.read_text(encoding='utf-8-sig'))
                except FileNotFoundError:
                    previous = []
                # An exited cache entry is not proof its retained result inherited
                # a concurrent handoff. Merge before retirement, and keep the
                # checkpoint until that exact native result carries its authority.
                present = {entry['session_id'] for entry in entries}
                receipts = completed_ownership([*entries, *retired])
                for entry in retired:
                    if entry['session_id'] not in present:
                        entries.append(entry)
                        retiring.add(entry['session_id'])
                        present.add(entry['session_id'])
                # A cached completed result can predate the recorded backend
                # scope. Reinserting it while output commits cannot downgrade
                # that resource to an ordinary host PID.
                opaque = {row['session_id']: row for row in previous
                          if row.get('pid_scope', 'host') != 'host' or row.get('wsl_chain')}
                for entry in entries:
                    if entry['session_id'] in retiring and (old := opaque.get(entry['session_id'])):
                        entry['pid_scope'], entry['wsl_chain'] = old.get('pid_scope', 'host'), old.get('wsl_chain', False)
                updates = merge_ownership(entries, previous)
                updates.update(merge_ownership(entries, receipts))
                capabilities = inherit_job_scope(entries, [*previous, *receipts])
                retained = {entry['id']: entry for entry in receipts}
                entries = [entry for entry in entries if entry['session_id'] not in retiring
                           or not job_quiescent(entry) or not (
                               persisted.get(entry['session_id'], -1) == entry.get('ownership_generation', 0) or (
                               (receipt := retained.get(entry['session_id']))
                               and receipt.get('result_complete') is True
                               and all(entry.get(field) == receipt.get(field)
                                       for field in (*OWNERSHIP_FIELDS, 'pid', 'host_start_time'))))]
                # Completed output history cannot discharge an opaque backend
                # resource, including when an older cache omitted its scope.
                present = {entry['session_id'] for entry in entries}
                entries.extend(entry for entry in previous if entry.get('session_id') in known
                               and (entry.get('pid_scope', 'host') != 'host' or not job_quiescent(entry))
                               and entry.get('session_id') not in present and entry.get('session_id') not in exclude_ids)
                known.update(entry.get('session_id') for entry in entries)
                entries.extend(self._foreign_handoffs(path, known))
                atomic_json_write(path, entries)
            # Never acquire the registry lock inside the file lock: an atomic
            # handoff holds them in registry→file order. Refresh stale caches only
            # after releasing the file so their later completion routes correctly.
            with nullcontext() if _lock_held else self._lock:
                for sid in capabilities:
                    session = self._running.get(sid) or self._finished.get(sid)
                    if session is not None:
                        session.native_job_scope = True
                for sid, update in updates.items():
                    session = self._running.get(sid) or self._finished.get(sid)
                    if session is not None and session.ownership_generation <= update['ownership_generation']:
                        for field, value in update.items():
                            setattr(session, field, value)
            return True
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)
            return False

    def _foreign_handoffs(self, path, known) -> list[dict[str, Any]]:
        """Only a writer that tracked/reconciled a resource may retire its evidence.

        In particular, remote PIDs and WSL launcher PIDs cannot establish the
        backend job's fate. Even a host probe is not this unrelated writer's
        authority to discard a record; recovery supplies explicit exclusions.
        """
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return []
        return [entry for entry in on_disk if isinstance(entry, dict) and entry.get("session_id") not in known]

    def recover_from_checkpoint(self, *, owns_entry=None, stop_scopes=False) -> int:
        """On gateway startup, probe PIDs from the checkpoint file; returns how many
        were recovered as detached sessions."""
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS, _checkpoint_path,
            _CHECKPOINT_DEFAULTS, _WATCHER_ROUTE_KEYS, _stop_systemd_unit,
        )
        from tools.process_registry_output_log import adoptable_log
        from tools.process_registry_job import job_quiescent

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
            if owns_entry is not None and not owns_entry(entry):
                unresolved_scope_entries.append(entry)
                continue
            pid, pid_scope = entry.get("pid"), entry.get("pid_scope", "host")
            if (type(pid) is not int or pid <= 0 or pid_scope != "host" or entry.get('wsl_chain')
                    or type(entry.get('host_start_time')) is not int or entry['host_start_time'] <= 0):
                # No backend handle survives interpreter loss. A WSL launcher
                # can exit while its Linux job lives; neither is a host PID
                # recovery target or proof of quiescence. Missing recorded
                # birth identity cannot be reconstructed from the current PID.
                unresolved_scope_entries.append(entry)
                continue
            # The registry is process-global, so every profile's checkpoint carries every live
            # process; a multiplexer recovering several homes must adopt each session once.
            with self._lock:
                already_tracked = entry.get("session_id") in self._running
            if already_tracked:
                continue
            if not entry.get('native_job_scope') and not entry.get('systemd_unit'):
                unresolved_scope_entries.append(entry)
                continue
            scope_done = entry.get('native_job_scope') and not entry.get('systemd_unit') and job_quiescent(entry)
            # Alive and still ours: re-attach. A start-time probe that cannot
            # be read is not proof the PID was reused — dropping it would leave
            # a live child unsupervised, and marking it exited would invent a
            # completion. A positive mismatch means the number was recycled:
            # do not adopt it and do not signal it.
            fate = self._detached_host_fate(pid, entry.get("host_start_time"))
            if scope_done:
                fate = 'running'  # Recover its output/status, without signalling the old PID.
            elif entry.get('native_job_scope') and not entry.get('systemd_unit') and fate != 'running':
                unresolved_scope_entries.append(entry)
                continue
            # A worker spawn's output log and recorded exit outlive both the spawner and the PID,
            # so its reader decides the outcome (a recycled PID is never signalled: fate != running).
            log = adoptable_log(entry.get("output_log"))
            if log:
                fate = "running"
            if fate == "reused" and not entry.get("systemd_unit"):
                logger.info(
                    "Not recovering session %s: pid %d is alive but its "
                    "start time no longer matches — PID was recycled onto "
                    "an unrelated process; refusing to adopt or signal it.",
                    entry.get("session_id", "?"), pid)
                continue
            if fate != "running":
                systemd_unit = entry.get("systemd_unit", "")
                if (systemd_unit and entry.get('persist_on_release')
                        and not entry.get('_scope_stop_pending') and not stop_scopes):
                    # A collected wrapper exit does not mean its native scope is
                    # empty. Ordinary startup preserves intentionally persisted jobs;
                    # an explicit Stop or a prior refused Stop may reap this scope.
                    unresolved_scope_entries.append(entry)
                    continue
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    logger.warning(
                        "Could not reap persisted scope %s for dead wrapper pid %s; "
                        "retaining checkpoint entry for the next startup",
                        systemd_unit, pid)
                    unresolved_scope_entries.append(entry)
                elif systemd_unit:
                    try:
                        from tools.process_registry_job import record_stopped_unit
                        record_stopped_unit(self, entry)
                    except (OSError, ValueError):
                        unresolved_scope_entries.append(entry)
                continue
            fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            fields.update(
                command=entry.get("command", "unknown"),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                started_at=entry.get("started_at", time.time()))
            # detached: no Popen handle, but can report status + kill; a worker spawn's output log
            # (and recorded exit) outlives its spawner, so tail it from the start.
            fields["output_log"] = log
            session = ProcessSession(id=entry["session_id"], detached=True, **fields)
            if session.output_log:
                self._track_started(session, self._log_reader_loop, f"proc-log-reader-{session.id}")
            elif scope_done:
                from tools.process_registry_job import read_job_receipt
                with self._lock:
                    self._running[session.id] = session
                self._finish_exited(session, read_job_receipt(entry)['exit_code'])
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
        # Recovery decided these records (including reaped scopes). The merge must not
        # resurrect them from its old image; explicitly retained neighbors/scopes are above.
        self._write_checkpoint(extra_entries=unresolved_scope_entries,
                               exclude_ids={entry.get('session_id') for entry in entries})
        return recovered
