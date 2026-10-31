"""Live and retained process enumeration for status and tool queries."""


class ProcessSessionsMixin:
    def count_running(self) -> int:
        """O(1) running count for status-bar polling; dict ``len()`` is atomic, no lock."""
        return len(self._running)

    def list_sessions(self, task_id: str | None = None, session_key: str | None = None, *, include_retained: bool = False) -> list:
        """Running and recently-finished processes for ``task_id`` and/or ``session_key``;
        cross-task entries sharing the gateway session (a forgotten preview server
        blocking session reset) are flagged ``"session_scoped": true``.

        When ``task_id`` is given, processes that task spawned (its ``owner_task_id``) are included. When
        ``session_key`` is also given, session-scoped background processes (``background: true``) registered
        under that gateway session are surfaced too, even if they belong to a different task — so the agent
        can discover a forgotten preview server that is blocking session reset (#29177).
        """
        from tools.process_registry import _WSL_CHAIN_NOTE, load_completed_results, time

        # Only an explicit tool query reads historical receipts. Status bars and
        # gateway liveness scans call this frequently and need the live registry.
        sessions = load_completed_results() if include_retained else {}
        with self._lock:
            sessions.update(self._finished)
            sessions.update(self._running)
        all_sessions = [
            refreshed for refreshed in (
                self._refresh_detached_session(s) for s in sessions.values()
            ) if refreshed is not None and not self._uncollected_gone(refreshed)
        ]
        if task_id or session_key:
            all_sessions = [
                s for s in all_sessions
                if (task_id and s.owner_task_id == task_id)
                or (session_key and s.session_key == session_key)
            ]
        result = []
        for s in all_sessions:
            # List-only refreshes must observe child exit even while descendants
            # keep the capture pipe open; retain the existing completion owner.
            self._reconcile_local_exit(s)
            entry = {
                "session_id": s.id,
                "command": s.command[:200],
                "cwd": s.cwd,
                "pid": s.pid,
                "owner_task_id": s.owner_task_id,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)),
                "uptime_seconds": int(time.time() - s.started_at),
                "status": "exited" if s.exited else "running",
                "output_preview": s.output_buffer[-200:] if s.output_buffer else "",
            }
            # Flag processes surfaced only because they share the gateway session (not the current task) —
            # these are the long-lived background processes a user may have forgotten about (#29177).
            if task_id and session_key and s.owner_task_id != task_id and s.session_key == session_key:
                entry["session_scoped"] = True
            if s.wsl_chain:
                entry["wsl_chain"] = True
                entry["wsl_note"] = _WSL_CHAIN_NOTE
            # Trigger metadata for goal-loop judges (a watcher may never exit).
            if s.watch_patterns and not s._watch_disabled:
                entry.update(watch_patterns=list(s.watch_patterns), watch_hit=s._watch_hits > 0)
            if s.notify_on_complete:
                entry["notify_on_complete"] = True
            if s.persist_on_release:
                entry["persist_on_release"] = True
            if s.exited:
                entry["exit_code"] = s.exit_code
                entry["exited_at"] = s.exited_at
                entry["completion_reason"] = s.completion_reason
            if s.detached:
                entry["detached"] = True
            entry["notification_pending"] = self._notification_pending(s)
            result.append(entry)
        return result
