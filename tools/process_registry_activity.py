"""Session/task activity queries and the bulk-kill sweep over tracked background processes:
whether anything is still running for a task or a gateway session (and how stale it may be
before it stops blocking a reset), the running-ID snapshot that marks a turn boundary, and the
task-scoped termination the shutdown, reaper and ``/stop`` paths call."""

from typing import Optional

import time


class ProcessActivityMixin:
    # ----- Session/Task Queries (for gateway integration) -----

    def _any_running(self, predicate) -> bool:
        """True if any still-running session satisfies *predicate*, after refreshing
        detached sessions so a finished-but-unreaped process reads as inactive."""
        with self._lock:
            sessions = list(self._running.values())
        for session in sessions:
            self._refresh_detached_session(session)
        with self._lock:
            return any(not s.exited and predicate(s) for s in self._running.values())

    def has_active_processes(self, task_id: str) -> bool:
        """Whether any process for ``task_id`` is still running."""
        return self._any_running(lambda s: s.task_id == task_id)

    def has_active_for_session(self, session_key: str, max_active_age: Optional[float] = None) -> bool:
        """Active processes for a gateway session key. Processes older than
        ``max_active_age`` seconds are ignored as stale so a forgotten ``http.server``
        can't freeze session idle/daily reset forever; ``None`` keeps legacy behaviour
        (any running process blocks)."""
        now = time.time()
        return self._any_running(
            lambda s: s.session_key == session_key
            and (max_active_age is None or (now - s.started_at) < max_active_age))

    def has_any_active(self) -> bool:
        """Whether ANY background process is running — scale-to-zero must not
        suspend a gateway with live background work or the process is lost."""
        return self._any_running(lambda s: True)

    def snapshot_running_ids(self, task_id: str) -> frozenset[str]:
        """Running IDs owned by ``task_id`` — a turn-boundary marker: on timeout
        only processes absent from the starting snapshot belong to the abandoned
        turn; older ones intentionally span turns and must survive."""
        with self._lock:
            return frozenset(s.id for s in self._running.values() if s.task_id == task_id and not s.exited)

    def kill_started_since(self, task_id: str, baseline_ids, *, source: str) -> int:
        """Kill ``task_id`` processes created after ``baseline_ids``. Output is
        consumed so an abandoned turn can't enqueue a follow-up reviving work the
        timeout deliberately stopped."""
        return self.kill_all(task_id, exclude_ids=frozenset(baseline_ids or ()), source=source, consume_output=True)

    def kill_all(
        self, task_id: Optional[str] = None, *, exclude_ids: frozenset = frozenset(),
        source: str = "kill_all", consume_output: bool = False) -> int:
        """Kill all running processes, optionally filtered by task_id. Returns count killed."""
        with self._lock:
            targets = [
                s for s in self._running.values()
                if (task_id is None or s.task_id == task_id) and s.id not in exclude_ids and not s.exited
            ]
        return sum(
            self.kill_process(s.id, source=source, consume_output=consume_output).get("status")
            in {"killed", "already_exited"}
            for s in targets)
