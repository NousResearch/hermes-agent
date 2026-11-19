"""Session and task liveness queries for the process registry."""

import time
from typing import Optional


class ProcessLivenessMixin:
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
        """Whether any process for ``task_id`` is still running. Ownership is
        ``owner_task_id`` (the raw spawning id) like the other task-scoped queries:
        ``task_id`` on a session is the collapsed container key, shared across
        turns and delegate children, so a container-key match alone would miss a
        delegate child's own background work (#120546)."""
        return self._any_running(lambda s: s.owner_task_id == task_id)

    def has_active_for_session(self, session_key: str, max_active_age: Optional[float] = None) -> bool:
        """Active processes for a gateway session key. Processes older than
        ``max_active_age`` seconds are ignored as stale so a forgotten ``http.server``
        can't freeze session idle/daily reset forever; ``None`` keeps legacy behaviour
        (any running process blocks)."""
        now = time.time()
        return self._any_running(
            lambda s: s.session_key == session_key
            and (max_active_age is None or (now - s.started_at) < max_active_age))

    def has_completion_work_for_session(self, session_key: str) -> bool:
        """Reader-observed bounded work, without I/O, for gateway typing ticks.

        Silent servers/watchers have no completion contract and must not keep a
        conversation typing forever. Readers/watchers own exit reconciliation;
        unlike list_sessions this query never probes a host or sandbox.
        """
        with self._lock:
            return any(s.session_key == session_key and s.notify_on_complete and not s.exited
                       for s in self._running.values())

    def has_any_active(self) -> bool:
        """Whether ANY background process is running — scale-to-zero must not
        suspend a gateway with live background work or the process is lost."""
        return self._any_running(lambda s: True)
