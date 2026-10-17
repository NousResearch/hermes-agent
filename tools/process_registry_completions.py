"""Process completion ownership and surface delivery state."""


class ProcessCompletionMixin:
    def is_completion_consumed(self, session_id: str) -> bool:
        """Check if a completion notification was already consumed via wait/log."""
        return session_id in self._completion_consumed

    def is_completion_delivered(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._completion_delivered

    def mark_completion_delivered(self, session_id: str) -> None:
        """A surface accepted the notice; the process output remains readable."""
        with self._lock:
            self._completion_delivered.add(session_id)

    def has_pending_completions_owned_by(self, owner_task_ids) -> bool:
        """Owned notify work, through exit publication and surface acceptance.

        Raw turn ownership is the boundary, never the shared environment key.
        A status poll does not acknowledge the autonomous completion notice.
        """
        owners = frozenset(owner_task_ids)
        if not owners:
            return False
        with self._lock:
            def pending(s):
                return (s.owner_task_id in owners and s.notify_on_complete
                        and s.id not in self._completion_consumed
                        and s.id not in self._completion_delivered)

            return (any(pending(s) for s in self._running.values())
                    or any(pending(s) and s.exit_code is not None
                           for s in self._finished.values()))

    def unread_completions_owned_by(self, owner_task_id: str) -> list:
        """Exited notify processes whose owner has not read their result."""
        with self._lock:
            return [s for s in self._finished.values()
                    if s.owner_task_id == owner_task_id and s.notify_on_complete
                    and s.exit_code is not None
                    and s.id not in self._completion_consumed and s.id not in self._poll_observed]
