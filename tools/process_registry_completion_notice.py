"""Completion-notice consumption and retirement for ProcessRegistry.

Split out of :mod:`tools.process_registry` (FILE_LINES ratchet): the notice-redundancy
verdicts are a self-contained concern. The origin module keeps the registries themselves;
these methods read ``_completion_consumed`` / ``_notice_retired`` / ``_poll_observed`` and
the live tables through ``self`` on the mixed-in class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession


class ProcessCompletionNoticeMixin:
    """Completion-notice dedup gates for ProcessRegistry (consumed via wait/log, retired by
    an exited poll(), or superseded by a re-run twin). See ProcessRegistry."""

    def _superseded_by(self, session: ProcessSession) -> Optional[str]:
        """Id of the newest twin — same session_key + command, later started_at — so a
        completion notice for a process the lane already re-ran says SUPERSEDED instead of
        burning a turn re-explaining it.
        ponytail: scans live tables only — a twin evicted past FINISHED_TTL reads as not
        superseded (under-report, never a false positive)."""
        if not session.session_key:
            return None
        with self._lock:
            twins = [s for s in (*self._running.values(), *self._finished.values())
                     if s.session_key == session.session_key and s.command == session.command
                     and s.started_at > session.started_at]
        return max(twins, key=lambda s: s.started_at).id if twins else None


    def is_completion_consumed(self, session_id: str) -> bool:
        """True once wait/log/kill/acked poll consumed it; keyed by the full id, never a short prefix."""
        return session_id in self._completion_consumed


    def is_completion_notice_retired(self, session_id: str) -> bool:
        """Watcher gate: the completion notice is redundant because the agent already
        holds the receipt — consumed via wait/log, or observed inline via an exited
        poll(). Delivery is at-most-once: an unread completion still delivers once."""
        return session_id in self._completion_consumed or session_id in self._notice_retired


    def retire_completion_notice(self, session_id: str) -> None:
        """Drop the pending completion notice for ``session_id`` without marking the
        output consumed (used when the receipt was reconciled outside wait/log)."""
        self._notice_retired.add(session_id)


    def _drain_should_skip(self, session_id: str, *, skip_poll_observed: bool = True) -> bool:
        """Skip a completion the agent already has — consumed via wait/log or retired
        by an exited poll() (``is_completion_notice_retired``). A poll of a *running*
        session retires nothing, so a later exit still delivers the watcher's turn.

        ``skip_poll_observed`` additionally drops exits merely observed via poll()
        (``_poll_observed``) for the CLI's same-turn dedup (#8228); gateway/tui watchers
        pass False and rely on the retirement gate.
        """
        return self.is_completion_notice_retired(session_id) or (
            skip_poll_observed and session_id in self._poll_observed)
