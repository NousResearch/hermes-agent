"""Completion consumption shared across registry processes in the same profile home."""

import logging
import string
import time
from pathlib import Path

from hermes_constants import get_hermes_home, mkdir_under_hermes_home

logger = logging.getLogger("tools.process_registry")
CONSUMED_MARKER_TTL_SECONDS = 7 * 86400  # Longer than the finished-session cache.
_CONSUMED_ID_ALLOWED = frozenset(string.ascii_letters + string.digits + "_.-")


def _consumed_marker_path(session_id: str) -> Path | None:
    # Validate before resolving the home or touching the filesystem.
    if (not isinstance(session_id, str) or not 1 <= len(session_id) <= 128
            or session_id in {".", ".."} or any(c not in _CONSUMED_ID_ALLOWED for c in session_id)):
        return None
    return get_hermes_home() / "processes-consumed" / f"{session_id}.consumed"


def _sweep_consumed_markers(now: float | None = None) -> None:
    """Best-effort expiry, independent of the shorter finished-session TTL."""
    cutoff = (time.time() if now is None else now) - CONSUMED_MARKER_TTL_SECONDS
    try:
        for path in (get_hermes_home() / "processes-consumed").glob("*.consumed"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not expire consumed marker", exc_info=True)
    except OSError:
        logger.debug("Could not sweep consumed markers", exc_info=True)


class ProcessConsumptionMixin:
    _completion_consumed: set[str]
    _poll_observed: set[str]

    def _mark_completion_consumed(self, session_id: str) -> None:
        """Keep the in-memory fast path even when durable storage is unavailable."""
        self._completion_consumed.add(session_id)
        try:
            path = _consumed_marker_path(session_id)
            if path is not None:
                mkdir_under_hermes_home(path.parent)
                # Only existence and mtime matter; refresh without truncating contents.
                path.touch(exist_ok=True)
        except OSError:
            logger.debug("Could not persist consumed marker", exc_info=True)

    def is_completion_consumed(self, session_id: str) -> bool:
        """A fresh registry sees consumption by another process in the active profile."""
        if session_id in self._completion_consumed:
            return True
        try:
            path = _consumed_marker_path(session_id)
            return path is not None and path.is_file()
        except OSError:
            logger.debug("Could not read consumed marker", exc_info=True)
            return False

    def _drain_should_skip(self, session_id: str, *, skip_poll_observed: bool = True) -> bool:
        """CLI inline dedup includes poll; gateway/TUI consumption deliberately does not."""
        return (self.is_completion_consumed(session_id)
                or (skip_poll_observed and session_id in self._poll_observed))
