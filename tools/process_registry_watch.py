"""Watch-pattern rate limiting: the per-session limiter and the process-wide breaker."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# PER SESSION: one watch-match notification per WATCH_MIN_INTERVAL_SECONDS; a match inside the
# cooldown is dropped and counts as one strike per window; WATCH_STRIKE_LIMIT consecutive strike
# windows permanently disable watching and fall back to notify_on_complete semantics.
WATCH_MIN_INTERVAL_SECONDS = 15
WATCH_STRIKE_LIMIT = 3
# Lifetime cap, independent of strikes. A process whose pattern recurs at a cadence just above
# WATCH_MIN_INTERVAL_SECONDS (e.g. a service restarted repeatedly over a day) never trips the
# consecutive-strike limit, since each match lands in its own clean cooldown window, yet still forces a
# full-context agent turn every single time (#93513). watch_patterns is documented as "ONLY for rare
# one-shot mid-process signals", so once a session has delivered this many matches over its whole life
# we disable it and fall back to notify_on_complete, same as the strike-limit path.
WATCH_LIFETIME_MAX_HITS = 8
# Global circuit breaker across all sessions so concurrent siblings can't collectively
# flood the user even when each is under its own cap.
WATCH_GLOBAL_MAX_PER_WINDOW = 15
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30


class WatchPhase(Enum):
    OPEN = "open"  # the current cooldown window has dropped nothing (or no match was delivered yet)
    STRUCK = "struck"  # the current window dropped a match: its one strike is counted
    DISABLED = "disabled"  # strike limit or lifetime cap reached; the session falls back to notify_on_complete


class WatchVerdict(Enum):
    DROP = "drop"
    DISABLE = "disable"  # this drop reached the strike limit
    DELIVER = "deliver"
    DELIVER_LAST = "deliver_last"  # delivered, and the lifetime cap is now reached


@dataclass(slots=True)
class WatchLimiter:
    """One session's watch-match rate limit. The caller holds the session lock."""

    phase: WatchPhase = WatchPhase.OPEN
    cooldown_until: float = 0.0
    strikes: int = 0  # consecutive struck windows
    hits: int = 0  # matches delivered
    suppressed: int = 0  # matches dropped since the last delivery

    def admit(self, now: float, matched: int) -> tuple[WatchVerdict, int]:
        """Rule on a chunk carrying *matched* matching lines; returns the verdict and the dropped
        matches the resulting event reports."""
        if now < self.cooldown_until:
            self.suppressed += matched
            if self.phase is WatchPhase.STRUCK:
                return WatchVerdict.DROP, self.suppressed
            self.strikes += 1
            if self.strikes < WATCH_STRIKE_LIMIT:
                self.phase = WatchPhase.STRUCK
                return WatchVerdict.DROP, self.suppressed
            self.phase = WatchPhase.DISABLED
            return WatchVerdict.DISABLE, self.suppressed
        # A previous window with no drops resets the consecutive-strike counter (healthy cadence again).
        if self.phase is WatchPhase.OPEN:
            self.strikes = 0
        self.cooldown_until = now + WATCH_MIN_INTERVAL_SECONDS
        self.hits += 1
        suppressed, self.suppressed = self.suppressed, 0
        if self.hits >= WATCH_LIFETIME_MAX_HITS:
            self.phase = WatchPhase.DISABLED
            return WatchVerdict.DELIVER_LAST, suppressed
        self.phase = WatchPhase.OPEN
        return WatchVerdict.DELIVER, suppressed


def _breaker_event(type_: str, message: str, **extra) -> dict:
    """Unaddressed (all-sessions) watch breaker event."""
    return {
        "session_id": "", "session_key": "", "command": "", "type": type_, **extra,
        "message": message,
        "platform": "", "chat_id": "", "user_id": "", "user_name": "", "thread_id": "",
    }


class WatchBreaker:
    """Process-wide breaker over delivered watch matches: past WATCH_GLOBAL_MAX_PER_WINDOW in a rolling
    window it trips for WATCH_GLOBAL_COOLDOWN_SECONDS with ONE "tripped" summary, and the cooldown's
    end emits ONE "released" summary."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._window_start = 0.0
        self._window_hits = 0
        self._tripped_until: Optional[float] = None
        self._suppressed_during_trip = 0

    def admit(self, now: float) -> tuple[bool, list[dict]]:
        """Whether this watch_match may pass, and the summary events to queue."""
        events = []
        with self._lock:
            # Handle cooldown expiry first so we can emit the release summary.
            if self._tripped_until is not None and now >= self._tripped_until:
                suppressed = self._suppressed_during_trip
                self._tripped_until = None
                self._suppressed_during_trip = 0
                self._window_start, self._window_hits = now, 0
                if suppressed > 0:
                    events.append(_breaker_event(
                        "watch_overflow_released",
                        f"Watch-pattern notifications resumed. "
                        f"{suppressed} match event(s) were suppressed during the flood.",
                        suppressed=suppressed))
            if self._tripped_until is not None:
                self._suppressed_during_trip += 1
                return False, events
            if now - self._window_start >= WATCH_GLOBAL_WINDOW_SECONDS:
                self._window_start, self._window_hits = now, 0
            if self._window_hits < WATCH_GLOBAL_MAX_PER_WINDOW:
                self._window_hits += 1
                return True, events
            self._tripped_until = now + WATCH_GLOBAL_COOLDOWN_SECONDS
            self._suppressed_during_trip += 1
            events.append(_breaker_event(
                "watch_overflow_tripped",
                f"Watch-pattern overflow: >{WATCH_GLOBAL_MAX_PER_WINDOW} "
                f"notifications in {WATCH_GLOBAL_WINDOW_SECONDS}s across all processes. "
                f"Suppressing further watch_match events for "
                f"{WATCH_GLOBAL_COOLDOWN_SECONDS}s."))
            return False, events
