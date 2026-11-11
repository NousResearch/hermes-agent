"""Watch-pattern rate limiting as the agent sees it: which watch events a sequence of matching
chunks queues, on a controlled wall clock."""

import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_watch import (
    WATCH_GLOBAL_COOLDOWN_SECONDS,
    WATCH_GLOBAL_MAX_PER_WINDOW,
    WATCH_MIN_INTERVAL_SECONDS,
    WATCH_STRIKE_LIMIT,
)


@pytest.fixture()
def clock(monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


def _session(sid="proc_watch"):
    return ProcessSession(id=sid, command="tail -f app.log", task_id="t1", started_at=0.0,
                          watch_patterns=["E"])


def _drain(registry):
    events = []
    while not registry.completion_queue.empty():
        events.append(registry.completion_queue.get_nowait())
    return [(e["type"], e.get("suppressed")) for e in events]


def _struck_window(registry, session, clock, drops=2):
    """One delivered match, then *drops* matches inside its cooldown, then the cooldown expires."""
    registry._check_watch_patterns(session, "E deliver\n")
    for _ in range(drops):
        clock[0] += 1
        registry._check_watch_patterns(session, "E drop\n")
    clock[0] += WATCH_MIN_INTERVAL_SECONDS


def test_consecutive_struck_windows_disable_watching_once(clock):
    registry, session = ProcessRegistry(), _session()
    for _ in range(WATCH_STRIKE_LIMIT - 1):
        _struck_window(registry, session, clock)
    assert _drain(registry) == [("watch_match", 0)] + [("watch_match", 2)] * (WATCH_STRIKE_LIMIT - 2)
    assert session.notify_on_complete is False

    # The window's first drop is the limiting strike; later drops find watching already off.
    _struck_window(registry, session, clock, drops=3)
    assert _drain(registry) == [("watch_match", 2), ("watch_disabled", 1)]
    assert session.notify_on_complete is True

    registry._check_watch_patterns(session, "E after\n")
    assert _drain(registry) == []


def test_a_clean_window_forgives_earlier_strikes(clock):
    registry, session = ProcessRegistry(), _session()
    _struck_window(registry, session, clock)
    registry._check_watch_patterns(session, "E clean window\n")
    clock[0] += WATCH_MIN_INTERVAL_SECONDS
    for _ in range(WATCH_STRIKE_LIMIT - 1):
        _struck_window(registry, session, clock)
    assert all(kind == "watch_match" for kind, _ in _drain(registry))
    assert session.notify_on_complete is False


def test_global_breaker_trips_drops_and_releases(clock):
    registry = ProcessRegistry()
    for i in range(WATCH_GLOBAL_MAX_PER_WINDOW + 2):
        registry._check_watch_patterns(_session(f"proc_{i}"), "E hit\n")
    events = _drain(registry)
    assert events.count(("watch_match", 0)) == WATCH_GLOBAL_MAX_PER_WINDOW
    assert [kind for kind, _ in events].count("watch_overflow_tripped") == 1

    clock[0] += WATCH_GLOBAL_COOLDOWN_SECONDS
    registry._check_watch_patterns(_session("proc_after"), "E hit\n")
    assert _drain(registry) == [("watch_overflow_released", 2), ("watch_match", 0)]
