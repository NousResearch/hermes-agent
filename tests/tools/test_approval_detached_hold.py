"""The approval timeout is an answer budget: it only runs down while a client can see the prompt.

A client that drops its socket (a phone app backgrounded mid-turn) reconnects and gets the prompt back
through ``open_requests``. If the countdown kept running while it was gone, it reconnects to a prompt
that already auto-denied, and the agent is told the user refused a command the user was never shown.
``approvals.detached_timeout`` still ends a prompt whose client never comes back.

The loop is driven on a fake clock (the event's wait advances it), so the tests assert on when the
wait ended rather than sleeping.
"""

import pytest

from tools import approval as mod
from tools import approval_gateway_wait as wait_mod

SESSION_KEY = "approval-detached-hold"


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


class _NeverAnswered:
    """Stands in for the entry's threading.Event: nobody answers, each slice advances the clock."""

    def __init__(self, clock):
        self.clock = clock

    def wait(self, timeout=None):
        self.clock.now += timeout
        return False


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(wait_mod, "time", clock)
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 5)
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_detached_timeout", lambda: 60)
    yield clock
    mod.unregister_gateway_notify(SESSION_KEY)


def _wait_seconds(clock) -> float:
    started = clock.now
    assert wait_mod._poll_event(_NeverAnswered(clock), SESSION_KEY, interrupt_log="%s") == "timeout"
    return clock.now - started


@pytest.mark.parametrize(
    ("attached_after", "expected"),
    [
        (None, 5),    # surface without a presence probe: plain wall-clock timeout
        (0, 5),       # client attached throughout: the budget runs normally
        (10, 15),     # detached for 10s: held, then the full 5s budget once the client is back
        (58, 63),     # reattached just before the ceiling: still gets its whole budget
        (1e9, 60),    # never comes back: the detached ceiling ends the prompt
    ],
)
def test_timeout_runs_only_while_a_client_is_attached(clock, attached_after, expected):
    started = clock.now
    presence = None if attached_after is None else (lambda: clock.now - started >= attached_after)
    mod.register_gateway_notify(SESSION_KEY, lambda data: None, presence=presence)

    assert _wait_seconds(clock) == pytest.approx(expected)


def test_presence_is_dropped_with_the_notify_callback_and_fails_open(clock):
    """A probe must not outlive its registration, and a probe that raises degrades to the plain timeout."""
    mod.register_gateway_notify(SESSION_KEY, lambda data: None, presence=lambda: False)
    mod.unregister_gateway_notify(SESSION_KEY)
    assert _wait_seconds(clock) == pytest.approx(5)

    def broken():
        raise RuntimeError("probe failed")

    mod.register_gateway_notify(SESSION_KEY, lambda data: None, presence=broken)
    assert _wait_seconds(clock) == pytest.approx(5)
