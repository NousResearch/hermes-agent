"""Regression for #102895 (sibling path): an explicit /stop must also
cancel a running background memory/skill review, not only the foreground
turn.

``_interrupt_session_turn`` (backing the ``session.interrupt`` RPC and the
WS-orphan reaper's interrupt-at-grace) gates its
``request_hard_interrupt(session.get("agent"))`` call on
``should_interrupt = bool(session.get("running"))``. The background-review
fork is a SEPARATE ``AIAgent`` (tracked only on the parent's
``_background_review_agent`` / ``_active_children`` —
``agent/background_review.py``) that is never reflected in
``session["running"]``. The common real-world shape is: the foreground turn
already finished (``running`` is ``False``), the post-turn review is still
running, and the user hits Stop expecting everything to stop — but the
`running`-gated branch never fires, so nothing ever interrupts the review.

See ``tests/tui_gateway/test_finalize_session_cancels_background_review.py``
for the session-teardown (delete/close/reap) counterpart of this same root
cause.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def server():
    from unittest.mock import patch
    import importlib

    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
    yield mod


def _session_with_finished_turn_and_live_review(review_fork):
    """A session whose foreground turn already ended but whose background
    review is still running — the exact shape from the #102895 log
    (finish_reason=stop at 20:03:09, review starts same timestamp)."""
    agent = MagicMock()
    agent._background_review_lock = threading.Lock()
    agent._background_review_agent = review_fork
    agent._background_review_run = None
    agent._active_children = [review_fork]
    agent._active_children_lock = threading.Lock()

    return {
        "agent": agent,
        "history_lock": threading.Lock(),
        "running": False,  # foreground turn already finished
        "queued_prompt": None,
        "session_key": "session-key-review-only",
        "_run_thread": None,
    }


def test_stop_interrupts_a_review_running_after_the_turn_finished(server, monkeypatch):
    """/stop with running=False must still reach a live background review."""
    review_fork = MagicMock()
    review_fork.hard_interrupt = MagicMock()
    session = _session_with_finished_turn_and_live_review(review_fork)

    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)

    response = server._methods["session.interrupt"](
        "stop", {"session_id": "ui-session"}
    )

    assert response["result"]["status"] == "interrupted"

    for _ in range(50):
        if review_fork.hard_interrupt.called:
            break
        threading.Event().wait(0.05)

    assert review_fork.hard_interrupt.called, (
        "session.interrupt with running=False must still cancel a live "
        "background review fork (#102895 sibling path) — otherwise a "
        "user who hits Stop believing everything stopped is left with a "
        "review silently retrying against the model forever"
    )
    _, kwargs = review_fork.hard_interrupt.call_args
    assert kwargs.get("tool_reason") == "session interrupted"


class _UnpublishedExit(threading.Event):
    """The fork's request exit, never published: the fork is parked in a non-interruptible
    tool call. Every wait is recorded and returns at once so a regression shows up as a
    recorded wait instead of a hung RPC."""

    def __init__(self) -> None:
        super().__init__()
        self.waits: list = []

    def wait(self, timeout=None):  # noqa: D102 — Event.wait's shape
        self.waits.append(timeout)
        return False


def test_stop_does_not_wait_for_the_forks_request_exit(server, monkeypatch):
    """``session.interrupt`` runs on a tui-rpc pool worker (also: the WS-orphan reaper, the
    lease-takeover path and the SIGTERM exit path, all through ``_interrupt_session_turn``).
    No live turn starts there — the next turn on the chat performs its own cancel + wait
    through the facade registry and the durable lease — so the cancel is passed through
    WITHOUT the live-turn acknowledgement wait, which escalates for 20s and then never
    returns for a fork parked in a non-interruptible tool call. A user hitting Stop while
    only the post-turn review is running must not see the RPC block on it."""
    from agent.background_review import _BackgroundReviewRun

    review_fork = MagicMock()
    review_fork.hard_interrupt = MagicMock()
    session = _session_with_finished_turn_and_live_review(review_fork)
    agent = session["agent"]
    agent.session_id = "parent-session"
    agent._background_review_agent = None
    run = _BackgroundReviewRun()
    assert run.begin_request(review_fork) is True
    run.request_done = _UnpublishedExit()
    agent._background_review_run = run

    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)

    response = server._methods["session.interrupt"](
        "stop", {"session_id": "ui-session"}
    )

    assert response["result"]["status"] == "interrupted"
    assert run.request_done.waits == [], (
        "session.interrupt parked on the review fork's request exit"
    )
    assert run.cancel_requested.is_set(), "the cancel fence must still land"

    for _ in range(50):
        if review_fork.hard_interrupt.called:
            break
        threading.Event().wait(0.05)

    assert review_fork.hard_interrupt.called
    _, kwargs = review_fork.hard_interrupt.call_args
    assert kwargs.get("tool_reason") == "session interrupted"


def test_stop_without_a_review_is_unaffected(server, monkeypatch):
    """No live review → the new cancellation path is a no-op; the ordinary
    interrupt contract (status, hard_interrupt on the foreground agent)
    must be unchanged."""
    agent = MagicMock()
    agent._background_review_lock = threading.Lock()
    agent._background_review_agent = None
    agent._background_review_run = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.hard_interrupt = MagicMock()

    session = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "running": True,
        "queued_prompt": None,
        "session_key": "session-key-normal-stop",
        "_run_thread": None,
    }

    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)

    response = server._methods["session.interrupt"](
        "stop", {"session_id": "ui-session"}
    )

    assert response["result"]["status"] == "interrupted"
    assert agent.hard_interrupt.called
