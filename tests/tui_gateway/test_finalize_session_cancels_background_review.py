"""Regression for #102895: deleting/closing a session must stop its
in-flight background memory/skill review fork.

The reported symptom: a desktop session's post-turn skill-review fork
degraded into a failed-tool-call retry loop (malformed tool arguments from
a fallback local model). The user deleted the session in the UI, but the
review fork kept calling the model for 70+ minutes — only a full app
restart stopped it.

Root cause: the review fork is a SEPARATE ``AIAgent`` instance tracked only
on the parent's ``_background_review_agent`` / ``_active_children``
(``agent/background_review.py``). It is never registered on
``session["running"]`` or ``session["_run_thread"]``, so it was invisible
to both:
  - ``_teardown_popped_session``'s run-thread join, and
  - ``session.interrupt``'s ``_interrupt_session_turn`` (gated on
    ``session.get("running")``).

Every session-teardown path (explicit close, idle-timeout reap, ws-orphan
reap, shutdown) funnels through ``_finalize_session`` — the single
``_finalized``-guarded chokepoint (see its own docstring) — so that is
where the missing cancellation belongs. The fix calls the existing
``agent.background_review.cancel_background_review_for_live_turn`` helper
(previously only invoked when a NEW live turn preempts a running review)
from ``_finalize_session`` as well, so "this session is ending" cancels an
in-flight review exactly like "a new turn arrived" already did.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from agent.background_review import _BackgroundReviewRun
from tui_gateway.server import _finalize_session


class _UnpublishedExit(threading.Event):
    """The fork's request exit, never published: the fork is parked in a non-interruptible
    tool call. Every wait is recorded and returns at once so a regression shows up as a
    recorded wait instead of a hung test."""

    def __init__(self) -> None:
        super().__init__()
        self.waits: list = []

    def wait(self, timeout=None):  # Event.wait's shape
        self.waits.append(timeout)
        return False


def _make_session(agent, session_key="test_key_bg_review"):
    return {
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": session_key,
        "_finalized": False,
    }


def _agent_with_live_review_fork():
    """A parent agent with a background-review fork that is still running.

    Mirrors the real shape installed by ``agent/agent_init.py`` and
    populated by ``agent/background_review.py``'s
    ``_run_review_in_thread`` while a review is in flight.
    """
    agent = MagicMock()
    agent._persist_session = MagicMock()
    agent.commit_memory_session = MagicMock()
    agent.session_id = "parent-session"
    agent.model = "test-model"
    agent.platform = "desktop"
    agent._session_messages = None

    review_fork = MagicMock()
    # hard_interrupt is looked up via inspect.getattr_static in
    # agent.interrupt_compat.request_hard_interrupt — a real callable
    # attribute (not an auto-speccing MagicMock surprise) is required.
    review_fork.hard_interrupt = MagicMock()

    lock = threading.Lock()
    agent._background_review_lock = lock
    agent._background_review_agent = review_fork
    agent._background_review_run = None  # legacy-pointer path
    agent._active_children = [review_fork]
    agent._active_children_lock = threading.Lock()

    return agent, review_fork


def _agent_with_admitted_review_run():
    """A parent whose review fork was admitted through the run handshake (the shape
    ``_run_review_in_thread`` installs) and has not published its request exit."""
    agent, review_fork = _agent_with_live_review_fork()
    agent._background_review_agent = None
    run = _BackgroundReviewRun()
    assert run.begin_request(review_fork) is True
    run.request_done = _UnpublishedExit()
    agent._background_review_run = run
    return agent, review_fork, run


class TestFinalizeSessionCancelsBackgroundReview:
    def test_finalize_does_not_wait_for_the_forks_request_exit(self):
        """Teardown is not a live turn: nothing overlaps the fork once the session is gone,
        and the next turn on the same session performs its own cancel + wait (facade
        registry, gateway prework, durable lease). So finalize passes the cancel through
        WITHOUT the live-turn acknowledgement wait: that wait escalates for 20s and then
        never returns for a fork parked in a non-interruptible tool call, stalling the
        reaper/Timer/atexit thread that called finalize — one wedged fork would hold an
        idle-reap sweep or the SIGTERM settle budget. Upstream's #102895 call site was
        written against a bounded warn-and-proceed wait."""
        agent, review_fork, run = _agent_with_admitted_review_run()
        session = _make_session(agent)

        _finalize_session(session, end_reason="idle_timeout")

        assert run.request_done.waits == [], (
            "finalize parked on the review fork's request exit"
        )
        assert run.cancel_requested.is_set(), "the cancel fence must still land"

        for _ in range(50):
            if review_fork.hard_interrupt.called:
                break
            threading.Event().wait(0.05)

        assert review_fork.hard_interrupt.called
        args, kwargs = review_fork.hard_interrupt.call_args
        assert args and "idle_timeout" in args[0]
        assert kwargs.get("tool_reason") == "session ended"

    def test_finalize_interrupts_live_review_fork(self):
        """_finalize_session must hard-interrupt a running review fork."""
        agent, review_fork = _agent_with_live_review_fork()
        session = _make_session(agent)

        _finalize_session(session, end_reason="tui_close")

        # cancel_background_review_for_live_turn dispatches the actual
        # interrupt() call on a short-lived daemon thread (#84423) so a
        # wedged abort hook cannot stall the caller — wait for it instead
        # of asserting synchronously.
        for _ in range(50):
            if review_fork.hard_interrupt.called:
                break
            threading.Event().wait(0.05)

        assert review_fork.hard_interrupt.called, (
            "session finalize must interrupt the in-flight background "
            "review fork (#102895): the failed-tool retry loop would "
            "otherwise keep calling the model after the session is gone"
        )
        _, kwargs = review_fork.hard_interrupt.call_args
        # tool_reason must describe session teardown, not the unrelated
        # (and here false) "superseded by a new live turn" default wording.
        assert kwargs.get("tool_reason") == "session ended"

    def test_finalize_reports_the_real_end_reason(self):
        """The interrupt message names the actual end_reason, not the
        generic live-turn-preemption wording, so logs/diagnostics are
        truthful about why the review stopped."""
        agent, review_fork = _agent_with_live_review_fork()
        session = _make_session(agent)

        _finalize_session(session, end_reason="idle_timeout")

        for _ in range(50):
            if review_fork.hard_interrupt.called:
                break
            threading.Event().wait(0.05)

        assert review_fork.hard_interrupt.called
        args, _kwargs = review_fork.hard_interrupt.call_args
        assert args and "idle_timeout" in args[0]

    def test_finalize_without_a_review_fork_is_a_no_op(self):
        """No live review → cancellation path must not raise or interfere
        with the ordinary finalize flow (most sessions have no review
        running at teardown time)."""
        agent = MagicMock()
        agent._persist_session = MagicMock()
        agent.commit_memory_session = MagicMock()
        agent.session_id = "parent-session-2"
        agent.model = "test-model"
        agent.platform = "desktop"
        agent._session_messages = None
        agent._background_review_lock = threading.Lock()
        agent._background_review_agent = None
        agent._background_review_run = None
        agent._active_children = []
        agent._active_children_lock = threading.Lock()

        session = _make_session(agent, session_key="no-review-session")

        # Must not raise.
        _finalize_session(session, end_reason="tui_close")

        agent.commit_memory_session.assert_not_called()  # empty history

    def test_finalize_tolerates_agent_missing_review_attributes(self):
        """A minimal/legacy agent object with no background-review
        attributes at all must not break finalize (defensive getattr)."""

        class BareAgent:
            session_id = "bare-agent-session"
            model = "test-model"
            platform = "tui"
            _session_messages = None

        session = _make_session(BareAgent(), session_key="bare-agent")

        # Must not raise despite BareAgent having no
        # _background_review_agent/_background_review_lock attributes.
        _finalize_session(session, end_reason="tui_close")
