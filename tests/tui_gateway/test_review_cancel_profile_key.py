"""Session teardown and interrupt fence the session's OWN review, under its profile's key.

``_finalize_session`` (close, idle/orphan reap, shutdown) and ``_interrupt_session_turn``
(``session.interrupt``, the orphan reaper's interrupt-at-grace) cancel a session's in-flight
background review (#102895). Both run on unscoped reaper, Timer and RPC-pool threads, whose
ambient home is the launch profile's, while a hosted secondary session's review is published
under that session's profile: ``agent/review_admission.py`` buckets the registry by
``(profile, session)`` because another profile may hold a review under the same session id.
The cancel names the session's profile explicitly, so it never fences the launch profile's
review and leaves the session's own running.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from agent import review_admission
from agent.background_review import prepare_background_review_run
from hermes_constants import hermes_home_key
from tui_gateway import server

SESSION_ID = "shared-session"


@pytest.fixture(autouse=True)
def _clean_registry():
    with review_admission._lock:
        review_admission._review_runs.clear()
    yield
    with review_admission._lock:
        review_admission._review_runs.clear()


def _agent():
    agent = MagicMock()
    agent._persist_session = MagicMock()
    agent.commit_memory_session = MagicMock()
    agent.session_id = SESSION_ID
    agent.model = "test-model"
    agent.platform = "desktop"
    agent._session_messages = None
    agent._background_review_lock = threading.Lock()
    agent._background_review_agent = None
    agent._background_review_run = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    return agent


def _secondary_session(tmp_path, *, running=False):
    """A session served from a secondary profile home, plus two reviews on its session id:
    its own (under its profile) and the launch profile's (under the ambient key)."""
    secondary_home = tmp_path / "secondary-profile"
    secondary_home.mkdir()
    own_key, launch_key = hermes_home_key(secondary_home), hermes_home_key()
    assert own_key != launch_key
    agent, launch_agent = _agent(), _agent()
    own_run = prepare_background_review_run(
        agent, session_id=SESSION_ID, profile_key=own_key
    )
    launch_run = prepare_background_review_run(
        launch_agent, session_id=SESSION_ID, profile_key=launch_key
    )
    assert own_run is not None and launch_run is not None
    session = {
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "running": running,
        "queued_prompt": None,
        "session_key": "secondary-session-key",
        "_run_thread": None,
        "_finalized": False,
        "profile_home": str(secondary_home),
    }
    return session, own_run, launch_run


def test_finalize_fences_the_sessions_own_review_not_the_launch_profiles(tmp_path):
    session, own_run, launch_run = _secondary_session(tmp_path)

    server._finalize_session(session, end_reason="tui_close")

    assert own_run.cancel_requested.is_set(), (
        "the session's own review was left running"
    )
    assert not launch_run.cancel_requested.is_set(), (
        "the launch profile's review was fenced by another profile's teardown"
    )


def test_interrupt_fences_the_sessions_own_review_not_the_launch_profiles(
    tmp_path, monkeypatch
):
    session, own_run, launch_run = _secondary_session(tmp_path)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)

    server._interrupt_session_turn("ui-session", session)

    assert own_run.cancel_requested.is_set(), (
        "the session's own review was left running"
    )
    assert not launch_run.cancel_requested.is_set(), (
        "the launch profile's review was fenced by another profile's /stop"
    )
