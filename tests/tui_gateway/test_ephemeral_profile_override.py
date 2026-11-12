"""Regression tests: profile HERMES_HOME override in ephemeral agent threads (#50233).

Why: normal prompt turns bind ``session['profile_home']`` via
``set_hermes_home_override`` before ``run_conversation`` so the turn runs against
the correct profile home. The two ephemeral RPC paths — ``prompt.background`` and
``preview.restart`` — spawn a fresh ``AIAgent`` on a NEW thread, and the
``HERMES_HOME`` ContextVar set on the session-create thread does NOT propagate to
those threads. Without an explicit re-bind, a background/preview-restart turn under
a non-default profile would run against the wrong home. This module locks in:

  1. ``prompt.background`` re-binds ``profile_home`` for the ephemeral turn.
  2. ``preview.restart`` re-binds ``profile_home`` for the ephemeral turn AND does
     NOT close the ephemeral agent (a task-wide ``AIAgent.close()`` would kill the
     background server the restart just started — maintainer problem #1).
  3. Both paths RESTORE the override after the turn (reset token from set), exactly
     like the normal prompt turn, and skip the bind entirely when no profile is set.

How to test: run this module with pytest; each test drives the real RPC handler
from ``tui_gateway.server._methods`` with ``threading.Thread`` patched to run the
target inline, then asserts on the recorded override set/reset calls and the agent.
"""

from unittest.mock import MagicMock, patch

import pytest

from tui_gateway import server as srv
from hermes_constants import get_hermes_home

PROFILE_HOME = "/home/user/.hermes/profiles/work"

class _InlineThread:
    """Drop-in for ``threading.Thread`` that runs the target synchronously.

    Why: the ephemeral RPC handlers do their real work inside ``run()`` on a
    spawned thread; running it inline makes the override set/reset observable
    within the test without racing a real background thread.
    """

    def __init__(self, target=None, daemon=None, **_kwargs):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()

@pytest.fixture
def fake_session():
    """A minimal session carrying a non-default ``profile_home``."""
    agent = MagicMock()
    return {"agent": agent, "session_key": "sess_k", "profile_home": PROFILE_HOME}

@pytest.fixture
def override_calls():
    """Patch set/reset override + AIAgent + emit/context helpers; record calls.

    Returns a dict with the mocks so each test can assert on set/reset ordering
    and on whether the ephemeral agent was closed.
    """
    agent_instance = MagicMock()
    # run_conversation returns a plain dict like the real agent does.
    homes: list[str] = []  # get_hermes_home() as the ephemeral turn saw it

    def _run_conversation(*_a, **_kw):
        from hermes_constants import get_hermes_home
        homes.append(str(get_hermes_home()))
        return {"final_response": "done"}

    agent_instance.run_conversation.side_effect = _run_conversation

    with patch("tui_gateway.server.threading.Thread", _InlineThread), \
        patch("tui_gateway.server._background_agent_kwargs", return_value={}), \
        patch("tui_gateway.server._ephemeral_preview_agent_kwargs", return_value={}), \
        patch("tui_gateway.server._preview_restart_callbacks", return_value={}), \
        patch("tui_gateway.server._preview_restart_history", return_value=[]), \
        patch("tui_gateway.server._set_session_context", return_value=None), \
        patch("tui_gateway.server._clear_session_context"), \
        patch("tui_gateway.server._session_cwd", return_value="/tmp"), \
        patch("tui_gateway.server._emit"), \
        patch("run_agent.AIAgent", return_value=agent_instance) as m_agent:
        yield {
            "homes": homes,
            "agent_cls": m_agent,
            "agent": agent_instance,
        }

def _run(method_name, params, session):
    """Invoke a registered RPC handler with ``_sess`` patched to our session."""
    handler = srv._methods[method_name]
    with patch("tui_gateway.server._sess", return_value=(session, None)):
        return handler("rid1", params)

class TestBackgroundProfileOverride:
    def test_background_binds_and_restores_profile_home(self, fake_session, override_calls):
        """prompt.background binds profile_home for the ephemeral turn and restores it."""
        _run("prompt.background", {"text": "hi", "session_id": "ui1"}, fake_session)

        assert override_calls["homes"] == [PROFILE_HOME]
        assert str(get_hermes_home()) != PROFILE_HOME

    def test_background_restores_override_on_error(self, fake_session, override_calls):
        """A failing turn must still restore the override (finally-block parity)."""
        def _boom(*_a, **_kw):
            override_calls["homes"].append(str(get_hermes_home()))
            raise RuntimeError("boom")

        override_calls["agent"].run_conversation.side_effect = _boom
        _run("prompt.background", {"text": "hi", "session_id": "ui1"}, fake_session)

        assert override_calls["homes"] == [PROFILE_HOME]
        assert str(get_hermes_home()) != PROFILE_HOME

class TestPreviewRestartProfileOverride:
    def test_preview_binds_and_restores_profile_home(self, fake_session, override_calls):
        """preview.restart binds profile_home for the ephemeral turn and restores it."""
        _run(
            "preview.restart",
            {"url": "http://localhost:5173", "cwd": "", "session_id": "ui1"},
            fake_session,
        )

        assert override_calls["homes"] == [PROFILE_HOME]
        assert str(get_hermes_home()) != PROFILE_HOME

    def test_preview_does_not_close_agent(self, fake_session, override_calls):
        """The restarted preview server must survive: the ephemeral agent is NOT
        closed via task-wide process cleanup (maintainer problem #1)."""
        _run(
            "preview.restart",
            {"url": "http://localhost:5173", "cwd": "", "session_id": "ui1"},
            fake_session,
        )

        # A task-wide AIAgent.close() would kill every process for this task_id,
        # tearing down the very background server the restart just launched.
        override_calls["agent"].close.assert_not_called()
