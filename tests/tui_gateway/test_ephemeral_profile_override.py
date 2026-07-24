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
    agent_instance.run_conversation.return_value = {"final_response": "done"}

    with patch("tui_gateway.server.threading.Thread", _InlineThread), \
        patch("tui_gateway.server.set_hermes_home_override", return_value="TOK") as m_set, \
        patch("tui_gateway.server.reset_hermes_home_override") as m_reset, \
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
            "set": m_set,
            "reset": m_reset,
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

        override_calls["set"].assert_called_once_with(PROFILE_HOME)
        override_calls["reset"].assert_called_once_with("TOK")
        override_calls["agent"].run_conversation.assert_called_once()

    def test_background_no_profile_skips_override(self, override_calls):
        """With no profile_home the background path never touches the override."""
        session = {"agent": MagicMock(), "session_key": "sess_k", "profile_home": None}
        _run("prompt.background", {"text": "hi", "session_id": "ui1"}, session)

        override_calls["set"].assert_not_called()
        override_calls["reset"].assert_not_called()
        override_calls["agent"].run_conversation.assert_called_once()

    def test_background_restores_override_on_error(self, fake_session, override_calls):
        """A failing turn must still restore the override (finally-block parity)."""
        override_calls["agent"].run_conversation.side_effect = RuntimeError("boom")
        _run("prompt.background", {"text": "hi", "session_id": "ui1"}, fake_session)

        override_calls["set"].assert_called_once_with(PROFILE_HOME)
        override_calls["reset"].assert_called_once_with("TOK")


class TestPreviewRestartProfileOverride:
    def test_preview_binds_and_restores_profile_home(self, fake_session, override_calls):
        """preview.restart binds profile_home for the ephemeral turn and restores it."""
        _run(
            "preview.restart",
            {"url": "http://localhost:5173", "cwd": "", "session_id": "ui1"},
            fake_session,
        )

        override_calls["set"].assert_called_once_with(PROFILE_HOME)
        override_calls["reset"].assert_called_once_with("TOK")
        override_calls["agent"].run_conversation.assert_called_once()

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

    def test_preview_no_profile_skips_override(self, override_calls):
        """With no profile_home the preview path never touches the override."""
        session = {"agent": MagicMock(), "session_key": "sess_k", "profile_home": None}
        _run(
            "preview.restart",
            {"url": "http://localhost:5173", "cwd": "", "session_id": "ui1"},
            session,
        )

        override_calls["set"].assert_not_called()
        override_calls["reset"].assert_not_called()
        override_calls["agent"].close.assert_not_called()
