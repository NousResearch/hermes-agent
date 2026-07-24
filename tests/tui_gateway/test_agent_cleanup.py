"""Tests for _cleanup_agent helper and its three call sites (#50197/#50403).

Covers:
- _cleanup_agent unit tests: ordering, independence, edge cases
- _teardown_session integration: delegates to _cleanup_agent
- prompt.background handler: agent materialized, _cleanup_agent called in finally
- preview.restart handler: same pattern, skip_memory ephemeral agent
"""

import threading
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(session_messages=None, *, has_shutdown=True, has_close=True):
    """Build a minimal mock AIAgent for cleanup tests.

    Why: _cleanup_agent inspects hasattr() and reads _session_messages, so
    the mock must reflect realistic attribute presence.
    Test: pass directly to _cleanup_agent and assert call patterns.
    """
    agent = MagicMock()
    agent._session_messages = session_messages
    if not has_shutdown:
        del agent.shutdown_memory_provider
    if not has_close:
        del agent.close
    return agent


# ---------------------------------------------------------------------------
# _cleanup_agent unit tests
# ---------------------------------------------------------------------------


class TestCleanupAgentBasic:
    """Basic behaviour: None guard, attribute dispatch, exception isolation."""

    def test_none_is_noop(self):
        """_cleanup_agent(None) must not raise and must call nothing.

        Why: callers may pass session.get("agent") which returns None on a
        session with no agent yet.
        Test: just assert no exception.
        """
        from tui_gateway.server import _cleanup_agent

        _cleanup_agent(None)  # must not raise

    def test_shutdown_and_close_both_called(self):
        """Both shutdown_memory_provider and close() are invoked.

        Why: the cleanup must drain memory AND free tool resources.
        Test: assert both mocks were called.
        """
        from tui_gateway.server import _cleanup_agent

        agent = _make_agent(session_messages=["msg"])
        _cleanup_agent(agent)

        agent.shutdown_memory_provider.assert_called_once()
        agent.close.assert_called_once()

    def test_agent_without_shutdown_gets_close(self):
        """Agent with no shutdown_memory_provider still gets close() called.

        Why: skip_memory agents omit shutdown; close() must still run.
        Test: remove shutdown mock, assert close is called, no AttributeError.
        """
        from tui_gateway.server import _cleanup_agent

        agent = _make_agent(has_shutdown=False)
        _cleanup_agent(agent)

        agent.close.assert_called_once()

    def test_agent_without_close_still_calls_shutdown(self):
        """Agent with no close() still gets shutdown_memory_provider called.

        Why: future agents may not expose close(); shutdown must not skip.
        Test: remove close mock, assert shutdown called.
        """
        from tui_gateway.server import _cleanup_agent

        agent = _make_agent(session_messages=[], has_close=False)
        _cleanup_agent(agent)

        agent.shutdown_memory_provider.assert_called_once()


# ---------------------------------------------------------------------------
# Ordering: shutdown BEFORE close
# ---------------------------------------------------------------------------


class TestCleanupAgentOrdering:
    """Shutdown must be called before close() so session_messages are intact."""

    def test_shutdown_before_close_ordering(self):
        """shutdown_memory_provider is called before close().

        Why: AIAgent.close() clears _session_messages; if close runs first,
        shutdown would flush an empty list and lose conversation history.
        Test: track call order via a shared list modified by side_effects.
        """
        from tui_gateway.server import _cleanup_agent

        call_order = []

        agent = MagicMock()
        agent._session_messages = ["turn1", "turn2"]
        agent.shutdown_memory_provider.side_effect = lambda *a, **k: call_order.append("shutdown")
        agent.close.side_effect = lambda *a, **k: call_order.append("close")

        _cleanup_agent(agent)

        assert call_order == ["shutdown", "close"], (
            f"Expected shutdown before close, got: {call_order}"
        )

    def test_shutdown_receives_populated_messages_not_empty_list(self):
        """shutdown_memory_provider sees the real messages, not a post-close empty list.

        Why: if close() ran first and cleared _session_messages, shutdown would
        receive [] and the memory provider would flush nothing — losing history.
        Test: simulate close() clearing the list; assert shutdown received the
        original populated list because it ran first.
        """
        from tui_gateway.server import _cleanup_agent

        original_messages = ["user: hello", "assistant: hi"]

        agent = MagicMock()
        agent._session_messages = original_messages

        # Simulate what AIAgent.close() does: clear _session_messages
        def _simulated_close():
            agent._session_messages = []

        agent.close.side_effect = _simulated_close

        _cleanup_agent(agent)

        # shutdown_memory_provider must have been called with the original messages
        agent.shutdown_memory_provider.assert_called_once_with(original_messages)


# ---------------------------------------------------------------------------
# session_messages argument passing
# ---------------------------------------------------------------------------


class TestCleanupAgentSessionMessages:
    """Correct argument passing to shutdown_memory_provider."""

    def test_populated_list_passed_as_positional(self):
        """When _session_messages is a non-empty list, pass it positionally.

        Why: shutdown_memory_provider(messages) flushes via on_session_end(messages);
        passing the real list ensures memory providers see the conversation.
        Test: assert called with the list.
        """
        from tui_gateway.server import _cleanup_agent

        msgs = [{"role": "user", "content": "hello"}]
        agent = _make_agent(session_messages=msgs)
        _cleanup_agent(agent)

        agent.shutdown_memory_provider.assert_called_once_with(msgs)

    def test_empty_list_passed_as_positional(self):
        """An empty list is still a list — passes it positionally.

        Why: an empty list is isinstance(..., list), so the branch takes the
        positional-call path even for empty conversations.
        Test: assert called with [].
        """
        from tui_gateway.server import _cleanup_agent

        agent = _make_agent(session_messages=[])
        _cleanup_agent(agent)

        agent.shutdown_memory_provider.assert_called_once_with([])

    def test_none_messages_calls_with_no_positional_arg(self):
        """When _session_messages is None, call shutdown with no positional arg.

        Why: None is not a list; falling back to the no-arg call keeps
        compatibility with agents that have no messages attribute.
        Test: assert called with no positional arguments.
        """
        from tui_gateway.server import _cleanup_agent

        agent = _make_agent(session_messages=None)
        _cleanup_agent(agent)

        agent.shutdown_memory_provider.assert_called_once_with()

    def test_absent_session_messages_calls_with_no_positional_arg(self):
        """When _session_messages is entirely absent, no-arg call is made.

        Why: getattr returns None for a missing attr, then isinstance(None, list)
        is False, so the no-arg path runs — safe for stub objects.
        Test: use an object with no _session_messages attribute.
        """
        from tui_gateway.server import _cleanup_agent

        class MinimalAgent:
            def shutdown_memory_provider(self):
                pass
            def close(self):
                pass

        mock_shutdown = MagicMock()
        mock_close = MagicMock()
        agent = MinimalAgent()
        agent.shutdown_memory_provider = mock_shutdown
        agent.close = mock_close

        _cleanup_agent(agent)

        mock_shutdown.assert_called_once_with()
        mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# Exception independence
# ---------------------------------------------------------------------------


class TestCleanupAgentExceptionIndependence:
    """Failure in one step must not prevent the other."""

    def test_close_called_even_when_shutdown_raises(self):
        """close() still runs if shutdown_memory_provider raises.

        Why: the two steps are independently guarded; a broken memory provider
        must not leak terminal sandboxes or browser daemons.
        Test: side_effect on shutdown, assert close still called.
        """
        from tui_gateway.server import _cleanup_agent

        agent = _make_agent(session_messages=["x"])
        agent.shutdown_memory_provider.side_effect = RuntimeError("memory provider exploded")

        _cleanup_agent(agent)  # must not raise

        agent.close.assert_called_once()

    def test_shutdown_already_ran_when_close_raises(self):
        """shutdown is not skipped because close raises.

        Why: close() is in a separate try block AFTER shutdown, so shutdown
        always runs first regardless of what close does.
        Test: side_effect on close; assert shutdown ran first.
        """
        from tui_gateway.server import _cleanup_agent

        call_order = []
        agent = MagicMock()
        agent._session_messages = ["msg"]
        agent.shutdown_memory_provider.side_effect = lambda *a, **k: call_order.append("shutdown")
        agent.close.side_effect = RuntimeError("close exploded")

        _cleanup_agent(agent)  # must not raise

        assert "shutdown" in call_order

    def test_both_raising_does_not_propagate(self):
        """Both shutdown and close raising must not propagate to the caller.

        Why: callers (teardown, background thread, preview thread) must not
        crash if the agent is in a bad state.
        Test: both raise; assert no exception escapes.
        """
        from tui_gateway.server import _cleanup_agent

        agent = MagicMock()
        agent._session_messages = []
        agent.shutdown_memory_provider.side_effect = RuntimeError("shutdown failed")
        agent.close.side_effect = RuntimeError("close failed")

        _cleanup_agent(agent)  # must not raise


# ---------------------------------------------------------------------------
# _teardown_session integration
# ---------------------------------------------------------------------------


class TestTeardownSession:
    """_teardown_session must delegate to _cleanup_agent for the session agent."""

    def test_teardown_calls_cleanup_agent_with_session_agent(self):
        """_teardown_session calls _cleanup_agent(session["agent"]).

        Why: the old code only called close(), missing shutdown_memory_provider.
        Test: patch _cleanup_agent; build a minimal session; assert called with agent.
        """
        agent = MagicMock()
        session = {
            "agent": agent,
            "history": [],
            "history_lock": threading.Lock(),
            "session_key": "key_abc",
            "_finalized": False,
        }

        with patch("tui_gateway.server._cleanup_agent") as mock_cleanup, \
             patch("tui_gateway.server._finalize_session"), \
             patch("tui_gateway.server.unregister_gateway_notify", create=True), \
             patch("tui_gateway.server._teardown_session.__wrapped__", None, create=True):
            # Import after patching to get the right module reference
            from tui_gateway.server import _teardown_session
            _teardown_session(session)

        mock_cleanup.assert_called_once_with(agent)

    def test_teardown_none_session_is_noop(self):
        """_teardown_session(None) must not raise.

        Why: callers may pass None when no session exists.
        Test: assert no exception.
        """
        from tui_gateway.server import _teardown_session

        _teardown_session(None)  # must not raise

    def test_teardown_cleanup_called_even_if_shutdown_raises(self):
        """cleanup_agent independence holds end-to-end through teardown.

        Why: _cleanup_agent's internal independence must propagate to callers;
        teardown must not crash if the agent is broken.
        Test: agent whose shutdown raises; verify teardown completes.
        """
        agent = _make_agent(session_messages=["msg"])
        agent.shutdown_memory_provider.side_effect = RuntimeError("broken")

        session = {
            "agent": agent,
            "history": [],
            "history_lock": threading.Lock(),
            "session_key": "test_key",
            "_finalized": False,
        }

        with patch("tui_gateway.server._finalize_session"), \
             patch("tui_gateway.server.unregister_gateway_notify", create=True):
            from tui_gateway.server import _teardown_session
            _teardown_session(session)  # must not raise

        agent.close.assert_called_once()


# ---------------------------------------------------------------------------
# prompt.background handler
# ---------------------------------------------------------------------------


class TestPromptBackgroundHandler:
    """The background agent must be materialized and _cleanup_agent called in finally."""

    def _run_background_thread_synchronously(self, server_mod, text="do it"):
        """Drive the prompt.background handler body synchronously.

        Why: threading.Thread(target=run, daemon=True).start() launches async;
        we patch Thread to capture the target and run it inline.
        """
        parent_agent = MagicMock()
        session = {
            "agent": parent_agent,
            "history": [],
            "history_lock": threading.Lock(),
            "session_key": "sess_bg",
            "_finalized": False,
            "cwd": "/tmp",
        }

        # Inject session into server._sessions so _sess can find it
        sid = "test_bg_sid"
        with server_mod._sessions_lock:
            server_mod._sessions[sid] = session

        try:
            thread_target = {}

            def fake_thread(target=None, daemon=None):
                t = MagicMock()
                thread_target["fn"] = target
                t.start = lambda: None
                return t

            with patch("tui_gateway.server.threading.Thread", side_effect=fake_thread):
                server_mod.dispatch({
                    "id": "r1",
                    "method": "prompt.background",
                    "params": {"session_id": sid, "text": text},
                })

            # Run the thread body synchronously
            if "fn" in thread_target:
                thread_target["fn"]()

        finally:
            with server_mod._sessions_lock:
                server_mod._sessions.pop(sid, None)

    @pytest.fixture()
    def server(self):
        with patch.dict("sys.modules", {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_bg")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        }):
            import importlib
            mod = importlib.import_module("tui_gateway.server")
            yield mod
            with mod._sessions_lock:
                mod._sessions.clear()

    def test_cleanup_called_on_success(self, server):
        """_cleanup_agent is called after successful run_conversation.

        Why: the agent leaked entirely before this fix — no cleanup on success.
        Test: stub run_agent in sys.modules so the local 'from run_agent import AIAgent'
        inside the thread body resolves to our mock; assert _cleanup_agent called
        with the constructed instance (AIAgent.return_value).
        """
        import sys

        fake_run_agent = MagicMock()
        # AIAgent(...) returns this instance inside the handler
        agent_inst = fake_run_agent.AIAgent.return_value
        agent_inst.run_conversation.return_value = {"final_response": "ok"}

        with patch.dict(sys.modules, {"run_agent": fake_run_agent}), \
             patch("tui_gateway.server._cleanup_agent") as mock_cleanup, \
             patch("tui_gateway.server._emit"):
            self._run_background_thread_synchronously(server)
            # _cleanup_agent must be called once with the instance AIAgent() returned
            assert mock_cleanup.call_count == 1
            assert mock_cleanup.call_args[0][0] is fake_run_agent.AIAgent.return_value

    def test_cleanup_called_when_run_conversation_raises(self, server):
        """_cleanup_agent is called even if run_conversation raises.

        Why: the inner finally must run on exception so resources are freed.
        Test: run_conversation raises; assert _cleanup_agent still called.
        """
        import sys

        fake_run_agent = MagicMock()
        agent_inst = fake_run_agent.AIAgent.return_value
        agent_inst.run_conversation.side_effect = RuntimeError("agent crashed")

        with patch.dict(sys.modules, {"run_agent": fake_run_agent}), \
             patch("tui_gateway.server._cleanup_agent") as mock_cleanup, \
             patch("tui_gateway.server._emit"):
            self._run_background_thread_synchronously(server)
            assert mock_cleanup.call_count == 1
            assert mock_cleanup.call_args[0][0] is fake_run_agent.AIAgent.return_value


# ---------------------------------------------------------------------------
# preview.restart handler
# ---------------------------------------------------------------------------


class TestPreviewRestartHandler:
    """The preview restart agent must be materialized and run, but NOT closed.

    Why: the ephemeral preview agent's session_id IS its task_id, and
    AIAgent.close() does a task-wide kill (process_registry.kill_all(task_id),
    cleanup_vm/cleanup_browser). preview.restart's whole purpose is to leave a
    detached server RUNNING under that task_id, so calling _cleanup_agent /
    close() here would tear down the very server the restart just started. This
    path therefore intentionally does NOT close the agent.
    """

    @pytest.fixture()
    def server(self):
        with patch.dict("sys.modules", {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_preview")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        }):
            import importlib
            mod = importlib.import_module("tui_gateway.server")
            yield mod
            with mod._sessions_lock:
                mod._sessions.clear()

    def _run_preview_thread_synchronously(self, server_mod):
        """Drive the preview.restart handler body synchronously."""
        parent_agent = MagicMock()
        session = {
            "agent": parent_agent,
            "history": [],
            "history_lock": threading.Lock(),
            "session_key": "sess_preview",
            "_finalized": False,
            "cwd": "/tmp",
        }

        sid = "test_preview_sid"
        with server_mod._sessions_lock:
            server_mod._sessions[sid] = session

        try:
            thread_target = {}

            def fake_thread(target=None, daemon=None):
                t = MagicMock()
                thread_target["fn"] = target
                t.start = lambda: None
                return t

            with patch("tui_gateway.server.threading.Thread", side_effect=fake_thread):
                server_mod.dispatch({
                    "id": "r2",
                    "method": "preview.restart",
                    "params": {
                        "session_id": sid,
                        "url": "http://localhost:3000",
                        "cwd": "/tmp",
                        "context": "",
                    },
                })

            if "fn" in thread_target:
                thread_target["fn"]()
        finally:
            with server_mod._sessions_lock:
                server_mod._sessions.pop(sid, None)

    def test_agent_run_but_not_closed_on_success(self, server):
        """The preview agent runs the conversation but is NOT closed on success.

        Why: closing would task-wide-kill the just-started restart server (its
        session_id IS task_id). The restarted server must survive.
        Test: stub run_agent; assert run_conversation ran and neither
        _cleanup_agent nor the agent's own close()/shutdown_memory_provider
        was invoked.
        """
        import sys

        fake_run_agent = MagicMock()
        agent_inst = fake_run_agent.AIAgent.return_value
        agent_inst.run_conversation.return_value = {"final_response": "restarted"}

        with patch.dict(sys.modules, {
                "run_agent": fake_run_agent,
                "tools.terminal_tool": MagicMock(),
            }), \
             patch("tui_gateway.server._cleanup_agent") as mock_cleanup, \
             patch("tui_gateway.server._emit"):
            self._run_preview_thread_synchronously(server)
            # The agent must have run.
            agent_inst.run_conversation.assert_called_once()
            # But it must NOT be closed — the restart server must survive.
            assert mock_cleanup.call_count == 0
            agent_inst.close.assert_not_called()
            agent_inst.shutdown_memory_provider.assert_not_called()

    def test_agent_not_closed_when_run_conversation_raises(self, server):
        """Even if run_conversation raises, the agent is NOT closed.

        Why: a task-wide close on the exception path would still kill any
        partially-started server processes registered under this task_id; the
        error path emits a failure event but must not tear down the task.
        Test: run_conversation raises; assert _cleanup_agent/close never called.
        """
        import sys

        fake_run_agent = MagicMock()
        agent_inst = fake_run_agent.AIAgent.return_value
        agent_inst.run_conversation.side_effect = RuntimeError("preview restart exploded")

        with patch.dict(sys.modules, {
                "run_agent": fake_run_agent,
                "tools.terminal_tool": MagicMock(),
            }), \
             patch("tui_gateway.server._cleanup_agent") as mock_cleanup, \
             patch("tui_gateway.server._emit"):
            self._run_preview_thread_synchronously(server)
            assert mock_cleanup.call_count == 0
            agent_inst.close.assert_not_called()
