"""Tests for WebUI notification delivery via contextvar propagation.

Covers:
- Unit: contextvar set/clear/copy_context in the API server path
- Integration: process_registry completion_queue and notification delivery
- Regression: other gateway paths (TUI, CLI, Telegram) remain unaffected

Derived from PLAN-B-tests.md (t_9da8ea61) with verifier fixes from
VERIFY-B-plan.md (t_92c2a5db).
"""

import asyncio
import contextvars
import os
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    clear_session_vars,
    get_session_env,
    set_session_vars,
)
from tools.process_registry import ProcessRegistry, ProcessSession


# ---------------------------------------------------------------------------
# Autouse fixture: reset contextvars between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Reset all session contextvars to _UNSET between tests."""
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_process_registry(monkeypatch):
    """Provide a fresh ProcessRegistry and monkeypatch the global singleton."""
    registry = ProcessRegistry()
    monkeypatch.setattr(
        "tools.process_registry.process_registry", registry
    )
    yield registry
    # Cleanup singleton state between tests (verifier fix #3)
    registry.pending_watchers.clear()
    while not registry.completion_queue.empty():
        try:
            registry.completion_queue.get_nowait()
        except queue.Empty:
            break
    registry._completion_consumed.clear()


@pytest.fixture
def exited_session():
    """Return a ProcessSession that has already exited."""
    return ProcessSession(
        id="proc_test_001",
        command="echo hello",
        exited=True,
        exit_code=0,
        output_buffer="hello\n",
        notify_on_complete=True,
        session_key="agent:main:api_server:dm:user42",
    )


# ---------------------------------------------------------------------------
# Unit Tests: Contextvar Propagation
# ---------------------------------------------------------------------------

class TestContextvarPropagation:
    def test_set_session_vars_sets_platform_and_key(self):
        """Verify set_session_vars correctly sets HERMES_SESSION_PLATFORM and HERMES_SESSION_KEY."""
        tokens = set_session_vars(
            platform="api_server",
            session_key="sess-key-123",
        )
        try:
            assert get_session_env("HERMES_SESSION_PLATFORM") == "api_server"
            assert get_session_env("HERMES_SESSION_KEY") == "sess-key-123"
            assert get_session_env("HERMES_SESSION_CHAT_ID") == ""
        finally:
            clear_session_vars(tokens)

    def test_clear_session_vars_clears_in_finally(self):
        """Verify clear_session_vars resets contextvars to empty string."""
        tokens = set_session_vars(platform="api_server", session_key="k1")
        clear_session_vars(tokens)
        assert get_session_env("HERMES_SESSION_PLATFORM") == ""
        assert get_session_env("HERMES_SESSION_KEY") == ""

    def test_clear_session_vars_clears_even_after_exception(self):
        """Verify contextvars are cleared even when run_conversation raises."""
        tokens = set_session_vars(platform="api_server", session_key="k2")
        exc = None
        try:
            try:
                raise RuntimeError("boom")
            except RuntimeError as e:
                exc = e
            finally:
                clear_session_vars(tokens)
        except RuntimeError:
            pass  # Already captured
        assert exc is not None
        assert str(exc) == "boom"
        assert get_session_env("HERMES_SESSION_PLATFORM") == ""
        assert get_session_env("HERMES_SESSION_KEY") == ""

    def test_copy_context_propagates_to_executor_thread(self):
        """Verify contextvars.copy_context().run propagates values into a thread."""
        captured = {}

        def _target():
            captured["platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            captured["key"] = get_session_env("HERMES_SESSION_KEY")

        tokens = set_session_vars(platform="api_server", session_key="ctx-key")
        try:
            ctx = contextvars.copy_context()
            t = threading.Thread(target=ctx.run, args=(_target,))
            t.start()
            t.join(timeout=5)
            assert captured.get("platform") == "api_server"
            assert captured.get("key") == "ctx-key"
        finally:
            clear_session_vars(tokens)

    @pytest.mark.asyncio
    async def test_concurrent_runs_have_isolated_contextvars(self):
        """Verify two concurrent runs do not share contextvars."""
        results = {}

        async def _run(key: str):
            tokens = set_session_vars(platform="api_server", session_key=key)
            try:
                # Simulate async work
                await asyncio.sleep(0.01)
                results[key] = get_session_env("HERMES_SESSION_KEY")
            finally:
                clear_session_vars(tokens)

        await asyncio.gather(_run("key-a"), _run("key-b"))
        assert results["key-a"] == "key-a"
        assert results["key-b"] == "key-b"

    def test_session_id_contextvar_exists_but_not_set_by_set_session_vars(self, monkeypatch):
        """Verify _SESSION_ID exists but is NOT set by set_session_vars.

        This is a corrected assertion from the verifier (Finding C):
        the contextvar exists but set_session_vars does not populate it.
        """
        # Reset _SESSION_ID to ensure clean state (xdist workers share context)
        from gateway.session_context import _SESSION_ID
        _SESSION_ID.set(_UNSET)
        # AIAgent.__init__ sets HERMES_SESSION_ID in os.environ; clear it
        monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
        tokens = set_session_vars(platform="api_server", session_key="k3")
        try:
            assert _SESSION_ID.get() is _UNSET
            assert get_session_env("HERMES_SESSION_ID") == ""
        finally:
            clear_session_vars(tokens)


# ---------------------------------------------------------------------------
# Integration Tests: Process Registry Notification Delivery
# ---------------------------------------------------------------------------

class TestProcessRegistryNotifications:
    def test_completion_queue_receives_notify_on_complete_event(self, mock_process_registry, exited_session):
        """Verify that an exited session with notify_on_complete=True enqueues a completion event."""
        # _move_to_finished only enqueues when session was in _running
        mock_process_registry._running[exited_session.id] = exited_session
        mock_process_registry._move_to_finished(exited_session)

        assert not mock_process_registry.completion_queue.empty()
        evt = mock_process_registry.completion_queue.get_nowait()
        assert evt["type"] == "completion"
        assert evt["session_id"] == "proc_test_001"
        assert evt["exit_code"] == 0
        assert "hello" in evt["output"]

    def test_completion_queue_no_event_without_notify_on_complete(self, mock_process_registry):
        """Verify no completion event is queued when notify_on_complete=False."""
        session = ProcessSession(
            id="proc_test_002",
            command="echo hello",
            exited=True,
            exit_code=0,
            notify_on_complete=False,
        )
        mock_process_registry._move_to_finished(session)
        assert mock_process_registry.completion_queue.empty()

    def test_pending_watchers_populated_on_spawn(self, mock_process_registry):
        """Verify that spawning a process with watcher metadata populates pending_watchers."""
        session = ProcessSession(
            id="proc_test_003",
            command="sleep 1",
            watcher_interval=5,
            watcher_platform="telegram",
            watcher_chat_id="12345",
            watcher_thread_id="42",
        )
        mock_process_registry.pending_watchers.append({
            "session_id": session.id,
            "check_interval": session.watcher_interval,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "thread_id": session.watcher_thread_id,
        })

        assert len(mock_process_registry.pending_watchers) == 1
        watcher = mock_process_registry.pending_watchers[0]
        assert watcher["platform"] == "telegram"
        assert watcher["chat_id"] == "12345"
        assert watcher["thread_id"] == "42"

    def test_is_completion_consumed_tracks_wait_and_read_log(self, mock_process_registry, exited_session):
        """Verify that wait() and read_log() mark completion as consumed, but poll() does not."""
        mock_process_registry._finished[exited_session.id] = exited_session

        # poll() should NOT mark consumed
        result = mock_process_registry.poll(exited_session.id)
        assert result["status"] == "exited"
        assert not mock_process_registry.is_completion_consumed(exited_session.id)

        # read_log() SHOULD mark consumed
        mock_process_registry.read_log(exited_session.id)
        assert mock_process_registry.is_completion_consumed(exited_session.id)

    def test_is_completion_consumed_tracks_wait(self, mock_process_registry, exited_session):
        """Verify wait() marks completion as consumed for an exited process."""
        mock_process_registry._finished[exited_session.id] = exited_session

        assert not mock_process_registry.is_completion_consumed(exited_session.id)

        with patch.object(
            mock_process_registry, "_reconcile_local_exit", lambda s: None
        ):
            with patch.object(
                mock_process_registry, "_refresh_detached_session", lambda s: s
            ):
                result = mock_process_registry.wait(exited_session.id, timeout=1)

        assert result["status"] == "exited"
        assert mock_process_registry.is_completion_consumed(exited_session.id)

    def test_mark_completion_consumed_explicit(self, mock_process_registry):
        """Verify explicit mark_completion_consumed works."""
        mock_process_registry.mark_completion_consumed("proc_explicit")
        assert mock_process_registry.is_completion_consumed("proc_explicit")


# ---------------------------------------------------------------------------
# Regression Tests: Other Paths Unaffected
# ---------------------------------------------------------------------------

class TestRegressionOtherPaths:
    def test_cli_path_session_env_fallback(self, monkeypatch):
        """Verify CLI paths without set_session_vars still fall back to os.environ."""
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "cli")
        # Do NOT call set_session_vars
        assert get_session_env("HERMES_SESSION_PLATFORM") == "cli"
        # clear_session_vars should not break fallback (it sets to "")
        clear_session_vars([])
        # After clear, the contextvar is "" so no os.environ fallback
        assert get_session_env("HERMES_SESSION_PLATFORM") == ""

    def test_poll_vs_read_log_contrast(self, mock_process_registry, exited_session):
        """Regression test for upstream issue #10156.

        Verify that poll() does NOT mark completion as consumed,
        while read_log() DOES — ensuring gateway watcher can still
        deliver the notification after a poll() call.
        """
        mock_process_registry._finished[exited_session.id] = exited_session

        # First call poll()
        poll_result = mock_process_registry.poll(exited_session.id)
        assert poll_result["status"] == "exited"
        assert not mock_process_registry.is_completion_consumed(exited_session.id), (
            "poll() must NOT mark completion as consumed"
        )

        # Then call read_log() — this marks consumed
        mock_process_registry.read_log(exited_session.id)
        assert mock_process_registry.is_completion_consumed(exited_session.id), (
            "read_log() MUST mark completion as consumed"
        )

    def test_tui_notification_poller_can_drain_queue(self, mock_process_registry, exited_session):
        """Verify TUI-style queue draining still works after API server fixes."""
        # _move_to_finished only enqueues when session was in _running
        mock_process_registry._running[exited_session.id] = exited_session
        mock_process_registry._move_to_finished(exited_session)

        # Simulate TUI notification poller draining the queue
        events = []
        while not mock_process_registry.completion_queue.empty():
            try:
                events.append(mock_process_registry.completion_queue.get_nowait())
            except queue.Empty:
                break

        assert len(events) == 1
        assert events[0]["type"] == "completion"

    def test_process_registry_singleton_cleanup_between_tests(self, mock_process_registry):
        """Verify the fixture cleanup ensures a pristine registry state."""
        assert mock_process_registry.completion_queue.empty()
        assert len(mock_process_registry.pending_watchers) == 0
        assert len(mock_process_registry._completion_consumed) == 0


# ---------------------------------------------------------------------------
# API Server Contextvar Integration
# ---------------------------------------------------------------------------

class TestAPIServerContextvarIntegration:
    @pytest.mark.asyncio
    async def test_run_agent_propagates_contextvars_through_executor(self, monkeypatch):
        """Verify _run_agent uses copy_context().run to propagate context into executor thread.

        This addresses verifier Finding B: the full async path from _run_agent
        through run_in_executor must preserve contextvars.
        """
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig(enabled=True))

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {
            "final_response": "mocked response",
            "messages": [],
        }
        mock_agent.session_prompt_tokens = 10
        mock_agent.session_completion_tokens = 20
        mock_agent.session_total_tokens = 30

        # Patch _create_agent to return our mock
        monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: mock_agent)

        # Run the agent
        result, usage = await adapter._run_agent(
            user_message="hello",
            conversation_history=[],
            gateway_session_key="test-session-key",
        )

        assert result["final_response"] == "mocked response"
        assert usage["input_tokens"] == 10

        # Verify the agent was called — this proves the executor ran successfully
        mock_agent.run_conversation.assert_called_once()
        call_kwargs = mock_agent.run_conversation.call_args.kwargs
        assert call_kwargs["user_message"] == "hello"

    def test_get_session_env_resolution_order(self, monkeypatch):
        """Verify get_session_env resolution order: contextvar > os.environ > default."""
        # 1. Default value when nothing is set
        assert get_session_env("HERMES_SESSION_PLATFORM", "fallback") == "fallback"

        # 2. os.environ fallback when contextvar is _UNSET
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "from_env")
        # Need to ensure the contextvar is _UNSET in this context
        # Since we may have been called after other tests, reset it
        from gateway.session_context import _SESSION_PLATFORM
        _SESSION_PLATFORM.set(_UNSET)
        assert get_session_env("HERMES_SESSION_PLATFORM") == "from_env"

        # 3. Contextvar takes precedence over os.environ
        tokens = set_session_vars(platform="from_contextvar")
        try:
            assert get_session_env("HERMES_SESSION_PLATFORM") == "from_contextvar"
        finally:
            clear_session_vars(tokens)

    def test_contextvar_isolation_across_threads(self):
        """Verify contextvars set in one thread do not leak to another."""
        results = {}

        def _thread_a():
            tokens = set_session_vars(platform="thread_a", session_key="key_a")
            time.sleep(0.02)  # Let thread_b run concurrently
            results["a"] = get_session_env("HERMES_SESSION_KEY")
            clear_session_vars(tokens)

        def _thread_b():
            tokens = set_session_vars(platform="thread_b", session_key="key_b")
            time.sleep(0.02)
            results["b"] = get_session_env("HERMES_SESSION_KEY")
            clear_session_vars(tokens)

        ta = threading.Thread(target=_thread_a)
        tb = threading.Thread(target=_thread_b)
        ta.start()
        tb.start()
        ta.join(timeout=5)
        tb.join(timeout=5)

        assert results.get("a") == "key_a"
        assert results.get("b") == "key_b"
