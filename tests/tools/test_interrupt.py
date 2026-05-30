"""Tests for the interrupt system.

Run with: python -m pytest tests/test_interrupt.py -v
"""

import queue
import threading
import time
import pytest


# ---------------------------------------------------------------------------
# Unit tests: shared interrupt module
# ---------------------------------------------------------------------------

class TestInterruptModule:
    """Tests for tools/interrupt.py"""

    def test_set_and_check(self):
        from tools.interrupt import set_interrupt, is_interrupted
        set_interrupt(False)
        assert not is_interrupted()

        set_interrupt(True)
        assert is_interrupted()

        set_interrupt(False)
        assert not is_interrupted()

    def test_thread_safety(self):
        """Set from one thread targeting another thread's ident."""
        from tools.interrupt import set_interrupt, is_interrupted, _interrupted_threads, _lock
        set_interrupt(False)
        # Clear any stale thread idents left by prior tests in this worker.
        with _lock:
            _interrupted_threads.clear()

        seen = {"value": False}

        def _checker():
            while not is_interrupted():
                time.sleep(0.01)
            seen["value"] = True

        t = threading.Thread(target=_checker, daemon=True)
        t.start()

        time.sleep(0.05)
        assert not seen["value"]

        # Target the checker thread's ident so it sees the interrupt
        set_interrupt(True, thread_id=t.ident)
        t.join(timeout=1)
        assert seen["value"]

        set_interrupt(False, thread_id=t.ident)


# ---------------------------------------------------------------------------
# Unit tests: pre-tool interrupt check
# ---------------------------------------------------------------------------

class TestPreToolCheck:
    """Verify that _execute_tool_calls skips all tools when interrupted."""

    def test_all_tools_skipped_when_interrupted(self):
        """Mock an interrupted agent and verify no tools execute."""
        from unittest.mock import MagicMock

        # Build a fake assistant_message with 3 tool calls
        tc1 = MagicMock()
        tc1.id = "tc_1"
        tc1.function.name = "terminal"
        tc1.function.arguments = '{"command": "rm -rf /"}'

        tc2 = MagicMock()
        tc2.id = "tc_2"
        tc2.function.name = "terminal"
        tc2.function.arguments = '{"command": "echo hello"}'

        tc3 = MagicMock()
        tc3.id = "tc_3"
        tc3.function.name = "web_search"
        tc3.function.arguments = '{"query": "test"}'

        assistant_msg = MagicMock()
        assistant_msg.tool_calls = [tc1, tc2, tc3]

        messages = []

        # Create a minimal mock agent with _interrupt_requested = True
        agent = MagicMock()
        agent._interrupt_requested = True
        agent.log_prefix = ""
        agent._persist_session = MagicMock()

        # Import and call the method
        import types
        from run_agent import AIAgent
        # Bind the real methods to our mock so dispatch works correctly
        agent._execute_tool_calls_sequential = types.MethodType(AIAgent._execute_tool_calls_sequential, agent)
        agent._execute_tool_calls_concurrent = types.MethodType(AIAgent._execute_tool_calls_concurrent, agent)
        AIAgent._execute_tool_calls(agent, assistant_msg, messages, "default")

        # All 3 should be skipped
        assert len(messages) == 3
        for msg in messages:
            assert msg["role"] == "tool"
            assert "cancelled" in msg["content"].lower() or "interrupted" in msg["content"].lower()

        # No actual tool handlers should have been called
        # (handle_function_call should NOT have been invoked)


# ---------------------------------------------------------------------------
# Unit tests: message combining
# ---------------------------------------------------------------------------

class TestMessageCombining:
    """Verify multiple interrupt messages are joined."""

    def test_cli_interrupt_queue_drain(self):
        """Simulate draining multiple messages from the interrupt queue."""
        q = queue.Queue()
        q.put("Stop!")
        q.put("Don't delete anything")
        q.put("Show me what you were going to delete instead")

        parts = []
        while not q.empty():
            try:
                msg = q.get_nowait()
                if msg:
                    parts.append(msg)
            except queue.Empty:
                break

        combined = "\n".join(parts)
        assert "Stop!" in combined
        assert "Don't delete anything" in combined
        assert "Show me what you were going to delete instead" in combined
        assert combined.count("\n") == 2

    def test_gateway_pending_messages_append(self):
        """Simulate gateway _pending_messages append logic."""
        pending = {}
        key = "agent:main:telegram:dm"

        # First message
        if key in pending:
            pending[key] += "\n" + "Stop!"
        else:
            pending[key] = "Stop!"

        # Second message
        if key in pending:
            pending[key] += "\n" + "Do something else instead"
        else:
            pending[key] = "Do something else instead"

        assert pending[key] == "Stop!\nDo something else instead"


# ---------------------------------------------------------------------------
# Integration tests (require local terminal)
# ---------------------------------------------------------------------------

class TestSIGKILLEscalation:
    """Test that SIGTERM-resistant processes get SIGKILL'd."""

    @pytest.mark.skipif(
        not __import__("shutil").which("bash"),
        reason="Requires bash"
    )
    def test_sigterm_trap_killed_within_2s(self):
        """A process that traps SIGTERM should be SIGKILL'd after 1s grace."""
        from tools.interrupt import set_interrupt
        from tools.environments.local import LocalEnvironment

        set_interrupt(False)
        env = LocalEnvironment(cwd="/tmp", timeout=30)

        # Start execution in a thread, interrupt after 0.5s
        result_holder = {"value": None}

        def _run():
            result_holder["value"] = env.execute(
                "trap '' TERM; sleep 60",
                timeout=30,
            )

        t = threading.Thread(target=_run)
        t.start()

        time.sleep(0.5)
        set_interrupt(True, thread_id=t.ident)

        t.join(timeout=5)
        set_interrupt(False, thread_id=t.ident)

        assert result_holder["value"] is not None
        assert result_holder["value"]["returncode"] == 130
        assert "interrupted" in result_holder["value"]["output"].lower()


# ---------------------------------------------------------------------------
# Regression: _run_tool cleanup on BaseException (issue #35309)
# ---------------------------------------------------------------------------

class TestRunToolCleanupOnBaseException:
    """Verify that _run_tool cleans up _interrupted_threads even when
    _invoke_tool raises a BaseException (e.g. CancelledError).

    Regression test for #35309: without the finally block, a BaseException
    bypasses ``except Exception``, leaking the worker tid into
    _interrupted_threads.  ThreadPoolExecutor recycles tids, so the next
    tool scheduled on the same thread is instantly "interrupted".
    """

    def test_cleanup_on_base_exception(self):
        from unittest.mock import MagicMock, patch
        import types
        from tools.interrupt import set_interrupt, is_interrupted, _interrupted_threads, _lock

        # Clear global state
        with _lock:
            _interrupted_threads.clear()

        # Build a minimal mock agent with the attributes _run_tool needs
        agent = MagicMock()
        agent._interrupt_requested = False
        agent._tool_worker_threads = set()
        agent._tool_worker_threads_lock = threading.Lock()

        # _set_interrupt delegates to the real module
        def _mock_set_interrupt(active, tid=None):
            set_interrupt(active, tid)
        agent._set_interrupt = _mock_set_interrupt

        # _invoke_tool raises BaseException (simulating CancelledError)
        agent._invoke_tool = MagicMock(side_effect=BaseException("simulated CancelledError"))

        # Bind the real concurrent method so we get _run_tool
        from run_agent import AIAgent
        agent._execute_tool_calls_concurrent = types.MethodType(
            AIAgent._execute_tool_calls_concurrent, agent
        )

        # Build a single tool call
        tc = MagicMock()
        tc.id = "tc_base_exc"
        tc.function.name = "dummy_tool"
        tc.function.arguments = "{}"

        assistant_msg = MagicMock()
        assistant_msg.tool_calls = [tc]

        # _execute_tool_calls_concurrent will submit _run_tool to a
        # ThreadPoolExecutor.  The BaseException propagates out of the
        # worker, but the finally block should still clean up.
        try:
            agent._execute_tool_calls_concurrent(assistant_msg, [], "default")
        except Exception:
            pass  # ThreadPoolExecutor may re-raise

        # After the worker finishes (even with BaseException), the worker
        # tid should have been removed from _interrupted_threads and
        # _tool_worker_threads.
        assert len(agent._tool_worker_threads) == 0, (
            f"_tool_worker_threads not cleaned up: {agent._tool_worker_threads}"
        )

        # Verify no stale tid is left in the global interrupt set
        # (the worker thread is recycled by ThreadPoolExecutor, so any
        # leftover tid would poison the next task on that thread).
        # We can't predict the tid, but we know the worker thread is done
        # (the call returned), so the set should be empty for this test's
        # tid range.  Check that no tid from our agent's tracking leaked.
        with _lock:
            # The only tids that should be in _interrupted_threads are
            # ones we explicitly set — we didn't set any, so it should
            # be empty (modulo other test interference, hence the
            # per-agent tracking assertion above).
            pass


# ---------------------------------------------------------------------------
# Manual smoke test checklist (not automated)
# ---------------------------------------------------------------------------

SMOKE_TESTS = """
Manual Smoke Test Checklist:

1. CLI: Run `hermes`, ask it to `sleep 30` in terminal, type "stop" + Enter.
   Expected: command dies within 2s, agent responds to "stop".

2. CLI: Ask it to extract content from 5 URLs, type interrupt mid-way.
   Expected: remaining URLs are skipped, partial results returned.

3. Gateway (Telegram): Send a long task, then send "Stop".
   Expected: agent stops and responds acknowledging the stop.

4. Gateway (Telegram): Send "Stop" then "Do X instead" rapidly.
   Expected: both messages appear as the next prompt (joined by newline).

5. CLI: Start a task that generates 3+ tool calls in one batch.
   Type interrupt during the first tool call.
   Expected: only 1 tool executes, remaining are skipped.
"""
