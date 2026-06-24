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

        # Verify no stale tid is left in the global interrupt set.  The
        # worker thread is recycled by ThreadPoolExecutor, so a leaked tid
        # would poison the next task on that thread.  We cleared the set at
        # the start and never set any interrupt ourselves, so a leak from
        # _run_tool is the only way an entry could land here.
        with _lock:
            leaked = set(_interrupted_threads)
        assert leaked == set(), f"leaked tids in _interrupted_threads: {leaked}"


# ---------------------------------------------------------------------------
# Regression: concurrent tool-batch aggregate deadline (HERMES_TOOL_BATCH_TIMEOUT)
# ---------------------------------------------------------------------------

class TestConcurrentBatchTimeout:
    """Verify the aggregate batch-deadline escape in
    ``execute_tool_calls_concurrent``.

    When a tool ignores the per-thread interrupt and runs past
    ``HERMES_TOOL_BATCH_TIMEOUT``, the executor must:

      1. stop blocking the turn (the ThreadPoolExecutor is shut down with
         ``wait=False`` so a wedged daemon worker can't pin the turn forever),
      2. synthesize a timeout result into the wedged tool's still-None slot,
         flagged ``is_error=True`` with a message naming the batch deadline.

    Critically, that synthesized result tuple is hand-built positionally at
    the deadline site, so a future change to the ``_run_tool`` results-tuple
    shape would silently desync it.  This test pins the shape: it runs a
    normal tool and a wedged tool in the SAME batch and asserts the
    synthesized timeout tuple unpacks identically to a real ``_run_tool``
    result (same 7-field arity / downstream contract).  If the tuple drifts,
    the post-execution unpack (``... = r``) raises ValueError and this test
    fails loudly — which is the whole point of the regression.
    """

    @pytest.mark.timeout(30, method="thread")
    def test_wedged_tool_synthesizes_timeout_result(self, monkeypatch):
        from unittest.mock import MagicMock
        import types

        from tools.interrupt import _interrupted_threads, _lock
        with _lock:
            _interrupted_threads.clear()

        # Trip the aggregate deadline almost immediately.  The wait loop polls
        # every 5s, so the deadline is observed on the first poll after the
        # blocked tool is still running.
        monkeypatch.setenv("HERMES_TOOL_BATCH_TIMEOUT", "0.5")

        # ── Capture each result tuple as it is unpacked downstream ──────
        # The post-execution loop unpacks every results[] slot via
        #   function_name, function_args, function_result, tool_duration,
        #   is_error, blocked, middleware_trace = r
        # then calls agent._append_guardrail_observation(name, args, result,
        # failed=is_error) for non-blocked tools.  By making that a real
        # function we observe the unpacked fields for BOTH the fast tool and
        # the synthesized-timeout tool — proving the synthesized tuple
        # unpacked cleanly into the same 7-field shape.
        observed = []  # list of (name, args, result, failed)

        def _record_guardrail_observation(name, args, result, failed=False):
            observed.append((name, args, result, failed))
            return result

        # ── Minimal mock agent ─────────────────────────────────────────
        agent = MagicMock()
        agent._interrupt_requested = False
        agent._tool_worker_threads = set()
        agent._tool_worker_threads_lock = threading.Lock()
        agent.quiet_mode = True
        agent.tool_progress_mode = "off"
        agent.tool_progress_callback = None
        agent.tool_start_callback = None
        agent.tool_complete_callback = None
        agent.verbose_logging = False
        agent._should_emit_quiet_tool_messages = MagicMock(return_value=False)
        agent._should_start_quiet_spinner = MagicMock(return_value=False)
        agent._append_guardrail_observation = _record_guardrail_observation
        # _tool_result_content_for_active_model passes the string through;
        # downstream make_tool_result_message wants a real string.
        agent._tool_result_content_for_active_model = lambda name, result: result
        agent._set_interrupt = lambda active, tid=None: None
        # No subdirectory hints (a truthy MagicMock would be string-concatenated
        # onto the result and blow up the post-execution loop).
        agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
        agent._apply_pending_steer_to_tool_results = MagicMock()

        # Guardrails must allow execution (a MagicMock decision is truthy,
        # but be explicit so a future bool() change can't silently block).
        agent._tool_guardrails.before_call.return_value.allows_execution = True

        # ── _invoke_tool: 'wedged' blocks past the deadline ignoring the
        # interrupt signal; 'fast' returns immediately. ────────────────
        def _invoke_tool(function_name, function_args, *a, **k):
            if function_name == "wedged":
                # Block far past the 0.5s deadline, ignoring interrupts — this
                # is the daemon worker the deadline must abandon.  Sleep well
                # beyond the executor's 5s poll interval so the only way the
                # turn can return is via the deadline (not natural completion).
                time.sleep(30.0)
                return "wedged finished (should have been abandoned)"
            return "fast ok"

        agent._invoke_tool = _invoke_tool

        # Bind the real concurrent executor.
        from run_agent import AIAgent
        agent._execute_tool_calls_concurrent = types.MethodType(
            AIAgent._execute_tool_calls_concurrent, agent
        )

        # ── Two tool calls in one batch: fast + wedged ─────────────────
        tc_fast = MagicMock()
        tc_fast.id = "tc_fast"
        tc_fast.function.name = "fast"
        tc_fast.function.arguments = "{}"

        tc_wedged = MagicMock()
        tc_wedged.id = "tc_wedged"
        tc_wedged.function.name = "wedged"
        tc_wedged.function.arguments = "{}"

        assistant_msg = MagicMock()
        assistant_msg.tool_calls = [tc_fast, tc_wedged]

        messages = []

        # (a) The call must RETURN (not hang), despite the wedged tool
        # sleeping 30s.  The executor polls every 5s, so the 0.5s deadline is
        # observed at the first poll boundary (~5s).  Anything well under the
        # wedged tool's 30s sleep proves the deadline — not natural
        # completion — released the turn.
        t0 = time.time()
        agent._execute_tool_calls_concurrent(assistant_msg, messages, "default")
        elapsed = time.time() - t0

        assert elapsed < 12.0, (
            f"batch did not honor the deadline; took {elapsed:.2f}s "
            "(approaching the wedged tool's 30s sleep — deadline did not fire)"
        )

        # (b) Both tools were observed downstream — i.e. both result tuples
        # unpacked into 7 fields without a ValueError.  The fast tool and the
        # synthesized-timeout tuple share the same shape; if the synthesized
        # tuple's arity drifted, the unpack at the post-execution loop would
        # raise and we'd never reach _append_guardrail_observation for it.
        observed_by_name = {name: (name, args, result, failed) for (name, args, result, failed) in observed}
        assert "fast" in observed_by_name, f"fast tool not observed: {observed}"
        assert "wedged" in observed_by_name, (
            f"wedged tool's synthesized timeout result was not unpacked "
            f"downstream (arity drift would cause this): {observed}"
        )

        # (c) The wedged tool's synthesized slot is flagged as an error and
        # its message names the batch deadline.
        _wname, _wargs, _wresult, _wfailed = observed_by_name["wedged"]
        assert _wfailed is True, "synthesized timeout result not flagged is_error=True"
        assert "batch" in _wresult.lower() and "deadline" in _wresult.lower(), (
            f"timeout message does not mention the batch deadline: {_wresult!r}"
        )

        # (d) A tool result message for the wedged tool reached the
        # conversation so the model isn't left with a dangling tool_call.
        wedged_tool_msgs = [
            m for m in messages
            if isinstance(m, dict) and m.get("role") == "tool"
            and m.get("tool_call_id") == "tc_wedged"
        ]
        assert wedged_tool_msgs, f"no tool result message for wedged tool: {messages}"


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
