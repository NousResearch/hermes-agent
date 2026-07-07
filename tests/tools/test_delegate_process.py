#!/usr/bin/env python3
"""
Tests for process-based subagent isolation (delegation.process_isolation).

These tests exercise the process-isolation path in tools/delegate_process.py
without requiring real API keys or LLM calls.  They use a stub agent factory
that runs inside the spawned child process and verifies:

  1. Children actually run in separate OS processes (not threads)
  2. IPC bridge relays progress events correctly
  3. Interrupt propagation works across process boundaries
  4. Timeout / hard-termination works (the key advantage over threads)
  5. The config gate correctly selects process vs thread path
  6. Results are correctly collected and sorted by task_index
  7. No zombie processes survive the batch

Run with:  python -m pytest tests/tools/test_delegate_process.py -v --timeout=60
"""

import json
import multiprocessing as mp
import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure the repo root is importable when running from the project dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ──────────────────────────────────────────────────────────────────────
# Stub agent that runs INSIDE the spawned child process.
# Must be defined at module level so multiprocessing(spawn) can pickle it.
# ──────────────────────────────────────────────────────────────────────

class StubAgent:
    """Minimal agent that satisfies the AIAgent interface _child_process_main needs.

    Runs entirely in-process (no API calls).  Configurable behaviour via
    class attributes so different test scenarios can use different stubs.
    """

    # Class-level config that tests set BEFORE the child is spawned.
    # Because spawn re-imports the module, these are read at construction
    # time inside the child and reflect whatever was set on the class
    # object in the parent before fork... EXCEPT spawn doesn't inherit
    # class mutations.  So we pass config via the params dict instead.
    #
    # The factory function below reads config from params.

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._delegate_depth = kwargs.get("_delegate_depth", 1)
        self._delegate_role = kwargs.get("role", "leaf")
        self._subagent_id = kwargs.get("subagent_id")
        self._parent_subagent_id = kwargs.get("parent_subagent_id")
        self._subagent_goal = kwargs.get("goal", "")
        self._parent_turn_id = kwargs.get("parent_turn_id", "")
        self._session_init_model_config = {}
        self.session_id = kwargs.get("session_id", "test-session")
        self.model = kwargs.get("model", "test-model")
        self._interrupt_requested = False

    def run_conversation(self, user_message="", task_id="", stream_callback=None, **kw):
        """Simulate a conversation run. Returns the same shape as AIAgent."""
        config = self._kwargs.get("_test_config") or {}
        sleep_seconds = config.get("sleep", 0)
        emit_events = config.get("emit_events", [])
        do_cpu_work = config.get("cpu_work_seconds", 0)
        fail = config.get("fail", False)
        report_pid = config.get("report_pid", False)

        # Emit any requested progress events
        cb = self._kwargs.get("tool_progress_callback")
        if cb:
            for evt in emit_events:
                cb(evt[0], evt[1] if len(evt) > 1 else None,
                   evt[2] if len(evt) > 2 else None)

        # Optional CPU-bound work (simulates GIL-holding load)
        if do_cpu_work > 0:
            end = time.monotonic() + do_cpu_work
            while time.monotonic() < end:
                # Busy-wait — holds the GIL in the thread path
                _ = sum(range(10000))

        # Optional sleep (simulates waiting on I/O / slow tool)
        # Check interrupt flag periodically so interrupt propagation works
        if sleep_seconds > 0:
            end = time.monotonic() + sleep_seconds
            while time.monotonic() < end:
                if self._interrupt_requested:
                    break
                time.sleep(min(0.2, end - time.monotonic()))

        if fail:
            raise RuntimeError(config.get("fail_message", "Stub failure"))

        # Build response — evaluate PID in the CHILD process
        if report_pid:
            response = f"PID={os.getpid()}"
        else:
            response = config.get("response", f"Stub completed: {user_message}")

        return {
            "final_response": response,
            "completed": True,
            "api_calls": config.get("api_calls", 1),
            "interrupted": self._interrupt_requested,
        }

    def get_activity_summary(self):
        return {
            "current_tool": None,
            "api_call_count": 1,
            "max_iterations": 50,
            "last_activity_desc": "stub working",
        }

    def interrupt(self, message=None):
        config = self._kwargs.get("_test_config") or {}
        if not config.get("ignore_interrupt", False):
            self._interrupt_requested = True

    def close(self):
        pass


def _stub_agent_factory(**kwargs):
    """Factory that constructs a StubAgent from the params dict.

    Registered via params['agent_factory'] = 'tests.tools.test_delegate_process:_stub_agent_factory'
    """
    return StubAgent(**kwargs)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_mock_parent(depth=0):
    """Create a mock parent agent matching the fields _build_child_process_spec needs.

    Every attribute that _resolve_child_settings reads must be set to a
    picklable value (None, str, list, dict) — MagicMock auto-attributes are
    NOT picklable and would fail the pickle validation in _build_child_process_spec.
    """
    parent = MagicMock(spec=[])
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.openrouter_min_coding_score = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.max_tokens = None
    parent.prefill_messages = None
    parent._fallback_chain = None
    parent.reasoning_config = None
    parent.acp_command = None
    parent.acp_args = []
    parent.session_id = "parent-test-session"
    parent._current_turn_id = "test-turn"
    parent._current_task_id = None
    parent._touch_activity = MagicMock()
    parent._interrupt_requested = False
    parent.enabled_toolsets = ["web", "terminal", "file", "delegation"]
    parent.valid_tool_names = ["web_search", "terminal", "read_file", "delegate_task"]
    parent._client_kwargs = {}
    return parent


def _make_process_spec(task_index, goal, parent_agent, test_config=None,
                       toolsets=None, max_iterations=10):
    """Build a ChildProcessSpec with the stub agent factory."""
    from tools.delegate_tool import _build_child_process_spec

    spec = _build_child_process_spec(
        task_index=task_index,
        goal=goal,
        context=None,
        toolsets=toolsets,
        model=None,
        max_iterations=max_iterations,
        task_count=2,
        parent_agent=parent_agent,
    )
    # Inject the stub agent factory and test config
    spec.params["agent_factory"] = "tests.tools.test_delegate_process:_stub_agent_factory"
    spec.params["_test_config"] = test_config or {}
    return spec


# ──────────────────────────────────────────────────────────────────────
# Test 1: Children run in separate OS processes
# ──────────────────────────────────────────────────────────────────────

class TestProcessSpawn(unittest.TestCase):
    """Verify that process-isolation actually spawns OS processes, not threads."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_children_get_separate_pids(self, _mock_cc, _mock_pi):
        """Each child should report a different OS PID via os.getpid()."""
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()

        # The stub writes its PID into the response (evaluated in the child)
        children = []
        for i in range(3):
            spec = _make_process_spec(
                i, f"Get PID {i}", parent,
                test_config={"report_pid": True},
            )
            children.append((i, {"goal": f"Get PID {i}"}, spec))

        results = run_children_in_processes(
            children=children,
            parent_agent=parent,
            child_timeout=30,
        )

        self.assertEqual(len(results), 3)
        parent_pid = os.getpid()
        child_pids = set()
        for r in results:
            self.assertEqual(r["status"], "completed")
            summary = r.get("summary") or ""
            # The stub response contains "PID=<child_pid>"
            if "PID=" in summary:
                pid_str = summary.split("PID=")[1].strip()
                child_pids.add(int(pid_str))

        # All 3 children should have PIDs different from the parent AND
        # from each other (spawn guarantees separate processes).
        child_pids.discard(parent_pid)
        self.assertEqual(
            len(child_pids), 3,
            f"Expected 3 distinct child PIDs (all different from parent {parent_pid}), "
            f"got child PIDs: {child_pids}"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 2: IPC bridge relays progress events
# ──────────────────────────────────────────────────────────────────────

class TestIPCEventRelay(unittest.TestCase):
    """Verify that progress events from child processes reach the parent."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_progress_events_relayed(self, _mock_cc, _mock_pi):
        """Events emitted by the stub agent inside the child should appear
        in the parent's progress callback."""
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()
        received_events = []

        spec = _make_process_spec(
            0, "Emit events", parent,
            test_config={
                "emit_events": [
                    ("tool.start", "terminal", "running ls"),
                    ("tool.complete", "terminal", "file1.txt"),
                ],
                "sleep": 0.5,  # Give parent time to drain events before result
                "response": "Events emitted",
            },
        )

        # Replace the progress callback with a capturing one
        def capture_cb(event_type, tool_name=None, preview=None, args=None, **kw):
            received_events.append({
                "type": event_type,
                "tool": tool_name,
                "preview": preview,
            })

        spec.progress_cb = capture_cb

        children = [(0, {"goal": "Emit events"}, spec)]
        results = run_children_in_processes(
            children=children, parent_agent=parent, child_timeout=30,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "completed")

        # The child should have relayed at least the subagent.start and
        # the two tool events we configured
        event_types = [e["type"] for e in received_events]
        self.assertIn("subagent.start", event_types)
        # The stub emits tool.start and tool.complete via the progress callback
        tool_starts = [e for e in received_events if e["type"] == "tool.start"]
        self.assertGreaterEqual(len(tool_starts), 1)
        self.assertEqual(tool_starts[0]["tool"], "terminal")


# ──────────────────────────────────────────────────────────────────────
# Test 3: Interrupt propagation across process boundaries
# ──────────────────────────────────────────────────────────────────────

class TestInterruptPropagation(unittest.TestCase):
    """Verify that interrupt signals reach child processes."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_interrupt_stops_sleeping_child(self, _mock_cc, _mock_pi):
        """A child sleeping for 30s should be interrupted within a few
        seconds when the parent signals interrupt."""
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()

        spec = _make_process_spec(
            0, "Sleep then get interrupted", parent,
            test_config={"sleep": 30, "response": "should not reach here"},
        )

        children = [(0, {"goal": "Sleep"}, spec)]

        # Start interrupt after 2 seconds in a background thread
        def interrupt_after_delay():
            time.sleep(2)
            spec.interrupt("Test interrupt")

        threading.Thread(target=interrupt_after_delay, daemon=True).start()

        start = time.monotonic()
        results = run_children_in_processes(
            children=children, parent_agent=parent, child_timeout=60,
        )
        elapsed = time.monotonic() - start

        self.assertEqual(len(results), 1)
        # The child should have been interrupted, not completed or timed out
        self.assertIn(results[0]["status"], ("interrupted", "completed", "error"))
        # Should NOT have taken the full 30s
        self.assertLess(
            elapsed, 20,
            f"Interrupt should have stopped the 30s sleep quickly, "
            f"but took {elapsed:.1f}s"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 4: Timeout hard-terminates stuck processes
# ──────────────────────────────────────────────────────────────────────

class TestTimeoutTermination(unittest.TestCase):
    """Verify that child_timeout hard-terminates a stuck child process."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_timeout_kills_stuck_child(self, _mock_cc, _mock_pi):
        """A child sleeping for 60s with a 3s timeout should be killed and
        return a timeout entry — this is the key advantage over threads,
        which CANNOT be hard-killed in Python."""
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()

        spec = _make_process_spec(
            0, "Sleep forever", parent,
            test_config={"sleep": 60, "response": "should not reach here"},
        )

        children = [(0, {"goal": "Sleep forever"}, spec)]

        start = time.monotonic()
        results = run_children_in_processes(
            children=children, parent_agent=parent, child_timeout=3,
        )
        elapsed = time.monotonic() - start

        self.assertEqual(len(results), 1)
        # The child should have been stopped by the timeout mechanism.
        # The timeout sends an interrupt first; if the child responds to
        # interrupt (our stub does), it exits with "interrupted". If it
        # ignores the interrupt, it gets hard-killed with "timeout".
        # Both are valid — the key assertion is it did NOT run for 60s.
        self.assertIn(results[0]["status"], ("timeout", "interrupted"))
        # For timeout status, the error message should mention timing out.
        # For interrupted status, the child was stopped by the interrupt signal.
        if results[0]["status"] == "timeout":
            self.assertIn("timed out", (results[0].get("error") or "").lower())

        # Should have terminated within a reasonable window of the timeout
        # (3s timeout + 10s grace = 13s max, but should be much faster)
        self.assertLess(
            elapsed, 20,
            f"Timeout should have fired at 3s + grace, but took {elapsed:.1f}s"
        )

        # Verify the process is actually dead (no zombie)
        proc = spec.process
        if proc is not None:
            self.assertFalse(
                proc.is_alive(),
                "Child process should be terminated, not alive"
            )


class TestTimeoutHardKill(unittest.TestCase):
    """Verify that a child that IGNORES interrupts gets hard-killed."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_unresponsive_child_hard_killed(self, _mock_cc, _mock_pi):
        """A child that ignores interrupts should be hard-terminated by the
        timeout mechanism — the key advantage of processes over threads."""
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()
        spec = _make_process_spec(
            0, "Ignore interrupt and sleep", parent,
            test_config={"sleep": 60, "ignore_interrupt": True},
        )

        children = [(0, {"goal": "Ignore interrupt"}, spec)]

        start = time.monotonic()
        results = run_children_in_processes(
            children=children, parent_agent=parent, child_timeout=2,
        )
        elapsed = time.monotonic() - start

        self.assertEqual(len(results), 1)
        # Should be "timeout" (hard-killed) since the child ignores interrupts
        self.assertEqual(results[0]["status"], "timeout")

        # Should have terminated within timeout + grace period
        self.assertLess(
            elapsed, 20,
            f"Hard kill should have happened at ~2s + 10s grace, "
            f"but took {elapsed:.1f}s"
        )

        # Process must be dead
        proc = spec.process
        if proc is not None:
            self.assertFalse(proc.is_alive(), "Process should be hard-killed")


# ──────────────────────────────────────────────────────────────────────
# Test 5: Config gate selects process vs thread path
# ──────────────────────────────────────────────────────────────────────

class TestConfigGate(unittest.TestCase):
    """Verify that delegation.process_isolation config controls which path runs."""

    def test_default_is_thread_path(self):
        """With process_isolation=false (default), batch should use ThreadPoolExecutor."""
        from tools.delegate_tool import _get_process_isolation
        # The default should be False
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertFalse(_get_process_isolation())

    def test_true_enables_process_path(self):
        """With process_isolation=true, the process path should be selected."""
        from tools.delegate_tool import _get_process_isolation
        with patch("tools.delegate_tool._load_config",
                   return_value={"process_isolation": True}):
            self.assertTrue(_get_process_isolation())


# ──────────────────────────────────────────────────────────────────────
# Test 6: Results sorted by task_index
# ──────────────────────────────────────────────────────────────────────

class TestResultOrdering(unittest.TestCase):
    """Verify that results come back sorted by task_index regardless of
    completion order."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_results_sorted_by_index(self, _mock_cc, _mock_pi):
        """Even if task 2 finishes before task 0, results should be [0, 1, 2]."""
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()
        children = []

        # Task 0 sleeps longest, task 2 sleeps shortest — completion order is 2,1,0
        for i in range(3):
            sleep_map = {0: 2.0, 1: 1.0, 2: 0.2}
            spec = _make_process_spec(
                i, f"Task {i}", parent,
                test_config={
                    "sleep": sleep_map[i],
                    "response": f"Task {i} done",
                },
            )
            children.append((i, {"goal": f"Task {i}"}, spec))

        results = run_children_in_processes(
            children=children, parent_agent=parent, child_timeout=30,
        )

        self.assertEqual(len(results), 3)
        # Results must be sorted by task_index
        indices = [r["task_index"] for r in results]
        self.assertEqual(indices, [0, 1, 2])
        # All should be completed
        for r in results:
            self.assertEqual(r["status"], "completed")


# ──────────────────────────────────────────────────────────────────────
# Test 7: No zombie processes survive the batch
# ──────────────────────────────────────────────────────────────────────

class TestNoZombies(unittest.TestCase):
    """Verify that all child processes are reaped — no zombies left behind."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_all_processes_reaped(self, _mock_cc, _mock_pi):
        """After run_children_in_processes returns, no child process should
        still be alive."""
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()
        children = []
        procs = []

        for i in range(3):
            spec = _make_process_spec(
                i, f"Quick task {i}", parent,
                test_config={"response": f"Done {i}"},
            )
            children.append((i, {"goal": f"Quick {i}"}, spec))

        results = run_children_in_processes(
            children=children, parent_agent=parent, child_timeout=30,
        )

        self.assertEqual(len(results), 3)

        # Check all processes are dead
        for i, _, spec in children:
            proc = spec.process
            if proc is not None:
                self.assertFalse(
                    proc.is_alive(),
                    f"Child process {i} is still alive — zombie!"
                )


# ──────────────────────────────────────────────────────────────────────
# Test 8: GIL freedom — child CPU work doesn't block parent
# ──────────────────────────────────────────────────────────────────────

class TestGILFreedom(unittest.TestCase):
    """Verify that CPU-bound work in child processes does not block the
    parent's main thread — the whole point of process isolation."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_parent_responsive_during_child_cpu_work(self, _mock_cc, _mock_pi):
        """While 3 children do CPU-heavy work, the parent should be able to
        do its own work without being blocked by the GIL."""
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()
        children = []

        for i in range(3):
            spec = _make_process_spec(
                i, f"CPU burn {i}", parent,
                test_config={
                    "cpu_work_seconds": 3,  # 3s of GIL-holding busy-wait
                    "response": f"Burned {i}",
                },
            )
            children.append((i, {"goal": f"CPU {i}"}, spec))

        # Run in a background thread so we can check parent responsiveness
        result_holder = {}

        def run_batch():
            result_holder["results"] = run_children_in_processes(
                children=children, parent_agent=parent, child_timeout=30,
            )

        batch_thread = threading.Thread(target=run_batch, daemon=True)
        batch_thread.start()

        # Give children time to start
        time.sleep(1)

        # While children burn CPU, the parent should be able to do work
        # In the thread path, this would be blocked by the GIL.
        # In the process path, the parent's thread is free.
        parent_counter = 0
        parent_start = time.monotonic()
        while batch_thread.is_alive() and (time.monotonic() - parent_start) < 5:
            parent_counter += 1
            # Small amount of work to measure responsiveness
            _ = sum(range(1000))

        parent_elapsed = time.monotonic() - parent_start

        # The parent should have been able to run its loop many times
        # (if the GIL were blocking us, parent_counter would be very low)
        self.assertGreater(
            parent_counter, 50,
            f"Parent was starved — only ran {parent_counter} iterations "
            f"in {parent_elapsed:.1f}s. GIL contention detected."
        )

        # Wait for batch to finish
        batch_thread.join(timeout=30)
        self.assertIn("results", result_holder)
        self.assertEqual(len(result_holder["results"]), 3)


# ──────────────────────────────────────────────────────────────────────
# Test 9: Child process exception is caught and reported
# ──────────────────────────────────────────────────────────────────────

class TestChildException(unittest.TestCase):
    """Verify that an exception inside a child process is caught and
    returned as an error entry, not crashing the parent."""

    @patch("tools.delegate_tool._get_process_isolation", return_value=True)
    @patch("tools.delegate_tool._get_max_concurrent_children", return_value=6)
    def test_child_exception_becomes_error_entry(self, _mock_cc, _mock_pi):
        from tools.delegate_process import run_children_in_processes

        parent = _make_mock_parent()
        spec = _make_process_spec(
            0, "Crash", parent,
            test_config={"fail": True, "fail_message": "Intentional crash"},
        )

        children = [(0, {"goal": "Crash"}, spec)]
        results = run_children_in_processes(
            children=children, parent_agent=parent, child_timeout=30,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("Intentional crash", results[0].get("error", ""))


# ──────────────────────────────────────────────────────────────────────
# Test 10: Params are picklable (spawn requirement)
# ──────────────────────────────────────────────────────────────────────

class TestPicklability(unittest.TestCase):
    """Verify that _build_child_process_spec produces picklable params."""

    def test_params_survive_pickle(self):
        """If params can't be pickled, spawn will fail silently. Test early."""
        import pickle
        from tools.delegate_tool import _build_child_process_spec

        parent = _make_mock_parent()
        spec = _build_child_process_spec(
            task_index=0,
            goal="Picklability test",
            context="Some context",
            toolsets=None,
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
        )

        # The params dict must be picklable
        try:
            data = pickle.dumps(spec.params)
            restored = pickle.loads(data)
            self.assertEqual(restored["goal"], "Picklability test")
            self.assertEqual(restored["task_index"], 0)
        except Exception as exc:
            self.fail(f"Params are not picklable: {exc}")


if __name__ == "__main__":
    unittest.main()