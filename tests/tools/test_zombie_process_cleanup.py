"""Tests for zombie process cleanup — verifies processes spawned by tools
are properly reaped when agent sessions end.

Reproduction for issue #7131: zombie process accumulation on long-running
gateway deployments.
"""

import os
import signal
import subprocess
import sys
import threading



def _spawn_sleep(seconds: float = 60) -> subprocess.Popen:
    """Spawn a portable long-lived Python sleep process (no shell wrapper)."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
    )


def _pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class TestZombieReproduction:
    """Demonstrate that subprocesses survive when cleanup is not called."""

    def test_orphaned_processes_survive_without_cleanup(self):
        """REPRODUCTION: processes spawned directly survive if no one kills
        them — this models the gap that causes zombie accumulation when
        the gateway drops agent references without calling close()."""
        pids = []

        try:
            for _ in range(3):
                proc = _spawn_sleep(60)
                pids.append(proc.pid)

            for pid in pids:
                assert _pid_alive(pid), f"PID {pid} should be alive after spawn"

            # Simulate "session end" by just dropping the reference
            del proc  # noqa: F821

            # BUG: processes are still alive after reference is dropped
            for pid in pids:
                assert _pid_alive(pid), (
                    f"PID {pid} died after ref drop — "
                    f"expected it to survive (demonstrating the bug)"
                )
        finally:
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    def test_explicit_terminate_reaps_processes(self):
        """Explicitly terminating+waiting on Popen handles works.
        This models what ProcessRegistry.kill_process does internally."""
        procs = []

        try:
            for _ in range(3):
                proc = _spawn_sleep(60)
                procs.append(proc)

            for proc in procs:
                assert _pid_alive(proc.pid)

            for proc in procs:
                proc.terminate()
                proc.wait(timeout=5)

            for proc in procs:
                assert proc.returncode is not None, (
                    f"PID {proc.pid} should have exited after terminate+wait"
                )
        finally:
            for proc in procs:
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass


class TestAgentCloseMethod:
    """Verify AIAgent.close() exists, is idempotent, and calls cleanup."""

    def test_close_calls_cleanup_functions(self):
        """close() should call kill_all, cleanup_vm, cleanup_browser."""
        from unittest.mock import patch

        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.session_id = "test-close-cleanup"
            agent._active_children = []
            agent._active_children_lock = threading.Lock()
            agent.client = None

            with patch("tools.process_registry.process_registry") as mock_registry, \
                 patch("run_agent.cleanup_vm") as mock_cleanup_vm, \
                 patch("run_agent.cleanup_browser") as mock_cleanup_browser:
                agent.close()

                mock_registry.kill_all.assert_called_once_with(
                    task_id="test-close-cleanup"
                )
                mock_cleanup_vm.assert_called_once_with("test-close-cleanup")
                mock_cleanup_browser.assert_called_once_with("test-close-cleanup")

    def test_close_is_idempotent(self):
        """close() can be called multiple times without error."""
        from unittest.mock import patch

        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.session_id = "test-close-idempotent"
            agent._active_children = []
            agent._active_children_lock = threading.Lock()
            agent.client = None

            agent.close()
            agent.close()
            agent.close()

    def test_close_propagates_to_children(self):
        """close() should call close() on all active child agents."""
        from unittest.mock import MagicMock, patch

        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.session_id = "test-close-children"
            agent._active_children_lock = threading.Lock()
            agent.client = None

            child_1 = MagicMock()
            child_2 = MagicMock()
            agent._active_children = [child_1, child_2]

            agent.close()

            child_1.close.assert_called_once()
            child_2.close.assert_called_once()
            assert agent._active_children == []

    def test_close_ends_owned_session_row(self):
        """close() finalizes the agent's owned SQLite session row."""
        from unittest.mock import MagicMock, patch

        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.session_id = "test-close-session-row"
            agent._active_children = []
            agent._active_children_lock = threading.Lock()
            agent.client = None
            agent._end_session_on_close = True
            agent._session_db = MagicMock()

            agent.close()

            agent._session_db.end_session.assert_called_once_with(
                "test-close-session-row", "agent_close"
            )

    def test_close_skips_session_end_for_forwarded_continuation_agents(self):
        """Helper agents that handed session ownership forward opt out."""
        from unittest.mock import MagicMock, patch

        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.session_id = "test-close-forwarded-session"
            agent._active_children = []
            agent._active_children_lock = threading.Lock()
            agent.client = None
            agent._end_session_on_close = False
            agent._session_db = MagicMock()

            agent.close()

            agent._session_db.end_session.assert_not_called()

    def test_close_session_end_noops_without_session_db(self):
        """close() is a no-op for session finalization when no DB is wired in."""
        from unittest.mock import patch

        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.session_id = "test-close-no-db"
            agent._active_children = []
            agent._active_children_lock = threading.Lock()
            agent.client = None
            # No _session_db / _end_session_on_close attributes at all —
            # getattr defaults must keep close() from raising.
            agent.close()  # must not raise

    def test_close_survives_partial_failures(self):
        """close() continues cleanup even if one step fails."""
        from unittest.mock import patch

        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.session_id = "test-close-partial"
            agent._active_children = []
            agent._active_children_lock = threading.Lock()
            agent.client = None

            with patch(
                "tools.process_registry.process_registry"
            ) as mock_reg, patch(
                "run_agent.cleanup_vm"
            ) as mock_vm, patch(
                "run_agent.cleanup_browser"
            ) as mock_browser:
                mock_reg.kill_all.side_effect = RuntimeError("boom")

                agent.close()

                mock_vm.assert_called_once()
                mock_browser.assert_called_once()


class TestGatewayCleanupWiring:
    """Verify gateway lifecycle calls close() on agents."""

    def test_gateway_stop_calls_close(self):
        """gateway stop() should call close() on all running agents."""
        import asyncio
        import threading
        from unittest.mock import MagicMock, patch

        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._running = True
        runner._running_agents = {}
        runner._running_agents_ts = {}
        runner.adapters = {}
        runner._background_tasks = set()
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._pending_model_notes = {}
        runner._shutdown_event = asyncio.Event()
        runner._exit_reason = None
        runner._exit_code = None
        runner._stop_task = None
        runner._draining = False
        runner._restart_requested = False
        runner._restart_task_started = False
        runner._restart_detached = False
        runner._restart_via_service = False
        runner._restart_drain_timeout = 0.1
        runner._voice_mode = {}
        runner._session_model_overrides = {}
        runner._update_prompt_pending = {}
        runner._busy_input_mode = "interrupt"
        runner._agent_cache = {}
        runner._agent_cache_lock = threading.Lock()
        runner._shutdown_all_gateway_honcho = lambda: None
        runner._update_runtime_status = MagicMock()

        mock_agent_1 = MagicMock()
        mock_agent_2 = MagicMock()
        runner._running_agents = {
            "session-1": mock_agent_1,
            "session-2": mock_agent_2,
        }

        loop = asyncio.new_event_loop()
        try:
            with patch("gateway.status.remove_pid_file"), \
                 patch("gateway.status.write_runtime_status"), \
                 patch("tools.terminal_tool.cleanup_all_environments"), \
                 patch("tools.browser_tool.cleanup_all_browsers"):
                loop.run_until_complete(GatewayRunner.stop(runner))
        finally:
            loop.close()

        mock_agent_1.close.assert_called()
        mock_agent_2.close.assert_called()

    def test_evict_does_not_call_close(self):
        """_evict_cached_agent() should NOT call close() — it's also used
        for non-destructive refreshes (model switch, branch, fallback)."""
        import threading
        from unittest.mock import MagicMock

        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._agent_cache_lock = threading.Lock()

        mock_agent = MagicMock()
        runner._agent_cache = {"session-key": (mock_agent, 12345)}

        GatewayRunner._evict_cached_agent(runner, "session-key")

        mock_agent.close.assert_not_called()
        assert "session-key" not in runner._agent_cache


class TestDelegationCleanup:
    """Verify subagent delegation cleans up child agents."""

    def test_run_single_child_calls_close(self):
        """_run_single_child finally block should call close() on child."""
        from unittest.mock import MagicMock
        from tools.delegate_tool import _run_single_child

        parent = MagicMock()
        parent._active_children = []
        parent._active_children_lock = threading.Lock()

        child = MagicMock()
        child._delegate_saved_tool_names = ["tool1"]
        child.run_conversation.side_effect = RuntimeError("test abort")

        parent._active_children.append(child)

        result = _run_single_child(
            task_index=0,
            goal="test goal",
            child=child,
            parent_agent=parent,
        )

        child.close.assert_called_once()
        assert child not in parent._active_children
        assert result["status"] == "error"

    def test_run_single_child_binds_parent_ui_return_address_in_worker(self):
        """The child's executor thread must retain the commissioning WebUI tab."""
        from unittest.mock import MagicMock

        from gateway.session_context import get_session_env
        from tools.delegate_tool import _run_single_child

        observed = {}
        parent = MagicMock()
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent._current_task_id = None

        child = MagicMock()
        child._credential_pool = None
        child._delegate_saved_tool_names = ["tool1"]
        child._origin_ui_session_id = "parent-tab"
        child._subagent_id = None
        child.tool_progress_callback = None

        def run_conversation(**_kwargs):
            observed["origin_ui_session_id"] = get_session_env(
                "HERMES_UI_SESSION_ID", ""
            )
            raise RuntimeError("stop after capture")

        child.run_conversation.side_effect = run_conversation
        parent._active_children.append(child)

        result = _run_single_child(
            task_index=0,
            goal="test goal",
            child=child,
            parent_agent=parent,
        )

        assert observed["origin_ui_session_id"] == "parent-tab"
        assert result["status"] == "error"

    def test_delegate_task_preserves_ui_origin_across_outer_and_inner_executors(
        self, monkeypatch
    ):
        """A real background delegate path stamps its process completion for the parent tab."""
        import json
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from gateway.session_context import bind_ui_session_id, get_session_env
        from tools import delegate_tool
        from tools.process_registry import ProcessRegistry, ProcessSession

        registry = ProcessRegistry()
        registry._write_checkpoint = MagicMock()
        observed = {}

        def run_conversation(**_kwargs):
            worker_origin = get_session_env("HERMES_UI_SESSION_ID", "")
            observed["worker_origin"] = worker_origin
            process = ProcessSession(
                id="proc_delegate_origin",
                command="printf done",
                session_key="child-execution-session",
                origin_ui_session_id=worker_origin,
                notify_on_complete=True,
                output_buffer="done",
            )
            registry._running[process.id] = process
            process.exited = True
            process.exit_code = 0
            registry._move_to_finished(process)
            return {
                "final_response": "child finished",
                "completed": True,
                "api_calls": 1,
                "messages": [],
            }

        child = SimpleNamespace(
            _credential_pool=None,
            _delegate_role="leaf",
            _subagent_id=None,
            tool_progress_callback=None,
            run_conversation=run_conversation,
            get_activity_summary=lambda: {"api_call_count": 1},
            close=MagicMock(),
            model="test-model",
            session_id="child-durable-session",
            session_prompt_tokens=0,
            session_completion_tokens=0,
            session_reasoning_tokens=0,
            session_estimated_cost_usd=0.0,
        )
        parent = SimpleNamespace(
            _delegate_depth=0,
            _active_children=[],
            _active_children_lock=threading.Lock(),
            _current_task_id=None,
            _interrupt_requested=False,
            _memory_manager=None,
            session_id="parent-durable-session",
            session_estimated_cost_usd=0.0,
            session_cost_source="none",
            session_cost_status="unknown",
            _touch_activity=lambda *_args, **_kwargs: None,
        )

        monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"max_iterations": 1})
        monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 2)
        monkeypatch.setattr(
            delegate_tool,
            "_resolve_delegation_credentials",
            lambda *_args, **_kwargs: {
                "model": None,
                "provider": None,
                "base_url": None,
                "api_key": None,
                "api_mode": None,
                "command": None,
                "args": None,
            },
        )
        monkeypatch.setattr(delegate_tool, "_build_child_agent", lambda **_kwargs: child)
        monkeypatch.setattr(
            "tools.approval.get_current_session_key",
            lambda default="": "parent-durable-session",
        )

        def dispatch_across_outer_executor(*, runner, origin_ui_session_id, **_kwargs):
            observed["dispatch_origin"] = origin_ui_session_id
            with ThreadPoolExecutor(max_workers=1) as executor:
                observed["outer_result"] = executor.submit(runner).result(timeout=5)
            return {"status": "dispatched", "delegation_id": "deleg_origin_test"}

        monkeypatch.setattr(
            "tools.async_delegation.dispatch_async_delegation_batch",
            dispatch_across_outer_executor,
        )

        with bind_ui_session_id("parent-tab"):
            payload = json.loads(
                delegate_tool.delegate_task(
                    goal="launch a notified process",
                    background=True,
                    parent_agent=parent,
                )
            )

        completion = registry.completion_queue.get_nowait()
        assert payload["status"] == "dispatched"
        assert observed["dispatch_origin"] == "parent-tab"
        assert observed["worker_origin"] == "parent-tab"
        assert observed["outer_result"]["results"][0]["status"] == "completed"
        assert completion["session_key"] == "child-execution-session"
        assert completion["origin_ui_session_id"] == "parent-tab"
        child.close.assert_called_once()
