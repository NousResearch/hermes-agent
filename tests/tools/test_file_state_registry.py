#!/usr/bin/env python3
"""Tests for the cross-agent FileStateRegistry (tools/file_state.py).

Covers the three layers added for safe concurrent subagent file edits:

  1. Cross-agent staleness detection via ``check_stale``
  2. Per-path serialization via ``lock_path``
  3. Delegate-completion reminder via ``writes_since``

Plus integration through the real ``read_file_tool`` / ``write_file_tool``
/ ``patch_tool`` handlers so the full hook wiring is exercised.

Run:
    python -m pytest tests/tools/test_file_state_registry.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tools import file_state
from tools.file_tools import (
    clear_file_ops_cache,
    read_file_tool,
    write_file_tool,
)


def _tmp_file(content: str = "initial\n") -> str:
    fd, path = tempfile.mkstemp(prefix="hermes_file_state_test_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class FileStateRegistryUnitTests(unittest.TestCase):
    """Direct unit tests on the registry singleton."""

    def setUp(self) -> None:
        file_state.get_registry().clear()
        self._tmpfiles: list[str] = []

    def tearDown(self) -> None:
        for p in self._tmpfiles:
            try:
                os.unlink(p)
            except OSError:
                pass
        file_state.get_registry().clear()

    def _mk(self, content: str = "x\n") -> str:
        p = _tmp_file(content)
        self._tmpfiles.append(p)
        return p

    def test_record_read_then_check_stale_returns_none(self):
        p = self._mk()
        file_state.record_read("A", p)
        self.assertIsNone(file_state.check_stale("A", p))

    def test_sibling_write_flags_other_agent_as_stale(self):
        p = self._mk()
        file_state.record_read("A", p)
        # Simulate sibling writing this file later
        time.sleep(0.01)  # ensure ts ordering across resolution
        file_state.note_write("B", p)
        warn = file_state.check_stale("A", p)
        self.assertIsNotNone(warn)
        self.assertIn("B", warn)
        self.assertIn("sibling", warn.lower())


    def test_lock_path_serializes_same_path(self):
        p = self._mk()
        events: list[tuple[str, int]] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            with file_state.lock_path(p):
                with lock:
                    events.append(("enter", i))
                time.sleep(0.01)
                with lock:
                    events.append(("exit", i))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every enter must be immediately followed by its matching exit.
        self.assertEqual(len(events), 8)
        for i in range(0, 8, 2):
            self.assertEqual(events[i][0], "enter")
            self.assertEqual(events[i + 1][0], "exit")
            self.assertEqual(events[i][1], events[i + 1][1])

    def test_lock_path_is_per_path_not_global(self):
        a = self._mk()
        b = self._mk()
        b_entered = threading.Event()

        def hold_a() -> None:
            with file_state.lock_path(a):
                b_entered.wait(timeout=2.0)

        def enter_b() -> None:
            time.sleep(0.02)  # let A grab its lock
            with file_state.lock_path(b):
                b_entered.set()

        ta = threading.Thread(target=hold_a)
        tb = threading.Thread(target=enter_b)
        ta.start()
        tb.start()
        self.assertTrue(b_entered.wait(timeout=3.0))
        ta.join(timeout=3.0)
        tb.join(timeout=3.0)

    def test_lock_path_state_is_released_after_last_waiter(self):
        p = self._mk()
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        def first() -> None:
            with file_state.lock_path(p):
                first_entered.set()
                release_first.wait(timeout=2.0)

        def second() -> None:
            first_entered.wait(timeout=2.0)
            with file_state.lock_path(p):
                second_entered.set()

        ta = threading.Thread(target=first)
        tb = threading.Thread(target=second)
        ta.start()
        tb.start()
        self.assertTrue(first_entered.wait(timeout=2.0))
        time.sleep(0.02)
        self.assertFalse(second_entered.is_set())
        release_first.set()
        ta.join(timeout=3.0)
        tb.join(timeout=3.0)

        registry = file_state.get_registry()
        self.assertTrue(second_entered.is_set())
        self.assertNotIn(p, registry._path_locks)
        self.assertNotIn(p, registry._path_lock_users)

    def test_clear_file_ops_cache_releases_task_state(self):
        p = self._mk()
        task_id = "finished-task"
        file_state.record_read(task_id, p)

        from tools import file_tools_read_tracking as rt

        rt._read_tracker[task_id] = {"dedup": {}}
        rt._patch_failure_tracker[task_id] = {p: 2}

        clear_file_ops_cache(task_id)

        self.assertEqual(file_state.known_reads(task_id), [])
        self.assertNotIn(task_id, rt._read_tracker)
        self.assertNotIn(task_id, rt._patch_failure_tracker)

    def test_forget_task_clears_last_writer_claims(self):
        """A finished task is not a concurrent sibling: forget_task must drop its writer
        claims so the next run of the same job (fresh ``cron:<job>:<uuid>`` id) can write
        the same scratch path without a "modified by sibling subagent" refusal."""
        p = self._mk()
        file_state.note_write("cron:JOB:run1", p)
        file_state.get_registry().forget_task("cron:JOB:run1")

        self.assertIsNone(file_state.check_stale("cron:JOB:run2", p))
        # A sibling that has NOT ended still triggers the guard.
        file_state.note_write("subagent-1-live", p)
        self.assertIn("sibling subagent 'subagent-1-live'", file_state.check_stale("cron:JOB:run2", p))

    def test_agent_close_forgets_every_task_id_it_ran(self):
        """``AIAgent.close()`` receives the session_id, but file tools key the registry by
        the per-turn task_id (cron ``cron:<job>:<uuid>``, subagent ``subagent-N-xxxx``).
        close() must release the file state of every task id the agent ran."""
        p = self._mk()
        file_state.record_read("cron:JOB:run1", p)
        file_state.note_write("cron:JOB:run1", p)
        with patch("run_agent.AIAgent.__init__", return_value=None):
            from run_agent import AIAgent
            agent = AIAgent.__new__(AIAgent)
            agent.session_id = "cron_JOB_20260918_060000"
            agent._process_owner_task_ids = {"cron:JOB:run1"}
            agent._active_children = []
            agent._active_children_lock = threading.Lock()
            agent.client = None
            with patch("run_agent.cleanup_vm"), patch("run_agent.cleanup_browser"), \
                 patch("tools.computer_use.tool.release_computer_use_session"):
                agent.close()

        self.assertEqual(file_state.known_reads("cron:JOB:run1"), [])
        self.assertIsNone(file_state.check_stale("cron:JOB:run2", p))

    def test_kill_switch_env_var(self):
        p = self._mk()
        os.environ["HERMES_DISABLE_FILE_STATE_GUARD"] = "1"
        try:
            file_state.record_read("A", p)
            file_state.note_write("B", p)
            self.assertIsNone(file_state.check_stale("A", p))
            self.assertEqual(file_state.known_reads("A"), [])
            self.assertEqual(
                file_state.writes_since("A", 0.0, [p]),
                {},
            )
        finally:
            del os.environ["HERMES_DISABLE_FILE_STATE_GUARD"]


class FileToolsIntegrationTests(unittest.TestCase):
    """Integration through the real file_tools handlers.

    These exercise the wiring: read_file_tool → registry.record_read,
    write_file_tool / patch_tool → check_stale + lock_path + note_write.
    """

    def setUp(self) -> None:
        file_state.get_registry().clear()
        self._tmpdir = tempfile.mkdtemp(prefix="hermes_file_state_int_")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        file_state.get_registry().clear()

    def _write_seed(self, name: str, content: str = "seed\n") -> str:
        p = os.path.join(self._tmpdir, name)
        with open(p, "w") as f:
            f.write(content)
        return p

    def test_sibling_agent_write_refuses_stale_overwrite_through_handler(self):
        p = self._write_seed("shared.txt")
        r = json.loads(read_file_tool(path=p, task_id="agentA"))
        self.assertNotIn("error", r)

        self.assertNotIn("error", json.loads(read_file_tool(path=p, task_id="agentB")))
        w_b = json.loads(write_file_tool(path=p, content="B wrote\n", task_id="agentB"))
        self.assertNotIn("error", w_b)

        w_a = json.loads(write_file_tool(path=p, content="A stale\n", task_id="agentA"))
        err = w_a.get("error", "")
        self.assertTrue(w_a.get("stale_write_blocked"), f"expected stale write refusal, got: {w_a}")
        # The cross-agent message names the sibling task_id; B's write survives.
        self.assertIn("agentB", err)
        self.assertIn("sibling", err.lower())
        with open(p) as f:
            self.assertEqual(f.read(), "B wrote\n")

    def test_migrate_task_transfers_reads_and_writer_claims_without_self_sibling_block(self):
        from tools.file_tools import migrate_file_ops_state

        p = self._write_seed("rotated.txt", "v1\n")
        rp = os.path.realpath(p)
        self.assertNotIn("error", json.loads(read_file_tool(path=p, task_id="sess_parent")))
        self.assertNotIn("error", json.loads(write_file_tool(path=p, content="v2\n", task_id="sess_parent")))

        migrate_file_ops_state("sess_parent", "sess_child")
        self.assertIsNone(file_state.check_stale("sess_child", rp))
        self.assertIsNone(file_state.check_stale("sess_parent", rp))
        # Mid-turn write under effective_task_id (sess_parent) after rotation must not block sess_child
        w_mid = json.loads(write_file_tool(path=p, content="v2_mid\n", task_id="sess_parent"))
        self.assertNotIn("error", w_mid)
        self.assertNotIn("_warning", w_mid)
        self.assertIsNone(file_state.check_stale("sess_child", rp))
        w_child = json.loads(write_file_tool(path=p, content="v3\n", task_id="sess_child"))
        self.assertNotIn("error", w_child)
        self.assertNotIn("_warning", w_child)

    def test_terminal_lifecycle_cache_invalidation_preserves_write_baselines(self):
        from tools.terminal_tool_lifecycle import _clear_file_ops_cache

        p = self._write_seed("survive_reap.txt", "initial\n")
        self.assertNotIn("error", json.loads(read_file_tool(path=p, task_id="sess_active")))
        _clear_file_ops_cache("sess_active")

        w_after = json.loads(write_file_tool(path=p, content="updated after reap\n", task_id="sess_active"))
        self.assertNotIn("error", w_after)

    def test_persist_disabled_fork_cleans_up_without_polluting_parent(self):
        from types import SimpleNamespace
        from agent.chat_completion_helpers import cleanup_task_resources

        p = self._write_seed("fork_read.txt", "hello\n")
        self.assertNotIn("error", json.loads(read_file_tool(path=p, task_id="fork_uuid_1")))
        fork = SimpleNamespace(
            session_id="parent_sess_1",
            _persist_disabled=True,
            verbose_logging=False,
        )
        with patch("run_agent.cleanup_vm"), patch("run_agent.cleanup_browser"):
            cleanup_task_resources(fork, "fork_uuid_1")

        self.assertEqual(file_state.known_reads("fork_uuid_1"), [])
        self.assertEqual(file_state.known_reads("parent_sess_1"), [])

    def test_module_reload_preserves_registry_and_read_tracker_identity(self):
        import importlib
        import tools.file_tools_read_tracking as ftrt

        p = self._write_seed("reload_keep.py", "x = 1\n")
        self.assertNotIn("error", json.loads(read_file_tool(path=p, task_id="reload_sess")))
        reg_before = file_state.get_registry()
        tracker_before = ftrt._read_tracker

        importlib.reload(file_state)
        importlib.reload(ftrt)

        self.assertIs(file_state.get_registry(), reg_before)
        self.assertIs(ftrt._read_tracker, tracker_before)
        rp = os.path.realpath(p)
        self.assertIn(rp, file_state.known_reads("reload_sess"))
        self.assertIn(rp, ftrt._read_tracker.get("reload_sess", {}).get("full_write_baselines", {}))


if __name__ == "__main__":
    unittest.main()
