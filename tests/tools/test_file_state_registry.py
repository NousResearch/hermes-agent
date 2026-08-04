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

from tools import file_state
from tools.file_tools import (
    read_file_tool,
    write_file_tool,
    patch_tool,
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
        assert isinstance(warn, str)
        self.assertIn("B", warn)
        self.assertIn("sibling", warn.lower())

    def test_remote_write_records_without_host_stat(self):
        remote_path = "/remote-only/project/file.txt"
        namespace = "ssh:physical-host"
        file_state.record_read(
            "A", remote_path, namespace=namespace, stat_path=False,
        )
        time.sleep(0.01)
        file_state.note_write(
            "B", remote_path, namespace=namespace, stat_path=False,
        )
        warn = file_state.check_stale("A", remote_path, namespace=namespace)
        self.assertIsNotNone(warn)
        assert isinstance(warn, str)
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

    def test_delegate_state_aggregates_task_targets_without_cross_target_conflicts(self):
        path = self._mk()
        file_state.record_read(("parent", "alpha"), path, namespace="alpha")
        file_state.record_read(("parent", "beta"), path, namespace="beta")
        since = time.time()
        time.sleep(0.01)
        file_state.note_write(("child", "alpha"), path, namespace="alpha")

        parent_reads = file_state.known_reads("parent")

        self.assertIn(f"alpha\0{path}", parent_reads)
        self.assertIn(f"beta\0{path}", parent_reads)
        out = file_state.writes_since("parent", since, [f"alpha\0{path}"])
        self.assertEqual(out, {"child": [path]})

    def test_writes_since_excludes_all_scopes_of_raw_parent_task(self):
        path = self._mk()
        file_state.record_read(("parent", "alpha"), path, namespace="alpha")
        since = time.time()
        time.sleep(0.01)
        file_state.note_write(("parent", "alpha"), path, namespace="alpha")

        out = file_state.writes_since(
            "parent", since, file_state.known_reads("parent"),
        )

        self.assertEqual(out, {})


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

    def test_sibling_agent_write_surfaces_warning_through_handler(self):
        p = self._write_seed("shared.txt")
        r = json.loads(read_file_tool(path=p, task_id="agentA"))
        self.assertNotIn("error", r)

        w_b = json.loads(write_file_tool(path=p, content="B wrote\n", task_id="agentB"))
        self.assertNotIn("error", w_b)

        w_a = json.loads(write_file_tool(path=p, content="A stale\n", task_id="agentA"))
        warn = w_a.get("_warning", "")
        self.assertTrue(warn, f"expected warning, got: {w_a}")
        # The cross-agent message names the sibling task_id.
        self.assertIn("agentB", warn)
        self.assertIn("sibling", warn.lower())


    def test_net_new_file_no_warning(self):
        p = os.path.join(self._tmpdir, "brand_new.txt")
        # Nobody has read or written this before.
        w = json.loads(write_file_tool(path=p, content="hi\n", task_id="agentX"))
        self.assertFalse(w.get("_warning"))
        self.assertNotIn("error", w)


if __name__ == "__main__":
    unittest.main()
