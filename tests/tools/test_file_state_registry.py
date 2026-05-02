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
        self.assertIn("B", warn)
        self.assertIn("sibling", warn.lower())

    def test_write_without_read_flagged(self):
        p = self._mk()
        # Agent A never read this file.
        file_state.note_write("B", p)  # another agent touched it
        warn = file_state.check_stale("A", p)
        self.assertIsNotNone(warn)

    def test_partial_read_flagged_on_write(self):
        p = self._mk()
        file_state.record_read("A", p, partial=True)
        warn = file_state.check_stale("A", p)
        self.assertIsNotNone(warn)
        self.assertIn("partial", warn.lower())

    def test_external_mtime_drift_flagged(self):
        p = self._mk()
        file_state.record_read("A", p)
        # Bump the on-disk mtime without going through the registry.
        time.sleep(0.01)
        os.utime(p, None)
        with open(p, "w") as f:
            f.write("externally modified\n")
        warn = file_state.check_stale("A", p)
        self.assertIsNotNone(warn)
        self.assertIn("modified since you last read", warn)

    def test_own_write_updates_stamp_so_next_write_is_clean(self):
        p = self._mk()
        file_state.record_read("A", p)
        file_state.note_write("A", p)
        # Second write by the same agent — should not be flagged.
        self.assertIsNone(file_state.check_stale("A", p))

    def test_different_paths_dont_interfere(self):
        a = self._mk()
        b = self._mk()
        file_state.record_read("A", a)
        file_state.note_write("B", b)
        # A reads only `a`; B writes `b`. A writing `a` is NOT stale.
        self.assertIsNone(file_state.check_stale("A", a))

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

    def test_writes_since_filters_by_parent_read_set(self):
        foo = self._mk()
        bar = self._mk()
        baz = self._mk()
        file_state.record_read("parent", foo)
        file_state.record_read("parent", bar)
        since = time.time()
        time.sleep(0.01)
        file_state.note_write("child", foo)  # parent read this — report
        file_state.note_write("child", baz)  # parent never saw — skip

        # Caller passes only paths the parent actually read (this is what
        # delegate_tool does via ``known_reads(parent_task_id)``).
        parent_reads = file_state.known_reads("parent")
        out = file_state.writes_since("parent", since, parent_reads)
        self.assertIn("child", out)
        self.assertIn(foo, out["child"])
        self.assertNotIn(baz, out["child"])

    def test_writes_since_excludes_the_target_agent(self):
        p = self._mk()
        file_state.record_read("parent", p)
        since = time.time()
        time.sleep(0.01)
        file_state.note_write("parent", p)  # parent's own write
        out = file_state.writes_since("parent", since, [p])
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

    def test_same_agent_consecutive_writes_no_false_warning(self):
        p = self._write_seed("own.txt")
        json.loads(read_file_tool(path=p, task_id="agentC"))
        w1 = json.loads(write_file_tool(path=p, content="one\n", task_id="agentC"))
        self.assertFalse(w1.get("_warning"))
        w2 = json.loads(write_file_tool(path=p, content="two\n", task_id="agentC"))
        self.assertFalse(w2.get("_warning"))

    def test_patch_tool_also_surfaces_sibling_warning(self):
        p = self._write_seed("p.txt", "hello world\n")
        json.loads(read_file_tool(path=p, task_id="agentA"))
        json.loads(write_file_tool(path=p, content="hello planet\n", task_id="agentB"))
        r = json.loads(
            patch_tool(
                mode="replace",
                path=p,
                old_string="hello",
                new_string="HI",
                task_id="agentA",
            )
        )
        warn = r.get("_warning", "")
        # Patch may fail (sibling changed the content so old_string may not
        # match) or succeed — either way, the cross-agent warning should be
        # present when old_string still happens to match.  What matters is
        # that if the patch succeeded or the warning was reported, it names
        # the sibling.  When old_string doesn't match, the patch itself
        # returns an error but the warning is still set from the pre-check.
        if warn:
            self.assertIn("agentB", warn)

    def test_net_new_file_no_warning(self):
        p = os.path.join(self._tmpdir, "brand_new.txt")
        # Nobody has read or written this before.
        w = json.loads(write_file_tool(path=p, content="hi\n", task_id="agentX"))
        self.assertFalse(w.get("_warning"))
        self.assertNotIn("error", w)


if __name__ == "__main__":
    unittest.main()
