#!/usr/bin/env python3
"""Tests for the cross-agent FileStateRegistry (tools/file_state.py).

Covers the layers added for safe concurrent subagent file edits:

  1. Cross-agent staleness detection via ``check_stale``
  2. Per-path serialization via ``lock_path``
  3. Delegate-completion reminder via ``writes_since``
  4. Conversation lineage: a task-id rebind inside ONE conversation (context-compression
     rotation, surface switch delegation id -> session id) is not a sibling subagent

Plus integration through the real ``read_file_tool`` / ``write_file_tool``
handlers so the full hook wiring is exercised.

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

    # ── Conversation lineage: one conversation, several task ids ────────────────────
    #
    # A long-lived conversation legitimately runs under more than one task id: a context
    # compression rotates the session (a delegation id keeps running, the gateway surface
    # runs the new session's id), a surface switch moves a ``sa-…`` delegated session onto
    # its transport's session-scoped id. The recorded writer is then the SAME agent under
    # its earlier id — calling that "sibling subagent" (and the guard's refusal that goes
    # with it) misreads the bookkeeping of the guard itself as a second agent.

    def test_task_id_rebind_in_one_conversation_is_not_a_sibling(self):
        """Same conversation, new task id, current bytes never seen: still refused, but as
        this agent's own unseen content — never as a sibling's write."""
        p = self._mk()
        session = "20260924_213135_8c7c8e"
        file_state.note_write("sa-0-d2839be9", p, session_id=session)

        warn = file_state.check_stale(session, p, session_id=session) or ""

        self.assertTrue(warn, "unseen current content must still be refused")
        self.assertNotIn("sibling subagent", warn)
        self.assertIn("was not read by this agent", warn)

    def test_task_id_rebind_after_re_read_does_not_block(self):
        """The continuation re-read the current bytes: nothing left to refuse."""
        p = self._mk()
        session = "20260924_213135_8c7c8e"
        file_state.note_write("sa-0-d2839be9", p, session_id=session)
        file_state.record_read(session, p, session_id=session)

        self.assertIsNone(file_state.check_stale(session, p, session_id=session))

    def test_task_id_rebind_read_before_the_write_reports_the_content_not_a_sibling(self):
        """The conversation read the file, then an earlier id of the SAME conversation wrote
        it: the refusal must cite the content that changed, not a phantom sibling."""
        p = self._mk()
        session = "20260924_213135_8c7c8e"
        file_state.record_read("earlier-turn", p, session_id=session)
        # The lane rewrote the file minutes later. Bump the mtime explicitly: the guard's
        # content signal is the mtime, and coarse (1 s) filesystems would not move it for a
        # rewrite that lands in the same second.
        with open(p, "w") as f:
            f.write("revised by the lane\n")
        later = os.path.getmtime(p) + 60.0
        os.utime(p, (later, later))
        file_state.note_write("sa-0-d2839be9", p, session_id=session)

        warn = file_state.check_stale("earlier-turn", p, session_id=session) or ""

        self.assertTrue(warn, "bytes that changed after our read must still be refused")
        self.assertNotIn("sibling subagent", warn)
        self.assertIn("modified since you last read it", warn)

    def test_foreign_conversation_still_gets_the_sibling_refusal(self):
        """The protection this layer exists for: a writer from ANOTHER conversation — a real
        concurrent subagent — keeps the sibling refusal, session ids or not."""
        p = self._mk()
        file_state.note_write("sa-0-e3259de9", p, session_id="20260924_211103_6fb9c5")

        warn = file_state.check_stale("20260924_213135_8c7c8e", p,
                                      session_id="20260924_213135_8c7c8e") or ""

        self.assertIn("sibling subagent 'sa-0-e3259de9'", warn)

    def test_writer_without_a_known_conversation_stays_a_sibling(self):
        """Conservative default: without a session id on both sides there is no evidence the
        writer is the same agent, so the refusal is unchanged."""
        p = self._mk()
        file_state.note_write("sa-0-unknown-session", p)  # no session_id recorded

        warn = file_state.check_stale("20260924_213135_8c7c8e", p,
                                      session_id="20260924_213135_8c7c8e") or ""

        self.assertIn("sibling subagent 'sa-0-unknown-session'", warn)

    def test_recorded_conversation_lets_a_later_check_recognize_it_without_the_session_id(self):
        """The conversation a task ran in is remembered by reads/writes, so a check that
        does not repeat the session id still knows whose write it is looking at."""
        p = self._mk()
        session = "20260924_213135_8c7c8e"
        file_state.record_read("continuation", p, session_id=session)
        file_state.note_write("sa-0-d2839be9", p, session_id=session)

        warn = file_state.check_stale("continuation", p)  # no session_id argument

        self.assertNotIn("sibling subagent", warn or "")


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

    def test_rebound_task_id_in_one_conversation_is_not_a_sibling_through_handler(self):
        """The incident this guards against: the lane ``sa-0-d2839be9`` wrote the file, the
        same conversation continued under its session-scoped task id and was told a sibling
        subagent had modified the file. It must be refused for what it is — content this task
        id has not seen — and never labelled a sibling."""
        session = "20260924_213135_8c7c8e"
        p = os.path.join(self._tmpdir, "narrative_delta.py")
        w_lane = json.loads(write_file_tool(path=p, content="lane wrote this\n",
                                            task_id="sa-0-d2839be9", session_id=session))
        self.assertNotIn("error", w_lane)

        w_cont = json.loads(write_file_tool(path=p, content="continuation retried\n",
                                            task_id=session, session_id=session))
        err = w_cont.get("error", "")
        self.assertTrue(w_cont.get("stale_write_blocked"))
        self.assertNotIn("sibling subagent", err)
        self.assertIn("was not read by this agent", err)
        # The refusal protected the lane's bytes.
        with open(p) as f:
            self.assertEqual(f.read(), "lane wrote this\n")

    def test_rebound_task_id_after_re_read_writes_through_handler(self):
        """Once the continuation has seen the current content, the write goes through — no
        phantom sibling stands in its way."""
        session = "20260924_213135_8c7c8e"
        p = os.path.join(self._tmpdir, "probe_narrative_delta.py")
        self.assertNotIn("error", json.loads(write_file_tool(
            path=p, content="v1\n", task_id="sa-0-d2839be9", session_id=session)))

        r = json.loads(read_file_tool(path=p, task_id=session, session_id=session))
        self.assertNotIn("error", r)

        w = json.loads(write_file_tool(path=p, content="v2\n", task_id=session,
                                       session_id=session))
        self.assertNotIn("error", w)
        with open(p) as f:
            self.assertEqual(f.read(), "v2\n")

    def test_foreign_conversation_write_refuses_through_handler(self):
        """A writer from another conversation keeps the full sibling refusal — the guard's
        protection does not depend on the label change."""
        p = os.path.join(self._tmpdir, "shared_scratch.py")
        self.assertNotIn("error", json.loads(write_file_tool(
            path=p, content="sibling wrote this\n",
            task_id="sa-0-e3259de9", session_id="20260924_211103_6fb9c5")))

        w = json.loads(write_file_tool(path=p, content="clobber\n",
                                       task_id="20260924_213135_8c7c8e",
                                       session_id="20260924_213135_8c7c8e"))
        err = w.get("error", "")
        self.assertTrue(w.get("stale_write_blocked"))
        self.assertIn("sibling subagent 'sa-0-e3259de9'", err)
        with open(p) as f:
            self.assertEqual(f.read(), "sibling wrote this\n")




if __name__ == "__main__":
    unittest.main()
