#!/usr/bin/env python3
"""Regression tests for the delegated-child session-key binding bug.

``delegation.worktree_isolation`` seeds the child's session-cwd record
correctly under ``child_task_id`` (``_ChildRun.seed_workspace()``,
tools/delegate_tool_child_run.py), but ``terminal_tool()``'s per-command cwd
resolution (``_resolve_command_cwd()``) and its post-command write-back
(``finalize_foreground_result()``) both key off ``get_current_session_key()``
-- an approval-context contextvar (tools/approval_context.py) -- not
``task_id``. Nothing bound that contextvar to the child's own
``child_task_id`` before this fix, so a delegated child's worker thread
(``contextvars.copy_context().run(...)`` inside ``_ChildRun.await_child()``)
inherited whichever session key the PARENT's context had, and the child's
FIRST real ``terminal_tool()`` call resolved cwd against the parent's
session, not its own worktree.

These tests drive the REAL ``_ChildRun.await_child()`` code path (the exact
executor/copy_context machinery the bug lives in), with a stub "child" whose
``run_conversation`` calls the REAL ``terminal_tool()`` -- no mocking of
record_session_cwd/get_session_cwd/session-key resolution, since the bug IS
the interaction between those real subsystems.
"""
import concurrent.futures
import contextvars
import json
import os
import subprocess
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.approval_context import get_current_session_key, reset_current_session_key, set_current_session_key
from tools.delegate_tool_child_run import _ChildRun
from tools.terminal_tool import get_session_cwd, record_session_cwd, terminal_tool


def _git(args, cwd, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@test"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "seed"], repo)
    return repo


class _StubChild:
    """Minimal stand-in for the real child AIAgent: ``run_conversation``
    issues one real ``terminal_tool()`` call (the thing that actually reads
    the session-key-keyed cwd) and returns the observed cwd for asserting
    against, in the shape ``_ChildRun.await_child()`` expects."""

    def __init__(self, command: str = "pwd"):
        self.command = command
        self.session_id = "stub-child-session"
        self.observed = []

    def run_conversation(self, *, user_message, task_id, stream_callback=None):
        raw = terminal_tool(command=self.command, task_id=task_id)
        payload = json.loads(raw)
        observed_cwd = (payload.get("cwd") or payload.get("output") or "").strip()
        self.observed.append(observed_cwd)
        return {
            "final_response": observed_cwd, "completed": True, "api_calls": 1,
            "messages": [], "interrupted": False,
        }

    def get_activity_summary(self):
        return {"api_call_count": len(self.observed)}


class ChildSessionKeyBindingTests(unittest.TestCase):
    """Direct reproduction of the bug via the real ``_ChildRun.await_child()``
    path: bind the parent's session key (as every real turn-start call site
    does), seed a child's worktree cwd record under its OWN child_task_id (as
    seed_workspace() does), then run the child through the real worker-thread
    machinery. The resolved command cwd must equal the child's worktree
    path, not whatever the parent's session had recorded."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hermes-childkey-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._parent_key = f"parent-session-{os.getpid()}-{id(self)}"
        self._child_task_id = f"subagent-0-{os.getpid()}-{id(self)}"

    def _make_run(self, stub_child, worktree_path):
        parent_agent = type("P", (), {"_current_task_id": self._parent_key})()
        run = _ChildRun(
            child=stub_child, parent_agent=parent_agent, task_index=0, goal="pwd",
            subagent_id=self._child_task_id, child_progress_cb=None,
        )
        run.child_task_id = self._child_task_id
        run.parent_task_id = self._parent_key
        run.worktree_info = {"path": worktree_path, "branch": "hermes-subagent/x"}
        return run

    def test_child_worker_thread_first_terminal_call_resolves_own_worktree_cwd(self):
        repo = _make_repo(self.tmp)
        parent_cwd = str(repo)
        worktree_path = str(self.tmp / "child-worktree")
        os.makedirs(worktree_path, exist_ok=True)

        parent_token = set_current_session_key(self._parent_key)
        try:
            # Parent's OWN session cwd is NOT the child's worktree -- exactly
            # what a buggy child inheriting the parent's session key would
            # fall through to instead of its own worktree record.
            record_session_cwd(self._parent_key, parent_cwd)
            # seed_workspace() records the worktree under child_task_id.
            record_session_cwd(self._child_task_id, worktree_path)

            stub_child = _StubChild(command="pwd")
            run = self._make_run(stub_child, worktree_path)

            # Real code path: await_child() submits the child's conversation
            # through contextvars.copy_context().run(...) on a worker thread.
            result, failure_entry, _deferred = run.await_child()
            self.assertIsNone(failure_entry, f"child failed unexpectedly: {failure_entry}")

            observed_cwd = stub_child.observed[0]
            self.assertTrue(
                os.path.realpath(observed_cwd) == os.path.realpath(worktree_path)
                or os.path.realpath(worktree_path) in os.path.realpath(observed_cwd),
                f"expected the child's first terminal call to resolve to its own worktree "
                f"({worktree_path!r}), got {observed_cwd!r} (parent cwd was {parent_cwd!r})",
            )
        finally:
            reset_current_session_key(parent_token)

    def test_child_binding_does_not_leak_back_into_parent_context(self):
        """Context-isolation contract: after await_child() returns, the
        PARENT's own get_current_session_key() must be unchanged -- a
        relationship between before and after, not a snapshot."""
        worktree_path = str(self.tmp / "wt")
        os.makedirs(worktree_path, exist_ok=True)
        record_session_cwd(self._child_task_id, worktree_path)

        parent_token = set_current_session_key(self._parent_key)
        try:
            before = get_current_session_key(default="")
            self.assertEqual(before, self._parent_key)

            stub_child = _StubChild(command="pwd")
            run = self._make_run(stub_child, worktree_path)
            run.await_child()

            after = get_current_session_key(default="")
            self.assertEqual(after, before, "child's session-key bind leaked back into the parent's context")
        finally:
            reset_current_session_key(parent_token)


class ConcurrentChildrenIsolationTests(unittest.TestCase):
    """Two children run concurrently through the real ``await_child()`` path,
    each worktree-isolated, must each keep resolving their OWN worktree cwd
    for the DURATION of the run -- guards against a fix that only patches the
    first call and regresses a later one once ``finalize_foreground_result``
    writes the observed cwd back under the (now correctly bound) session key."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hermes-childkey-concurrent-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _make_run(self, stub_child, child_task_id, worktree_path):
        parent_agent = type("P", (), {"_current_task_id": "shared-parent"})()
        run = _ChildRun(
            child=stub_child, parent_agent=parent_agent, task_index=0, goal="pwd",
            subagent_id=child_task_id, child_progress_cb=None,
        )
        run.child_task_id = child_task_id
        run.parent_task_id = "shared-parent"
        run.worktree_info = {"path": worktree_path, "branch": "hermes-subagent/x"}
        return run

    def test_two_concurrent_children_stay_pinned_to_their_own_worktrees(self):
        _make_repo(self.tmp)
        wt_a = str(self.tmp / "wt-a")
        wt_b = str(self.tmp / "wt-b")
        os.makedirs(wt_a, exist_ok=True)
        os.makedirs(wt_b, exist_ok=True)
        child_a_id = f"subagent-a-{id(self)}"
        child_b_id = f"subagent-b-{id(self)}"
        record_session_cwd(child_a_id, wt_a)
        record_session_cwd(child_b_id, wt_b)

        stub_a = _StubChild(command="pwd; pwd")  # two commands to exercise a LATER call too
        stub_b = _StubChild(command="pwd; pwd")

        class _TwoCallChild(_StubChild):
            def run_conversation(self, *, user_message, task_id, stream_callback=None):
                observed = []
                for _ in range(2):
                    raw = terminal_tool(command="pwd", task_id=task_id)
                    payload = json.loads(raw)
                    observed.append((payload.get("cwd") or payload.get("output") or "").strip())
                self.observed = observed
                return {"final_response": "ok", "completed": True, "api_calls": 2, "messages": [], "interrupted": False}

        stub_a = _TwoCallChild()
        stub_b = _TwoCallChild()
        run_a = self._make_run(stub_a, child_a_id, wt_a)
        run_b = self._make_run(stub_b, child_b_id, wt_b)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            fut_a = executor.submit(run_a.await_child)
            fut_b = executor.submit(run_b.await_child)
            result_a, fail_a, _ = fut_a.result(timeout=30)
            result_b, fail_b, _ = fut_b.result(timeout=30)
        finally:
            executor.shutdown(wait=False)

        self.assertIsNone(fail_a, f"child A failed: {fail_a}")
        self.assertIsNone(fail_b, f"child B failed: {fail_b}")

        for observed in stub_a.observed:
            self.assertTrue(
                os.path.realpath(observed) == os.path.realpath(wt_a)
                or os.path.realpath(wt_a) in os.path.realpath(observed),
                f"child A drifted off its worktree: {observed!r} (expected under {wt_a!r})",
            )
        for observed in stub_b.observed:
            self.assertTrue(
                os.path.realpath(observed) == os.path.realpath(wt_b)
                or os.path.realpath(wt_b) in os.path.realpath(observed),
                f"child B drifted off its worktree: {observed!r} (expected under {wt_b!r})",
            )


if __name__ == "__main__":
    unittest.main()
