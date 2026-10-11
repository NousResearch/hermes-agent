"""Regression: a delegated child's terminal must resolve its cwd under the CHILD's
task id, not the ambient (parent's) session key.

The defect this pins: ``_ChildRun.seed_workspace`` records the child's isolated
worktree cwd under ``child_task_id``, but the child inherits the parent's
process-global session key (``delegated_child_context`` never rebinds it), and the
terminal reader preferred that ambient key — so a worktree-isolated child executed
in the PARENT checkout while reporting a worktree path. The post-command cwd writer
(``finalize_foreground_result``) uses the same resolved key, so reader and writer
must agree per-child.

The test crosses the REAL writer/reader seam: the actual ``seed_workspace`` writer
and the actual ``terminal_tool`` reader, under a delegated-child context with an
ambient parent session key present. It fails on pre-fix behavior (resolved cwd ==
parent checkout) and passes only when the child resolves its own record.
"""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tools.terminal_tool as terminal_tool  # noqa: E402
from agent.delegation_context import delegated_child_context  # noqa: E402
from tools.approval_context import (  # noqa: E402
    reset_current_session_key,
    set_current_session_key,
)
from tools.delegate_tool_child_run import _ChildRun  # noqa: E402


def _git(args, cwd, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check,
    )


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@test"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "seed"], repo)
    return repo


@contextmanager
def _ambient_session_key(key: str):
    """Bind the approval session-key contextvar AND the process env var — the
    desktop/TUI reality where a non-empty parent key wins every fallback."""
    token = set_current_session_key(key)
    old_env = os.environ.get("HERMES_SESSION_KEY")
    os.environ["HERMES_SESSION_KEY"] = key
    try:
        yield
    finally:
        reset_current_session_key(token)
        if old_env is None:
            os.environ.pop("HERMES_SESSION_KEY", None)
        else:
            os.environ["HERMES_SESSION_KEY"] = old_env


class _FakeChild:
    provider = "openrouter"
    model = "some/model"


class _FakeParent:
    _current_task_id = None


class _RecordingEnv:
    """Minimal env double: records the cwd each command was executed in.

    Returns ``cwd_observed``/``cwd`` so the REAL finalize_foreground_result
    writer path fires (tools/terminal_tool_result.py:220-224) and the key it
    records under can be asserted — pinning the writer half of the defect,
    not just the reader half.
    """

    env = {}

    def __init__(self):
        self.executed_cwds = []

    def execute(self, command, **kwargs):
        self.executed_cwds.append(kwargs.get("cwd"))
        return {"output": "ok", "returncode": 0,
                "cwd_observed": True, "cwd": kwargs.get("cwd")}


def _clear_cwd_store():
    """Drop this test's process-global _session_cwd / alias entries so tests
    stay independent (no autouse reset exists for these module globals)."""
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
    with terminal_tool._container_alias_lock:
        terminal_tool._container_aliases.clear()


def _drive_reader(task_id: str):
    """Call the REAL terminal_tool reader path for *task_id*; return (result, env)."""
    env = _RecordingEnv()
    with mock.patch.object(terminal_tool, "_get_env_config") as cfg, \
            mock.patch.object(terminal_tool, "_check_all_guards",
                              return_value={"approved": True}), \
            mock.patch.object(terminal_tool, "_run_approval_guards",
                              return_value=mock.Mock(note="", approved_run=True)), \
            mock.patch.object(terminal_tool, "_acquire_env",
                              side_effect=lambda plan, tid: env):
        cfg.return_value = {"env_type": "local", "cwd": os.getcwd(), "timeout": 60, "lifetime_seconds": 3600}
        result = terminal_tool.terminal_tool(command="pwd", task_id=task_id)
    return result, env


def test_delegated_child_resolves_own_worktree_cwd_not_parent_session_key(tmp_path):
    """A worktree-isolated child must execute in ITS worktree even when the ambient
    (parent's) session key and the parent's cwd record are both present — the exact
    desktop/TUI conditions under which the defect fired."""
    _clear_cwd_store()
    repo = _make_repo(tmp_path)
    parent_key = "20261005_051011_8e546f"  # the parent's process-global session key
    child_task_id = "sa-0-3b5d8aed"

    run = _ChildRun(_FakeChild(), _FakeParent(), 0, "Do the thing", child_task_id, None)

    # The parent's cwd record exists under the parent key (live session reality):
    # the parent sits in its own checkout at the repo root.
    terminal_tool.record_session_cwd(parent_key, str(repo))

    # REAL writer: seed_workspace creates the worktree and records its cwd under
    # child_task_id — with the parent's ambient key bound, as at a live dispatch.
    with mock.patch("tools.delegate_tool_child_run._create_isolated_worktree") as iso, \
            delegated_child_context(child_task_id), \
            _ambient_session_key(parent_key), \
            mock.patch("tools.delegate_tool_config._get_worktree_isolation",
                       return_value=True), \
            mock.patch.object(terminal_tool, "_get_env_config") as cfg:
        from tools import subagent_worktree as sw
        real_create = sw.create_subagent_worktree
        iso.side_effect = lambda parent_agent, parent_task_id, subagent_id: real_create(
            str(repo), subagent_id=child_task_id)
        cfg.return_value = {"env_type": "local", "cwd": os.getcwd(), "timeout": 60, "lifetime_seconds": 3600}
        run.seed_workspace()

    assert run.worktree_info is not None, "worktree isolation must engage for this test"
    worktree_path = run.worktree_info["path"]
    assert os.path.isdir(worktree_path), "the worktree must actually exist"
    assert worktree_path != str(repo), "worktree must differ from the parent checkout"
    # Writer recorded the worktree under the CHILD's own key.
    assert terminal_tool.get_session_cwd(child_task_id) == worktree_path
    # The parent's own record is untouched under its key (what the pre-fix reader
    # would wrongly resolve for the child).
    assert terminal_tool.get_session_cwd(parent_key) == str(repo)

    # REAL reader: a terminal call on the child's task id, inside the child
    # context, with the ambient parent key still bound (inherited verbatim).
    with delegated_child_context(child_task_id), _ambient_session_key(parent_key):
        result, env = _drive_reader(child_task_id)

    payload = json.loads(result)
    assert payload.get("exit_code") == 0, payload
    # THE assertion: the child executes in its own worktree, not the parent checkout.
    assert env.executed_cwds == [worktree_path], (
        f"child terminal resolved to {env.executed_cwds!r}, expected its own worktree {worktree_path!r}"
    )
    # And explicitly not the parent checkout — the pre-fix behavior this test pins.
    assert env.executed_cwds != [str(repo)], (
        "child terminal resolved the parent's checkout — the isolation defect"
    )
    # Writer half: the REAL finalize must have recorded the observed cwd under
    # the CHILD's key, leaving the parent's record untouched. Pre-fix, both the
    # reader and this write-back used the parent's key (record corruption; with
    # concurrent children, siblings overwrote one shared record).
    assert terminal_tool.get_session_cwd(child_task_id) == worktree_path, (
        "finalize did not write the child's observed cwd under the CHILD key"
    )
    assert terminal_tool.get_session_cwd(parent_key) == str(repo), (
        "finalize corrupted the parent's cwd record — the writer half of the defect"
    )


def test_ambient_key_still_wins_for_non_children(tmp_path):
    """Scope guard: a NON-child caller with an ambient session key keeps the
    historical behavior (ambient key's record wins), so the patch cannot have
    widened the blast radius to ordinary sessions."""
    _clear_cwd_store()
    repo = _make_repo(tmp_path)
    session_key = "gateway-session-1"

    terminal_tool.record_session_cwd(session_key, str(repo))
    assert terminal_tool.get_session_cwd(session_key) == str(repo)

    # NOT in a delegated-child context: ambient key must keep resolving its own record.
    with _ambient_session_key(session_key):
        result, env = _drive_reader("unrelated-task-id")
    payload = json.loads(result)
    assert payload.get("exit_code") == 0, payload
    assert env.executed_cwds == [str(repo)], (
        "non-child readers must keep resolving the ambient session key's record"
    )
