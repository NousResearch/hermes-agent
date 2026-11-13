"""A delegated child's terminal runs in its OWN directory even though it inherits the parent's session key.

Children inherit the parent's approval session key (approvals, completion routing), and the terminal used that
key for its per-session cwd record, so every child of one conversation shared ONE cwd record: the last child to
run a command moved every sibling into its directory, its isolation worktree included. Found live: two stopped
children continued one after the other; the second one's first command ran ``cd`` into the first one's (by then
removed) worktree. Only the model is scripted; delegation, worktree isolation and the terminal are the real ones.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _completion, _parent, _ScriptedModel


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    monkeypatch.setattr("hermes_agent.tools.delegate_tool._get_worktree_isolation", lambda: True)
    monkeypatch.setattr("hermes_agent.tools.delegate_tool._resolve_workspace_hint", lambda _p: str(repo))
    monkeypatch.setenv("TERMINAL_CWD", str(repo))
    return repo


@pytest.fixture
def bound_session_key():
    """The parent conversation's session key, as the gateway/CLI/TUI turn binds it (children inherit it)."""
    from hermes_agent.tools.approval_context import reset_current_session_key, set_current_session_key
    token = set_current_session_key("parent-sess")
    yield
    reset_current_session_key(token)


def _pwd_model(stop_first_turn_of_each_child=False):
    """Each child turn runs a real terminal ``pwd``; optionally its first turn then stops on a usage limit."""
    from hermes_agent.tools.terminal_tool import terminal_tool

    model, seen, stopped = _ScriptedModel(), [], set()

    def build(**kw):
        child = model(**kw)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, **rest):
            out = json.loads(terminal_tool("pwd", task_id=rest.get("task_id")))
            seen.append((rest.get("task_id"), out["exit_code"], out["output"].strip()))
            result = ok(user_message, **rest)
            if stop_first_turn_of_each_child and rest.get("task_id") not in stopped:
                stopped.add(rest.get("task_id"))
                return {**result, "completed": False, "failed": True, "error": "limit", "failure_reason": "billing",
                        "failure_resets_at": time.time() + 3600}
            return result
        child.run_conversation.side_effect = run_conversation
        return child
    return build, seen


def _spawn(parent, goal):
    from hermes_agent.tools.delegate_tool import delegate_task
    out = json.loads(delegate_task(tasks=[{"goal": goal}], background=True, parent_agent=parent))
    _completion(out["delegation_id"])
    return out["subagent_ids"][0]


def _own_tree(repo, task_id):
    return os.path.realpath(repo / ".worktrees" / f"subagent-{task_id}")


def test_continued_children_each_run_in_their_own_worktree_after_a_sibling_finished(repo, bound_session_key):
    from hermes_agent.tools.delegate_tool import delegate_task

    build, seen = _pwd_model(stop_first_turn_of_each_child=True)
    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=build):
        a, b = _spawn(parent, "A"), _spawn(parent, "B")  # both stop resumable; their trees are kept
        for sid in (a, b):  # A is continued, finishes and its tree is removed; then B is continued
            out = json.loads(delegate_task(action="continue", subagent_id=sid, message="go on", background=True,
                                           parent_agent=parent))
            _completion(out["delegation_id"])

    continued = seen[2:]
    assert [(tid, code, os.path.realpath(pwd)) for tid, code, pwd in continued] == [
        (a, 0, _own_tree(repo, a)), (b, 0, _own_tree(repo, b))], seen


def test_a_new_child_starts_in_its_own_worktree_not_the_previous_siblings(repo, bound_session_key):
    build, seen = _pwd_model()
    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=build):
        a, b = _spawn(parent, "A"), _spawn(parent, "B")

    assert [(tid, code, os.path.realpath(pwd)) for tid, code, pwd in seen] == [
        (a, 0, _own_tree(repo, a)), (b, 0, _own_tree(repo, b))], seen
