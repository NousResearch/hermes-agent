"""Durable resumable subagents: a stopped child leaves a state.db record that ``delegate_task(action='continue')``
rebuilds it from after the process is gone, the record says how it may resume, and its worktree is kept from the
startup sweep for the record's life.

Only the model is scripted; the child build, registry, SessionDB, async dispatch, boot recovery and the worktree
sweep are the real ones, against a temp HERMES_HOME.
"""

import json
import os
import subprocess
import time
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _completion, _parent, _ScriptedModel, _spawn


def _db_rows(parent):
    import sqlite3
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test reads a temp-home store
    try:
        return {sid: (state, pinned, {**json.loads(raw), "stop_cause": cause}) for sid, state, pinned, raw, cause in
                conn.execute("SELECT subagent_id, state, pinned, record_json, stop_cause FROM resumable_subagents")}
    finally:
        conn.close()


def _forget_process_state():
    """What a restart loses: the in-process registry of finished children."""
    from tools.delegate_tool_registry import _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        _recent_subagents.clear()


class _FailingModel(_ScriptedModel):
    """First turn ends on a usage limit with a reset time, as the conversation loop reports it."""

    def __init__(self, reason="billing", resets_at=None):
        super().__init__()
        self.reason, self.resets_at = reason, resets_at

    def __call__(self, **kw):
        child = super().__call__(**kw)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, conversation_history=None, **rest):
            result = ok(user_message, conversation_history=conversation_history, **rest)
            if len(self.turns) == 1:
                result = {**result, "completed": False, "failed": True, "error": "limit", "failure_reason": self.reason,
                          **({"failure_resets_at": self.resets_at, "usage_limit": {"window": "5h"}}
                             if self.resets_at else {})}
            return result
        child.run_conversation.side_effect = run_conversation
        return child


def test_a_limit_stopped_child_is_continued_after_a_restart_on_its_own_transcript():
    from tools.delegate_tool import delegate_task

    resets = time.time() + 3600
    model = _FailingModel(resets_at=resets)
    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}], background=True, parent_agent=parent))
        done = _completion(out["delegation_id"])
        sid = out["subagent_ids"][0]
        assert '"failure_resets_at"' in json.dumps(done), done  # the parent sees when the limit resets
        state, _pinned, record = _db_rows(parent)[sid]
        assert (state, record["resume_mode"], record["failure_resets_at"]) == ("resumable", "AUTO_AT", resets)
        assert record["usage_limit"] == {"window": "5h"}

        _forget_process_state()
        cont = json.loads(delegate_task(action="continue", subagent_id=sid, message="the limit reset, go on",
                                        background=True, parent_agent=parent))
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])

    first, again = model.builds
    assert again["session_id"] == model.turns[0]["session_id"] and again["model"] == first["model"]
    assert [m["content"] for m in model.turns[1]["history"]][:1] == ["write the parser"]
    assert _db_rows(parent)[sid][0] == "completed"


@pytest.mark.parametrize("cause", ["gateway", "user"])
def test_a_gateway_shutdown_leaves_a_resumable_record_and_a_user_stop_does_not(cause):
    import threading

    from tools.async_delegation import interrupt_all
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, interrupt_subagent

    started, release = threading.Event(), threading.Event()
    model = _ScriptedModel()

    def build(**kw):
        child = model(**kw)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, **rest):
            started.set()
            release.wait(10)
            ok(user_message, **rest)
            return {"final_response": "Operation interrupted.", "interrupted": True, "completed": False}
        child.run_conversation.side_effect = run_conversation
        return child

    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=build):
        out = json.loads(delegate_task(tasks=[{"goal": "long job"}], background=True, parent_agent=parent))
        sid = out["subagent_ids"][0]
        assert started.wait(10)
        if cause == "gateway":
            interrupt_all(reason="gateway shutdown (test)", resumable=True)  # gateway/run_shutdown.py's call
        else:
            with _active_subagents_lock:
                assert sid in _active_subagents
            interrupt_subagent(sid)
        release.set()
        _completion(out["delegation_id"])

    state, pinned, record = _db_rows(parent)[sid]
    if cause == "gateway":
        assert (state, record["resume_mode"], record["stop_cause"]) == ("resumable", "AUTO_ON_BOOT", "gateway_shutdown")
    else:
        assert (state, pinned) == ("stopped", 0)


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


def test_crashed_childs_worktree_survives_the_sweep_until_its_record_expires(tmp_path):
    """The owner died mid-run (no stop record): boot recovery pins the tree the child last worked in; the real
    sweep keeps it past 24 h; once the record expires the same sweep archives and removes it."""
    import sqlite3

    from hermes_cli.worktree_sweep import plan_sweep, run_sweep, sweep_settings
    from hermes_state import SessionDB
    from tools.async_delegation import recover_abandoned_delegations
    from tools.delegate_tool_resumable import _upsert
    from hermes_constants import get_hermes_home

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    tree = repo / ".worktrees" / "subagent-sa-crash"
    _git("worktree", "add", "-q", str(tree), "-b", "subagent-sa-crash", cwd=repo)
    (tree / "work.txt").write_text("unsaved child work")
    SessionDB(db_path=get_hermes_home() / "state.db").close()

    db = get_hermes_home() / "state.db"
    conn = sqlite3.connect(db)  # health: allow HX015 -- test seeds a temp-home store
    with conn:
        _upsert(conn, "sa-crash", {"session_id": "s", "cwd": str(tree), "owner_pid": 2 ** 22 + 7,
                                   "owners": ["parent-sess"], "worktrees_at_start": [str(repo)]},
                state="running", pinned=True, expires_at=time.time() + 3600)
    conn.close()
    recover_abandoned_delegations()
    old = time.time() - 30 * 3600
    git_dir = repo / ".git" / "worktrees" / "subagent-sa-crash"
    for path in (tree, tree / "work.txt", git_dir / "HEAD", git_dir / "logs" / "HEAD"):
        os.utime(path, (old, old))

    plan = {d.name: d.action for d in plan_sweep(str(repo), 24)}
    assert plan["subagent-sa-crash"] == "skip-resumable", plan
    run_sweep(str(repo), 24, sweep_settings())
    assert tree.is_dir()

    conn = sqlite3.connect(db)  # health: allow HX015 -- test ages the record
    with conn:
        conn.execute("UPDATE resumable_subagents SET expires_at=?", (time.time() - 1,))  # its 48 h are over
    conn.close()
    assert {d.name: d.action for d in plan_sweep(str(repo), 24)}["subagent-sa-crash"] == "archive-remove"
    run_sweep(str(repo), 24, sweep_settings())
    assert not tree.exists()
    refs = subprocess.run(["git", "for-each-ref", "refs/hermes/archive"], cwd=repo, capture_output=True, text=True,
                          check=True)
    assert "subagent-sa-crash" in refs.stdout
