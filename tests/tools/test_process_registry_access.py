"""process_manage session actions are scoped to the caller's identity (#108413).

A session id is not a capability: another conversation that learns a live id must not poll,
kill, or write to it. The caller matches when it owns the process (owner_task_id/task_id), shares
its gateway session_key, or is that session_key's compression continuation.
"""

import json
from unittest.mock import MagicMock

import pytest

import tools.approval_context as ac
from tools import process_registry as pr
from tools import process_registry_results as results
from tools.process_registry import ProcessRegistry, ProcessSession


def _session(owner_task_id="task-a", session_key="key-a", task_id="container-a", exited=True):
    sess = ProcessSession(id="proc_owned1", command="echo hi", task_id=task_id, output_buffer="secret-out")
    sess.owner_task_id = owner_task_id
    sess.session_key = session_key
    sess.exited = exited
    sess.exit_code = 0 if exited else None
    return sess


@pytest.fixture()
def reg(monkeypatch):
    monkeypatch.setattr("hermes_cli.process_identity._update_supervised_pgids", lambda verb, pgids: None)
    registry = ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", registry)
    # Compression lineage lookups never touch a real state.db here.
    monkeypatch.setattr(results, "_owns_result", lambda owner, parent: bool(parent) and owner == parent)
    return registry


def _call(monkeypatch, action, task_id, session_key, **args):
    monkeypatch.setattr(ac, "get_current_session_key", lambda default="": session_key)
    return json.loads(pr._handle_process({"action": action, "session_id": "proc_owned1", **args}, task_id=task_id))


def _finished(reg, **kw):
    sess = _session(**kw)
    reg._finished[sess.id] = sess
    return sess


def test_owner_task_id_may_poll(reg, monkeypatch):
    _finished(reg)
    assert _call(monkeypatch, "poll", "task-a", "")["status"] != "not_found"


def test_foreign_task_gets_not_found_identical_to_missing(reg, monkeypatch):
    _finished(reg)
    out = _call(monkeypatch, "poll", "task-b", "key-b")
    assert out == pr._not_found("proc_owned1")
    assert "secret-out" not in json.dumps(out)


def test_foreign_task_cannot_read_log(reg, monkeypatch):
    _finished(reg)
    assert _call(monkeypatch, "log", "task-b", "key-b") == pr._not_found("proc_owned1")


def test_same_session_key_may_poll_across_tasks(reg, monkeypatch):
    """A session-scoped process started by a sibling task is still ours (#29177)."""
    _finished(reg)
    assert _call(monkeypatch, "poll", "task-b", "key-a")["status"] != "not_found"


def test_compression_continuation_keeps_access(reg, monkeypatch):
    monkeypatch.setattr(results, "_owns_result", lambda owner, parent: (owner, parent) == ("key-a-child", "key-a"))
    _finished(reg)
    assert _call(monkeypatch, "poll", "task-b", "key-a-child")["status"] != "not_found"



def test_resumed_api_run_keeps_access_after_compression(monkeypatch):
    """/v1/runs binds task_id to the conversation's live compression tip and the session key to a
    fresh run id, so after a rotation only the task's own lineage ties the caller to its process."""
    from hermes_state import SessionDB
    monkeypatch.setattr("hermes_cli.process_identity._update_supervised_pgids", lambda verb, pgids: None)
    registry = ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", registry)
    db = SessionDB()
    try:
        db.create_session("api-conv", "api_server")
        db.end_session("api-conv", end_reason="compression")
        db.create_session("api-conv-tip", "api_server", parent_session_id="api-conv")
    finally:
        db.close()
    registry._finished["proc_owned1"] = _session(owner_task_id="api-conv", session_key="run-1", task_id="api-conv")
    assert _call(monkeypatch, "poll", "api-conv-tip", "run-2")["status"] != "not_found"
    assert _call(monkeypatch, "poll", "other-conv", "run-3") == pr._not_found("proc_owned1")


def test_container_task_id_may_poll(reg, monkeypatch):
    _finished(reg)
    assert _call(monkeypatch, "poll", "container-a", "")["status"] != "not_found"


def test_unidentified_caller_passes(reg, monkeypatch):
    """Internal plumbing calls the handler without task_id/session_key."""
    _finished(reg)
    assert _call(monkeypatch, "poll", "", "")["status"] != "not_found"


def test_unowned_session_is_actionable(reg, monkeypatch):
    _finished(reg, owner_task_id="", session_key="", task_id="")
    assert _call(monkeypatch, "poll", "task-b", "key-b")["status"] != "not_found"


def test_foreign_task_cannot_kill_running_process(reg, monkeypatch):
    sess = _session(exited=False)
    reg._running[sess.id] = sess
    killed = []
    monkeypatch.setattr(reg, "kill_process", lambda sid, **kw: killed.append(sid) or {"status": "killed"})
    assert _call(monkeypatch, "kill", "task-b", "key-b") == pr._not_found("proc_owned1")
    assert killed == []
    assert _call(monkeypatch, "kill", "task-a", "")["status"] == "killed"
    assert killed == ["proc_owned1"]


@pytest.mark.parametrize("action", ["write", "submit", "close"])
def test_foreign_task_cannot_touch_stdin(reg, monkeypatch, action):
    sess = _session(exited=False)
    sess.process = MagicMock()
    reg._running[sess.id] = sess
    assert _call(monkeypatch, action, "task-b", "key-b", data="x") == pr._not_found("proc_owned1")
    sess.process.stdin.write.assert_not_called()
    sess.process.stdin.close.assert_not_called()


def test_allowed_short_id_acts_on_the_session_that_was_checked(reg, monkeypatch):
    """The action runs on the checked session's FULL id: a prefix that later resolves to another
    conversation's process (the first one finished and was pruned) cannot be retargeted."""
    mine = _session()
    mine.id = "proc_abcd1111aaaa"
    reg._finished[mine.id] = mine
    seen = []
    monkeypatch.setitem(pr._SESSION_ACTIONS, "poll", (lambda sid, args: seen.append(sid) or {"status": "exited"}, False))
    monkeypatch.setattr(ac, "get_current_session_key", lambda default="": "key-a")
    pr._handle_process({"action": "poll", "session_id": "abcd1111"}, task_id="task-a")
    assert seen == ["proc_abcd1111aaaa"]


def test_unresolved_prefix_is_never_dispatched(reg, monkeypatch):
    """An ambiguous prefix passed no ownership check: refuse it rather than let the action
    resolve it again after a prune has made it unique to another conversation's process."""
    for sid, owner in (("proc_abcd1111aaaa", "task-a"), ("proc_abcd1111bbbb", "task-b")):
        sess = _session(owner_task_id=owner, session_key="")
        sess.id = sid
        reg._finished[sid] = sess
    seen = []
    monkeypatch.setitem(pr._SESSION_ACTIONS, "kill", (lambda sid, args: seen.append(sid) or {"status": "killed"}, False))
    monkeypatch.setattr(ac, "get_current_session_key", lambda default="": "")
    out = json.loads(pr._handle_process({"action": "kill", "session_id": "abcd1111"}, task_id="task-a"))
    assert out == pr._not_found("abcd1111")
    assert seen == []
