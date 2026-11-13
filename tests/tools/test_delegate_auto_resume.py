"""``delegation.auto_resume``: the scheduler continues a limit-stopped child on its own conversation, never more than
``MAX_AUTO_ATTEMPTS`` times, and never once its parent moved on (the parent wins).

Only the model is scripted; the child build, registry, SessionDB, async dispatch, durable records and the scheduler's
own tick are real, against a temp HERMES_HOME.
"""

import json
import time
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _completion, _parent, _ScriptedModel
from tests.tools.test_delegate_resumable import _db_rows, _git


class _AlwaysLimited(_ScriptedModel):
    """Every turn ends on a provider limit whose reset has already passed (due at once)."""

    def __call__(self, **kw):
        child = super().__call__(**kw)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, conversation_history=None, **rest):
            result = ok(user_message, conversation_history=conversation_history, **rest)
            return {**result, "completed": False, "failed": True, "error": "limit", "failure_reason": "rate_limit",
                    "failure_resets_at": time.time() - 1}
        child.run_conversation.side_effect = run_conversation
        return child


def _settled(delegation_id):
    """The unit's completion once it finished: the queued event, or, for a unit every child of which waits on a
    promised automatic resume (silent: nothing queued), the event its ledger row recorded."""
    from hermes_agent.tools.async_delegation import _transaction
    from tests.waiting import wait_until

    def row():
        with _transaction() as conn:
            got = conn.execute("SELECT state, delivery_state, event_json FROM async_delegations WHERE delegation_id=?",
                               (delegation_id,)).fetchone()
        return got if got and got[0] not in ("running", "finalizing") and got[2] else None
    _state, delivery, event = wait_until(row, what=f"{delegation_id} to finish")
    return _completion(delegation_id) if delivery == "pending" else json.loads(event)


def _repo(tmp_path, name):
    repo = tmp_path / name
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    return repo


def _enable(*, gateway=True, **extra):
    """``delegation.auto_resume`` on; *gateway*: this process owns the profile the way a gateway does (the only
    process whose watcher resumes its children, so the only one whose stop notices promise a resume)."""
    from hermes_agent.gateway.runtime_ownership import process_ownership
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(json.dumps(
        {"delegation": {"auto_resume": True, "worktree_isolation": True, **extra}}))
    if gateway:
        process_ownership.reserve([get_hermes_home()])


def _scheduler(parent, dispatched):
    """The scheduler's ``start``: the same continue the gateway performs, on this test's parent."""
    from hermes_agent.tools.delegate_tool_continue import continue_subagent

    def start(record, unit, message):
        out = json.loads(continue_subagent(record["subagent_id"], message, parent, background=True, auto=True))
        if out.get("status") != "dispatched":
            return out.get("error") or "not dispatched"
        dispatched.append(out["delegation_id"])
        return None
    return start


def test_a_child_that_keeps_hitting_the_limit_is_resumed_at_most_twice_and_says_why(tmp_path):
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import MAX_AUTO_ATTEMPTS, tick
    from hermes_agent.tools.terminal_tool import record_session_cwd

    _enable()
    model, parent, dispatched = _AlwaysLimited(), _parent(), []
    parent._current_task_id = "parent-task"
    record_session_cwd("parent-task", str(_repo(tmp_path, "repo")))
    with patch("run_agent.AIAgent", side_effect=model):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}], background=True, parent_agent=parent))
        first = _settled(out["delegation_id"])
        sid = out["subagent_ids"][0]
        assert first["results"][0]["auto_resume"]["at"] <= time.time()  # the stop notice promises the resume
        for _ in range(MAX_AUTO_ATTEMPTS + 2):
            before = len(dispatched)
            tick(_scheduler(parent, dispatched), capacity=3)
            if len(dispatched) > before:
                last = _settled(dispatched[-1])
    assert len(dispatched) == MAX_AUTO_ATTEMPTS  # every attempt hit the limit again; no third one ran
    assert len(model.turns) == 1 + MAX_AUTO_ATTEMPTS and {t["session_id"] for t in model.turns} == {
        model.turns[0]["session_id"]}  # always the child's own conversation
    assert "auto_resume" not in last["results"][0]  # the last stop no longer promises a resume
    state, _pinned, record = _db_rows(parent)[sid]
    assert (state, record["auto_attempts"]) == ("resumable", MAX_AUTO_ATTEMPTS)  # still continuable by hand
    assert f"gave up after {MAX_AUTO_ATTEMPTS} automatic attempts" in record["auto_note"]
    assert last["results"][0]["auto_gave_up"] and "gave up" in last["results"][0]["error"]  # the one failure


@pytest.mark.parametrize("parent_moved_on", ["new_work_same_repo", "parent_reset", "new_work_other_repo"])
def test_the_parent_wins_a_child_is_not_resumed_once_its_parent_moved_on(tmp_path, parent_moved_on):
    """Explicit cancel only: a limit-stopped child is superseded (kept, pinned, never resumed) when the parent
    conversation was reset; new work the parent starts, in the same repository or elsewhere, does not cancel it."""
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import SUPERSEDED, tick
    from hermes_agent.tools.terminal_tool import record_session_cwd

    _enable()
    parent, dispatched = _parent(), []
    parent._current_task_id = "parent-task"
    repo = _repo(tmp_path, "repo")
    record_session_cwd("parent-task", str(repo))
    limited, fine = _AlwaysLimited(), _ScriptedModel()
    with patch("run_agent.AIAgent", side_effect=limited):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}], background=True, parent_agent=parent))
        _settled(out["delegation_id"])
        sid = out["subagent_ids"][0]
    with patch("run_agent.AIAgent", side_effect=fine):
        if parent_moved_on == "parent_reset":
            parent._session_db.end_session(parent.session_id, "new_session")
        else:
            if parent_moved_on == "new_work_other_repo":
                record_session_cwd("parent-task", str(_repo(tmp_path, "elsewhere")))
            again = json.loads(delegate_task(tasks=[{"goal": "write the parser, take two"}], background=True,
                                             parent_agent=parent))
            _settled(again["delegation_id"])
        with patch("run_agent.AIAgent", side_effect=limited):
            resumed = tick(_scheduler(parent, dispatched), capacity=3)
            if dispatched:
                _settled(dispatched[-1])

    state, pinned, record = _db_rows(parent)[sid]
    if parent_moved_on in ("new_work_other_repo", "new_work_same_repo"):
        assert resumed == [sid] and record["auto_attempts"] == 1
        return
    assert (resumed, state, pinned) == ([], SUPERSEDED, 1), record.get("auto_note")
    assert len(limited.turns) == 1  # the old child never ran again
    assert record["auto_note"]


def test_stop_on_a_waiting_child_dismisses_it_and_releases_its_pin(tmp_path):
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    from hermes_agent.tools.terminal_tool import record_session_cwd

    _enable()
    parent, dispatched = _parent(), []
    parent._current_task_id = "parent-task"
    record_session_cwd("parent-task", str(_repo(tmp_path, "repo")))
    with patch("run_agent.AIAgent", side_effect=_AlwaysLimited()):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}], background=True, parent_agent=parent))
        _settled(out["delegation_id"])
        sid = out["subagent_ids"][0]
        stopped = json.loads(delegate_task(action="stop", subagent_id=sid, parent_agent=parent))
        assert tick(_scheduler(parent, dispatched), capacity=3) == []
    assert stopped["status"] == "dismissed", stopped
    state, pinned, record = _db_rows(parent)[sid]
    assert (state, pinned, dispatched) == ("stopped", 0, [])
    assert record["auto_note"] == "dismissed by its parent"
