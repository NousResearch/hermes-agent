"""Durable resumable subagents, review round 1: the durable claim is the one gate for every continue, a claim never
outlives the run that took it, boot recovery holds no write lock through its git probes, a continue picks up the
child's newest conversation segment, and each profile's pins obey that profile's own cap.

Only the model is scripted; the child build, registry, SessionDB, async dispatch and boot recovery are real, against
a temp HERMES_HOME.
"""

import json
import os
import sqlite3
import subprocess
import time
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _completion, _parent, _ScriptedModel, _spawn
from tests.tools.test_delegate_resumable import _db_rows, _FailingModel, _forget_process_state, _git

_DEAD_PID = 2 ** 22 + 7


def _set_row(parent, sid, **cols):
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test edits a temp-home store
    with conn:
        conn.execute(f"UPDATE resumable_subagents SET {', '.join(f'{k}=?' for k in cols)} WHERE subagent_id=?",
                     (*cols.values(), sid))
    conn.close()


def test_a_resumable_outcome_without_a_durable_record_is_pruned_as_before(tmp_path, monkeypatch):
    """No record (its write failed): nothing pins the tree, so the stop prunes it as main did."""
    from hermes_constants import get_hermes_home
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.terminal_tool import get_session_cwd, record_session_cwd

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"delegation": {"worktree_isolation": True}}))
    monkeypatch.setattr("hermes_agent.tools.delegate_tool_resumable.record_start", lambda run: None)
    cwds = []
    model = _FailingModel(resets_at=time.time() + 3600)

    def build(**kw):
        child = model(**kw)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, task_id=None, **rest):
            cwds.append(get_session_cwd(task_id))
            return ok(user_message, task_id=task_id, **rest)
        child.run_conversation.side_effect = run_conversation
        return child

    parent = _parent()
    parent._current_task_id = "parent-task"
    record_session_cwd("parent-task", str(repo))
    with patch("hermes_agent.run_agent.AIAgent", side_effect=build):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}], background=True, parent_agent=parent))
        _completion(out["delegation_id"])
    assert os.path.realpath(cwds[0]) != os.path.realpath(repo)  # isolation engaged
    assert not os.path.exists(cwds[0])


@pytest.mark.parametrize("progress", ["relayed", "none"])
def test_a_finished_tool_call_checkpoints_the_running_childs_cwd(tmp_path, monkeypatch, progress):
    """A hard kill between heartbeats still knows where the child was, also for a child with no progress
    callback at all (no parent callback, live transcript unavailable)."""
    from hermes_agent.tools.delegate_tool import delegate_task
    if progress == "none":
        monkeypatch.setattr("hermes_agent.tools.delegation_live_log.create_live_transcripts",
                            lambda tasks, *a, **k: (None, [None] * len(tasks), [None] * len(tasks)))
    from hermes_agent.tools.delegate_tool_routes import child_profile_scope
    from hermes_agent.tools.terminal_tool import record_session_cwd

    moved = tmp_path / "moved"
    moved.mkdir()
    model, mid_run = _ScriptedModel(), {}
    parent = _parent()
    if progress == "none":
        parent._delegate_spinner = None  # no spinner and no parent callback: no relay is built

    def build(**kw):
        child = model(**kw)
        child.tool_progress_callback = kw.get("tool_progress_callback")  # as AIAgent stores it (None: no relay)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, task_id=None, **rest):
            if progress == "none":
                assert child.tool_progress_callback is not None  # the checkpoint installed its own
            with child_profile_scope(child):
                record_session_cwd(task_id, str(moved))  # the child's terminal cd'd
            child.tool_progress_callback("tool.completed", "terminal", None, None, duration=0.1)
            mid_run.update({sid: (state, record.get("cwd")) for sid, (state, _p, record) in _db_rows(parent).items()})
            return ok(user_message, task_id=task_id, **rest)
        child.run_conversation.side_effect = run_conversation
        return child

    with patch("hermes_agent.run_agent.AIAgent", side_effect=build):
        out = json.loads(delegate_task(tasks=[{"goal": "cd"}], background=True, parent_agent=parent))
        _completion(out["delegation_id"])
    assert mid_run == {out["subagent_ids"][0]: ("running", os.path.realpath(moved))}


@pytest.mark.parametrize("reason", ["billing", None])
def test_a_resumable_isolated_child_keeps_its_worktree_and_continues_in_it(tmp_path, reason):
    """A usage limit (AUTO_AT) or a provider failure (ON_FIX) keeps the child's isolation tree for the continue."""
    from hermes_constants import get_hermes_home
    from hermes_agent.tools.delegate_tool import delegate_task
    from hermes_agent.tools.terminal_tool import get_session_cwd, record_session_cwd

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"delegation": {"worktree_isolation": True}}))

    cwds, trees = [], []
    model = _FailingModel(resets_at=time.time() + 3600) if reason else _FailingModel(reason="", resets_at=None)

    def build(**kw):
        child = model(**kw)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, task_id=None, **rest):
            cwd = get_session_cwd(task_id)
            cwds.append(cwd)
            trees.append(subprocess.run(["git", "worktree", "list"], cwd=repo, capture_output=True, text=True,
                                        check=True).stdout)
            if len(cwds) == 1:
                with open(os.path.join(cwd, "work.txt"), "w") as fh:
                    fh.write("child work")  # uncommitted edits in its tree
            return ok(user_message, task_id=task_id, **rest)
        child.run_conversation.side_effect = run_conversation
        return child

    parent = _parent()
    parent._current_task_id = "parent-task"
    record_session_cwd("parent-task", str(repo))
    with patch("hermes_agent.run_agent.AIAgent", side_effect=build):
        out = json.loads(delegate_task(tasks=[{"goal": "write the parser"}], background=True, parent_agent=parent))
        done = _completion(out["delegation_id"])
        sid = out["subagent_ids"][0]
        state, pinned, record = _db_rows(parent)[sid]
        assert state == "resumable", done
        tree = cwds[0]
        assert os.path.realpath(tree) != os.path.realpath(repo)  # isolation engaged
        assert os.path.isfile(os.path.join(tree, "work.txt")), done  # not archived away by the run's finalize
        assert pinned == 1 and os.path.realpath(tree) in record["pin_paths"]

        _forget_process_state()
        cont = json.loads(delegate_task(action="continue", subagent_id=sid, message="go on", background=True,
                                        parent_agent=parent))
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    assert os.path.realpath(cwds[1]) == os.path.realpath(tree)
    assert trees[1].count(".worktrees/") == 1, trees[1]  # the continue made no second tree
    assert not os.path.exists(tree)  # once it completes, the run's finalize archives and removes it as before


def test_a_continue_racing_onto_the_durable_row_does_not_run_the_child_twice(monkeypatch):
    """B arrives between A's in-process claim and A's durable claim, so B falls through to the durable row."""
    import hermes_agent.tools.delegate_tool_resumable as resumable
    from hermes_agent.tools.delegate_tool_continue import _claim
    from hermes_agent.tools.delegate_tool_registry import _recent_subagents

    parent = _parent()
    with patch("hermes_agent.run_agent.AIAgent", side_effect=_ScriptedModel()):
        sid, _ = _spawn(parent, "write the parser")

    real_claim, raced = resumable.claim, []

    def claim(agent, subagent_id):
        if not raced:
            raced.append(None)
            raced[0] = _claim(subagent_id, agent)
        return real_claim(agent, subagent_id)

    monkeypatch.setattr(resumable, "claim", claim)
    first = _claim(sid, parent)
    winners = [got for got in (first, raced[0]) if got[0] is not None]
    assert len(winners) == 1, (first, raced)
    loser = first if first[0] is None else raced[0]
    assert "being continued" in loser[2], loser
    assert _recent_subagents[sid].get("resume")  # the refused caller handed its in-process spec back


def test_a_continue_that_fails_before_its_run_records_itself_hands_the_claim_back():
    from hermes_agent.tools.delegate_tool import delegate_task
    import hermes_agent.tools.delegate_tool as dt

    parent = _parent()
    model = _ScriptedModel()
    real_lease = dt._lease_child_credential
    with patch("hermes_agent.run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(parent, "write the parser")
        _forget_process_state()
        with patch.object(dt, "_lease_child_credential", side_effect=RuntimeError("pool gone")):
            cont = json.loads(delegate_task(action="continue", subagent_id=sid, message="go on", background=True,
                                            parent_agent=parent))
            assert cont.get("status") == "dispatched", cont
            _completion(cont["delegation_id"])
        assert _db_rows(parent)[sid][0] == "completed"  # not left `claimed` by this live process

        with patch.object(dt, "_lease_child_credential", side_effect=real_lease):
            again = json.loads(delegate_task(action="continue", subagent_id=sid, message="go on", background=True,
                                             parent_agent=parent))
            assert again.get("status") == "dispatched", again
            _completion(again["delegation_id"])
    assert len(model.turns) == 2


@pytest.mark.parametrize("owner", ["dead", "alive"])
def test_claim_reclaims_a_row_whose_owner_died_after_boot_recovery(owner):
    from hermes_agent.gateway.status import get_process_start_time
    from hermes_agent.tools.delegate_tool import delegate_task

    parent = _parent()
    model = _ScriptedModel()
    with patch("hermes_agent.run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(parent, "write the parser")
        pid = _DEAD_PID if owner == "dead" else os.getpid()
        _set_row(parent, sid, state="running", owner_pid=pid,
                 owner_started_at=None if owner == "dead" else get_process_start_time(pid))
        _forget_process_state()
        cont = json.loads(delegate_task(action="continue", subagent_id=sid, message="go on", background=True,
                                        parent_agent=parent))
        if owner == "alive":
            assert "still running" in cont["error"], cont
            assert _db_rows(parent)[sid][0] == "running"
            return
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    assert len(model.turns) == 2 and _db_rows(parent)[sid][0] == "completed"


def test_boot_recovery_holds_no_write_lock_while_it_probes_git(monkeypatch):
    import hermes_agent.tools.delegate_tool_resumable as resumable
    from hermes_constants import get_hermes_home
    from hermes_agent.hermes_state import SessionDB
    from hermes_agent.tools.async_delegation import recover_abandoned_delegations

    db = get_hermes_home() / "state.db"
    SessionDB(db_path=db).close()
    conn = sqlite3.connect(db)  # health: allow HX015 -- test seeds a temp-home store
    with conn:
        for sid in ("sa-dead-1", "sa-dead-2"):
            resumable._upsert(conn, sid, {"session_id": "s", "cwd": None, "owner_pid": _DEAD_PID, "owners": ["p"]},
                              state="running", pinned=True, expires_at=time.time() + 3600)
    conn.close()

    real, blocked = resumable._pin_paths, []

    def probing(record):
        other = sqlite3.connect(db, timeout=0.2)  # health: allow HX015 -- another writer during the probe
        try:
            other.execute("BEGIN IMMEDIATE")
            other.rollback()
        except sqlite3.OperationalError as exc:
            blocked.append(str(exc))
        finally:
            other.close()
        return real(record)

    monkeypatch.setattr(resumable, "_pin_paths", probing)
    recover_abandoned_delegations()
    assert not blocked, blocked
    conn = sqlite3.connect(db)  # health: allow HX015 -- test reads a temp-home store
    states = dict(conn.execute("SELECT subagent_id, state FROM resumable_subagents"))
    conn.close()
    assert states == {"sa-dead-1": "resumable", "sa-dead-2": "resumable"}


@pytest.mark.parametrize("stopped", ["recorded", "crashed"])
def test_a_continue_resumes_the_childs_newest_conversation_segment(stopped):
    """``compression.in_place: false``: the child's conversation moved to a new session mid-run."""
    from hermes_agent.tools.delegate_tool import delegate_task

    model = _ScriptedModel()
    rotated = []

    def build(**kw):
        child = model(**kw)
        ok = child.run_conversation.side_effect
        db = kw["session_db"]

        def run_conversation(user_message, **rest):
            result = ok(user_message, **rest)
            if not rotated:  # the compression rotation, as the loop publishes it
                old, new = child.session_id, f"{child.session_id}-seg2"
                db.end_session(old, "compression")
                db.create_session(new, "subagent", parent_session_id=old,
                                  model_config={"_delegate_from": "parent-sess", "_subagent_id": child._subagent_id})
                db.append_message(new, "user", "[compressed] write the parser")
                db.append_message(new, "assistant", "after rotation")
                child.session_id = new
                rotated.append((old, new))
            return result
        child.run_conversation.side_effect = run_conversation
        return child

    parent = _parent()
    with patch("hermes_agent.run_agent.AIAgent", side_effect=build):
        sid, _ = _spawn(parent, "write the parser")
        old, new = rotated[0]
        if stopped == "crashed":  # the stop record never landed: the row still names the first segment
            conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- temp-home store
            with conn:
                conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.session_id', ?) "
                             "WHERE subagent_id=?", (old, sid))
            conn.close()
        else:
            assert _db_rows(parent)[sid][2]["session_id"] == new
        _forget_process_state()
        cont = json.loads(delegate_task(action="continue", subagent_id=sid, message="go on", background=True,
                                        parent_agent=parent))
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    assert model.builds[1]["session_id"] == new
    assert [m["content"] for m in model.turns[1]["history"]][-1] == "after rotation"


def test_each_profiles_pins_obey_that_profiles_own_cap():
    import hermes_agent.tools.delegate_tool_resumable as resumable
    from hermes_constants import get_hermes_home
    from hermes_agent.hermes_state import SessionDB

    default = get_hermes_home()
    work = default / "profiles" / "work"
    work.mkdir(parents=True)
    (default / "config.yaml").write_text(json.dumps({"delegation": {"max_resumable": 1}}))
    (work / "config.yaml").write_text(json.dumps({"delegation": {"max_resumable": 3}}))
    for home, n in ((default, 2), (work, 3)):
        SessionDB(db_path=home / "state.db").close()
        conn = sqlite3.connect(home / "state.db")  # health: allow HX015 -- test seeds a temp-home store
        with conn:
            for i in range(n):
                resumable._upsert(conn, f"sa-{home.name}-{i}", {"pin_paths": [str(home / f"tree-{i}")]},
                                  state="resumable", pinned=True, expires_at=time.time() + 3600)
        conn.close()

    pins = resumable.pinned_worktrees()  # the sweep, run under the default profile
    assert sum(1 for p in pins if p.startswith(os.path.realpath(work))) == 3, pins
    assert sum(1 for p in pins if not p.startswith(os.path.realpath(work))) == 1, pins
