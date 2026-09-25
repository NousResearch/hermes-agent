"""kanban_withdraw: the run that FILED a card retracts it before the dispatcher claims it.

Mechanism (a) of the card's done-when. ``create_task`` always received a
``creator_task_id`` but never persisted it, so no verb could prove authorship of a
card that is not the caller's own — and the lifecycle guard (correctly) refuses to
let a run close a card it is not working. This persists the provenance at insert
time and adds one verb scoped to it: the filer moves its own mis-filed duplicate to
the terminal ``archived`` state, out of dispatch, without gaining any authority
over another run's card.

The pin is the dispatcher tick: the withdrawn card is observed to stay unclaimed
while a sibling created in the same shape is claimed by that same tick.
"""
from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from tools import kanban_tools as kt


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """A worker run with its own card claimed, pointed at an empty board."""
    from hermes_cli import profiles

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    monkeypatch.setattr(kt, "_resolve_notify_target", lambda: None)
    kb.init_db()
    with kbc.connect_closing() as conn:
        # A finished parent, so the cards below are filed the way the incident's
        # were: linked, parked in todo, and promoted by the next tick.
        parent = kb.create_task(conn, title="filed brief", assignee="vx-dev")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        own = kb.create_task(conn, title="filer's own card", assignee="vx-dev")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (own,))
        assert kb.claim_task(conn, own, claimer="worker") is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", own)
    return own, parent


def _file(kt_args: dict) -> str:
    result = json.loads(kt._handle_create(kt_args))
    assert result["ok"], result
    return result["task_id"]


def _card(conn, tid: str):
    task = kb.get_task(conn, tid)
    assert task is not None, f"no card {tid}"
    return task


def test_filer_withdraws_its_own_duplicate_and_the_tick_skips_it(worker):
    """The pin: filed, withdrawn by its filer, never claimed across a dispatcher tick."""
    own, parent = worker
    with kbc.connect_closing() as conn:
        dup = _file({"title": "duplicate of the live card", "assignee": "vx-dev",
                     "parents": [parent]})
        live = _file({"title": "the live card", "assignee": "vx-dev",
                      "parents": [parent]})
        assert _card(conn, dup).creator_task_id == own
        assert _card(conn, live).creator_task_id == own

        result = json.loads(kt._handle_withdraw(
            {"task_id": dup, "reason": f"duplicate of {live} — filed twice by the same run"}))
        assert result["ok"], result
        assert result["withdrawn"] is True
        assert _card(conn, dup).status == "archived"

        spawned: list[str] = []

        def _spawn(task, workspace, **kw):
            spawned.append(task.id)
            return 9001

        res = kbd.dispatch_once(conn, spawn_fn=_spawn)
        assert res.spawned is not None
        assert live in spawned, "control card must be claimed — otherwise the tick proves nothing"
        assert dup not in spawned
        assert _card(conn, dup).status == "archived"
        # And it stays out: a second tick claims nothing further.
        spawned.clear()
        kbd.dispatch_once(conn, spawn_fn=_spawn)
        assert spawned == []

        events = [e for e in kb.list_events(conn, dup) if e.kind == "withdrawn"]
        assert len(events) == 1
        payload = events[0].payload or {}
        assert payload.get("reason") == f"duplicate of {live} — filed twice by the same run"
        assert payload.get("provenance") == own
        assert payload.get("status") == "archived"


def _refusal(raw: str) -> str:
    payload = json.loads(raw)
    assert "error" in payload, payload
    return payload["error"]


def test_card_filed_by_another_run_is_refused(worker, monkeypatch):
    """Provenance is matched against the row, never against a caller-supplied id."""
    own, _parent = worker
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_someoneelse")
    with kbc.connect_closing() as conn:
        theirs = _file({"title": "another run's card", "assignee": "vx-dev"})
    monkeypatch.setenv("HERMES_KANBAN_TASK", own)
    with kbc.connect_closing() as conn:
        error = _refusal(kt._handle_withdraw({"task_id": theirs, "reason": "mine now"}))
        assert "was not filed by" in error
        assert "Only the run that filed a card may withdraw it" in error
        assert _card(conn, theirs).status != "archived"
        assert not [e for e in kb.list_events(conn, theirs) if e.kind == "withdrawn"]


def test_operator_filed_card_has_no_provenance_to_match(worker):
    """A card filed outside any run (CLI/dashboard) is not a worker's to retract."""
    _own, _parent = worker
    with kbc.connect_closing() as conn:
        orphan = kb.create_task(conn, title="filed by an operator", assignee="vx-dev")
        error = _refusal(kt._handle_withdraw({"task_id": orphan}))
        assert "was not filed by" in error
        assert "none recorded" in error
        assert _card(conn, orphan).status != "archived"


def test_the_card_a_worker_is_working_is_refused(worker):
    """Withdrawing is the filer's verb, never a second exit for one's own card."""
    own, _parent = worker
    error = _refusal(kt._handle_withdraw({"task_id": own}))
    assert "is the card you are working" in error
    assert "kanban_complete" in error


def test_withdraw_outside_a_run_hands_over_to_the_operator_surface(worker, monkeypatch):
    """No HERMES_KANBAN_TASK: this caller is an operator, and hermes kanban archive is the verb."""
    _own, _parent = worker
    with kbc.connect_closing() as conn:
        card = _file({"title": "filed while the env named a run", "assignee": "vx-dev"})
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kbc.connect_closing() as conn:
        error = _refusal(kt._handle_withdraw({"task_id": card}))
        assert "needs the filing run's own" in error
        assert "hermes kanban archive" in error
        assert _card(conn, card).status != "archived"


def test_an_already_finished_card_is_refused(worker, monkeypatch):
    """A done/archived duplicate is already out of dispatch; its record is the evidence."""
    _own, _parent = worker
    with kbc.connect_closing() as conn:
        card = _file({"title": "duplicate", "assignee": "vx-dev"})
        assert json.loads(kt._handle_withdraw({"task_id": card, "reason": "dup"}))["ok"]
        error = _refusal(kt._handle_withdraw({"task_id": card}))
        assert "nothing left to withdraw" in error

        done = kb.create_task(conn, title="finished", assignee="vx-dev")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (done,))
        error = _refusal(kt._handle_withdraw({"task_id": done}))
        assert "is done: nothing left to withdraw" in error


def test_delegate_child_may_not_withdraw(worker, monkeypatch):
    """A delegate child inherits HERMES_KANBAN_* env but owns no run."""
    own, _parent = worker
    monkeypatch.setattr(kt, "_delegation_ctx", lambda predicate, default: True)
    error = _refusal(kt._handle_withdraw({"task_id": "t_whatever"}))
    assert "delegate_task child agents are not Kanban run owners" in error
    assert own  # the worker's own card was never touched either


def test_unknown_card_is_refused(worker):
    error = _refusal(kt._handle_withdraw({"task_id": "t_nosuchcard"}))
    assert "unknown task" in error


def test_task_id_is_required(worker):
    error = _refusal(kt._handle_withdraw({}))
    assert "task_id is required" in error
