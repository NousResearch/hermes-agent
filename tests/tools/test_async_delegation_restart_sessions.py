"""Regression for #121544: interrupted detached children must not remain active after restart."""
import json
import time

from tools import async_delegation as ad


def test_restart_recovers_exact_children_and_shutdown_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
    now = time.time()
    with ad._DB_LOCK, ad._transaction() as conn:
        for sid, source, ended_at in (
            ("child-a", "subagent", None), ("child-b", "subagent", now - 1),
            ("unrelated", "subagent", None), ("other-source", "cli", None),
        ):
            conn.execute("INSERT INTO sessions(id, source, started_at, ended_at, parent_session_id) VALUES (?, ?, ?, ?, ?)",
                         (sid, source, now - 10, ended_at, "shared-parent"))
        for name, task in (
            ("graceful", {"child_session_ids": {"0": "child-a", "1": "child-b", "2": "other-source"}}),
            ("crashed", {"child_session_ids": {"0": "unrelated"}}),
        ):
            conn.execute("""INSERT INTO async_delegations
                         (delegation_id, origin_session, parent_session_id, state, dispatched_at,
                          updated_at, delivery_state, owner_pid, task_json)
                         VALUES (?, '', 'shared-parent', 'running', ?, ?, 'pending', ?, ?)""",
                         (name, now - 10, now - 10, 99999999, json.dumps(task)))
    monkeypatch.setattr(ad, "_records", {"graceful": {"delegation_id": "graceful", "status": "running",
                                                    "interrupt_fn": lambda: None}})
    assert ad.interrupt_all("gateway shutdown") == 1
    assert ad.recover_abandoned_delegations() == 2
    assert ad.recover_abandoned_delegations() == 0
    with ad._DB_LOCK, ad._transaction() as conn:
        states = dict(conn.execute("SELECT delegation_id, state FROM async_delegations"))
        rows = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT id, ended_at, end_reason FROM sessions")}
        events = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT delegation_id, event_json FROM async_delegations")}
    assert states == {"graceful": "interrupted", "crashed": "unknown"}
    assert events["graceful"]["status"] == "interrupted"
    assert events["crashed"]["status"] == "unknown"
    assert rows["child-a"][0] is not None and rows["child-a"][1] == "interrupted"
    assert rows["unrelated"][0] is not None and rows["unrelated"][1] == "interrupted"
    assert rows["child-b"][0] == now - 1
    assert rows["other-source"][0] is None
