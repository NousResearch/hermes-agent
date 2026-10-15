"""Standalone ``serve`` (no session authority) must not run a legacy turn past the gateway's
durable ledger: an ``unknown`` admission is never replayed, queued input runs in FIFO order."""

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input, begin_runtime_epoch, claim_session_input, list_session_admissions,
    recover_session_inputs, resolve_unknown_session_input,
)
from tui_gateway import server


class _InlineThread:
    def __init__(self, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


@pytest.mark.parametrize("ledger", ["unknown", "queued"])
def test_legacy_submit_refuses_while_canonical_ledger_holds_work(monkeypatch, tmp_path, ledger):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    for name in ("_schedule_agent_build", "_schedule_session_cap_enforcement", "_register_session_cwd",
                 "_start_agent_build"):
        monkeypatch.setattr(server, name, lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a: False)
    turns = []
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *a: turns.append(a[1]))
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    created = server.handle_request({"id": "c", "method": "session.create", "params": {"source": "desktop"}})
    sid, key = created["result"]["session_id"], created["result"]["stored_session_id"]
    db.create_session(key, "cli")
    first = begin_runtime_epoch(db, instance_id="gateway-1")
    admit_session_input(db, epoch=first, principal_id="owner", session_id=key, request_id="r1",
                        payload={"text": "lost?"})
    epoch = first
    if ledger == "unknown":
        claim_session_input(db, epoch=first, session_id=key)
        epoch = begin_runtime_epoch(db, instance_id="gateway-2")
        recover_session_inputs(db, epoch=epoch)

    def submit():
        return server.handle_request({"id": "p", "method": "prompt.submit", "params": {"session_id": sid, "text": "hi"}})

    try:
        refused = submit()
        assert refused["error"]["code"] == 4090, refused
        assert refused["error"]["data"] == {"reason": "SESSION_NOT_OWNED"}
        assert turns == [] and db.get_messages_as_conversation(key) == []
        if ledger == "unknown":
            # Once the operator discards the lost turn through the gateway the barrier lifts.
            (row,) = list_session_admissions(db, session_id=key)
            resolve_unknown_session_input(db, epoch=epoch, admission_id=row["admission_id"],
                                          generation=row["generation"])
            assert submit()["result"]["status"] == "streaming"
            assert turns == [sid]
    finally:
        server._sessions.pop(sid, None)
        db.close()
