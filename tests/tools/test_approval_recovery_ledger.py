"""Durable approval recovery ledger coverage."""

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError,
    claim_approval_recovery,
    consume_approval_recovery,
    finish_approval_recovery,
    record_approval_recovery,
)


KW = dict(
    principal_id="principal-a",
    session_id="session-a",
    request_id="approval-a",
    recovery_token="token-a",
    payload_digest="digest-a",
    execution_generation=7,
    turn_id="turn-a",
)


def test_recovery_ledger_survives_reopen_and_consumes_once(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    record_approval_recovery(db, **KW, status="expired")
    db.close()

    reopened = SessionDB(path)
    assert consume_approval_recovery(reopened, **KW) == {
        "status": "expired",
        "blocker": "approval expired; user consent was not received",
    }
    assert consume_approval_recovery(reopened, **KW)["status"] == "failed"
    reopened.close()


def test_approval_boundary_uses_durable_record_after_restart(tmp_path):
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    path = tmp_path / "state.db"
    db = SessionDB(path)
    approval.register_gateway_notify(
        "session-a", lambda _: None, recovery_db=db,
        principal_id="principal-a", durable_session_id="session-a")
    entry = _ApprovalEntry({
        "request_id": KW["request_id"],
        "command": "echo durable",
        "command_digest": KW["payload_digest"],
        "execution_generation": KW["execution_generation"],
        "turn_id": KW["turn_id"],
        "recovery_token": KW["recovery_token"],
    })
    approval._record_approval_recovery(entry, "session-a", "expired")
    db.close()
    reopened = SessionDB(path)
    assert approval.recover_gateway_approval(
        "session-a", KW["request_id"], command_digest=KW["payload_digest"],
        recovery_token=KW["recovery_token"], execution_generation=7, turn_id="turn-a",
        recovery_db=reopened, principal_id="principal-a", durable_session_id="session-a",
    )["status"] == "admitting"
    reopened.close()


def test_recovery_ledger_replay_is_idempotent_and_token_conflict_is_atomic(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    first = record_approval_recovery(db, **KW, status="pending")
    replay = record_approval_recovery(db, **KW, status="pending")
    assert replay["seq"] == first["seq"]
    with pytest.raises(RuntimeStoreError, match="recovery_conflict"):
        record_approval_recovery(db, **{**KW, "recovery_token": "other-token"}, status="pending")
    row = db._read_one(
        "SELECT recovery_token,status FROM approval_recovery_admissions WHERE request_id=?",
        (KW["request_id"],),
    )
    assert tuple(row) == (KW["recovery_token"], "pending")
    db.close()


def test_recovery_admission_replays_the_normal_submit_result(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    record_approval_recovery(db, **KW, command_snapshot="echo exact", status="expired")
    claimed = claim_approval_recovery(db, **KW)
    assert claimed["status"] == "admitting"
    result = {"status": "streaming", "user_row_id": 9}
    assert finish_approval_recovery(db, seq=claimed["seq"], status="streaming", result=result)["result"] == result
    assert claim_approval_recovery(db, **KW) == {"status": "replay", "result": result}
    db.close()
