"""Unsupported prompt observations remain visible data, never decision authority."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import hosted_room_messaging_approvals as approvals
from gateway.hosted_room_driver import TaskIdentity
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime


def observation(last=False):
    prompt = {"kind": "approval", "prompt_id": "prompt", "request_id": "prompt", "description": "Review command",
              "command": "fixture command", "choices": ["once", "deny"], "control_supported": False,
              "admission_id": "admission", "target_execution_generation": 17, "execution_generation": 17}
    info = {"run_id": "run", "admission_id": "admission", "target_execution_generation": 17,
            "control_supported": False}
    info["last_observed_pending_controls" if last else "approval"] = [prompt] if last else prompt
    actions = []
    runtime = object.__new__(HostedRoomRuntime)
    runtime.pending_action = lambda _room, _member, action: actions.append(action)
    runtime._leases = {}
    runtime.process_generation = "observer"
    task = {"identity": TaskIdentity("room", "task", "thread", "turn"), "execution_generation": 3,
            "payload": {"target_member_id": "member", "target_profile": "ops"}}
    runtime._report_pending_action(HostedRoomBinding("room", "home", 1), task, session_id="session", info=info)
    return actions[0]


@pytest.mark.parametrize("last", [False, True])
def test_driver_and_normalizer_preserve_nonactionable_target_evidence(last):
    raw = observation(last)
    assert raw["kind"] == "approval"
    assert raw["approval"]["choices"] == []
    result = approvals.normalize_pending_approval("room", "member", raw)
    assert result["control_supported"] is False
    assert result["admission_id"] == "admission"
    assert result["target_execution_generation"] == 17
    assert result["execution_generation"] == 3
    assert result["approval"]["choices"] == []


def test_persisted_observation_has_no_picker_or_decision_and_survives_ordinary_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(approvals, "_require_observer_lease", Mock())
    db = tmp_path / "metadata.db"
    pending = approvals.persist_pending_approval(db, room_id="room", member_id="member", action=observation())
    rows = approvals.list_pending_approvals(db, room_id="room")
    assert rows == [pending]
    conn = approvals._connect(db)
    try:
        conn.execute("UPDATE hosted_room_pending_approvals SET updated_at=0")
        approvals._prune_locked(conn, now=approvals.PENDING_APPROVAL_TTL_SECONDS + 10)
        conn.commit()
    finally:
        conn.close()
    rows = approvals.list_pending_approvals(db, room_id="room")
    assert rows == [pending]
    service = SimpleNamespace(status=lambda _: {"pending_actions": rows})
    room = {"room_id": "room", "name": "Room", "authority_gateway_id": "home", "authority_epoch": 1,
            "members": [{"member_id": "member", "profile": "ops", "handle": "ops"}]}
    text = approvals.format_pending_approvals(service, room, room_reference="room")
    assert "Review command" in text and "unavailable" in text.lower()
    assert "Approve once:" not in text and "Deny:" not in text
    assert approvals.approval_picker_choices(room, rows) == []
    monkeypatch.setattr(approvals, "_connect", Mock(side_effect=AssertionError("decision must refuse before storage")))
    with pytest.raises(approvals.MessagingApprovalError, match="unavailable"):
        approvals.begin_approval_command(db, command_id="command", pending=pending, choice="once")
    with pytest.raises(approvals.MessagingApprovalError, match="unavailable"):
        approvals.apply_pending_decision(db, command_id="already-queued", pending=pending, choice="once",
                                         apply=Mock(side_effect=AssertionError("control is held")))
    approvals._connect.assert_not_called()


def test_nonactionable_observation_does_not_change_legacy_choices():
    raw = observation()
    supported = deepcopy(raw)
    supported.pop("control_supported", None)
    supported.pop("admission_id", None)
    supported.pop("target_execution_generation", None)
    supported["approval"] = {"description": "Legacy", "command": "fixture", "choices": ["once", "deny"]}
    result = approvals.normalize_pending_approval("room", "member", supported)
    assert result["approval"]["choices"] == ["once", "deny"]
    assert "control_supported" not in result


def test_existing_pending_approvals_survive_schema_upgrade(tmp_path, monkeypatch):
    monkeypatch.setattr(approvals, "_require_observer_lease", Mock())
    db = tmp_path / "metadata.db"
    action = observation()
    for key in ("control_supported", "admission_id", "target_execution_generation"):
        action.pop(key, None)
    action["approval"] = {"description": "Legacy", "command": "fixture", "choices": ["once", "deny"]}
    pending = approvals.persist_pending_approval(db, room_id="room", member_id="member", action=action)
    conn = approvals._connect(db)
    try:
        # Restore the pre-change schema while retaining the actual stored row.
        for column in ("control_supported", "target_admission_id", "target_execution_generation"):
            conn.execute(f"ALTER TABLE hosted_room_pending_approvals DROP COLUMN {column}")
        conn.commit()
    finally:
        conn.close()
    assert approvals.list_pending_approvals(db, room_id="room") == [pending]
    retained = approvals.persist_pending_approval(db, room_id="room", member_id="member", action=observation())
    assert approvals.list_pending_approvals(db, room_id="room") == [retained]
    assert retained["control_supported"] is False
    assert retained["target_execution_generation"] == 17


@pytest.mark.parametrize("field,value", [("admission_id", "other"), ("target_execution_generation", 18)])
def test_conflicting_target_evidence_is_rejected(field, value):
    action = observation()
    action["approval"][field] = value
    with pytest.raises(approvals.MessagingApprovalError, match="identity changed"):
        approvals.normalize_pending_approval("room", "member", action)
