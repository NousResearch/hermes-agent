"""Connection operation transitions, settlement, and payload snapshots."""

import pytest

from tools.connectors import contract as c
from tools.connectors import operation as op
from tools.operations import Owner


def _two(kind="connector"):
    return [op.Leg("gmail", kind, "connect"), op.Leg("notion", kind, "connect")]


def test_deadline_is_a_constant_not_a_config_key():
    operation = op.ConnectionOperation(_two())
    assert operation.deadline_at == pytest.approx(operation.created_at + op.OPERATION_DEADLINE_SECONDS)
    assert op.OPERATION_DEADLINE_SECONDS == 300
    assert not hasattr(op, "resolve_wait_timeout")


def test_connection_operation_owner_is_stamped_when_opened():
    """Construction does not fix the profile: the tool thread builds the operation before it enters
    the turn's override, so the table stamps the owner at open, from the opening thread."""
    from tools.operations import operations

    operation = op.ConnectionOperation(_two(), session_key="session")
    assert operation.owner is None
    operations.open(operation)
    try:
        assert operation.owner == Owner.current("session")
    finally:
        operations.close(operation)


def test_transition_enforces_the_contract_and_names_the_actor():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher)
    with pytest.raises(op.IllegalTransition):
        operation.transition("gmail", c.LegState.connected, c.Actor.user)
    with pytest.raises(op.IllegalTransition):
        operation.transition("gmail", c.LegState.pending, c.Actor.backend_watcher)
    operation.transition("gmail", c.LegState.connected, c.Actor.backend_watcher)
    assert operation.leg("gmail").state == c.LegState.connected


def test_transition_returns_the_change_and_wakes_the_waiter():
    operation = op.ConnectionOperation(_two())
    assert not operation.wake.is_set()
    change = operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher, detail="link minted")
    assert change == {"leg": "gmail", "from": "pending", "to": "initiated", "actor": "backend_watcher", "detail": "link minted"}
    assert operation.wake.is_set()


def test_same_state_transition_is_a_no_op_not_an_error():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher)
    operation.wake.clear()
    assert operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher) is None
    assert not operation.wake.is_set()


def test_settle_is_exactly_once_and_stamps_not_connected():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher)
    operation.transition("gmail", c.LegState.connected, c.Actor.backend_watcher)
    assert operation.settle(c.SettleReason.continue_) is True
    frozen = operation.result()
    assert operation.settle("deadline") is False
    assert operation.result() == frozen
    by = {leg["name"]: leg for leg in frozen["legs"]}
    assert by["gmail"]["state"] == "connected"
    assert by["notion"]["state"] == "not_connected" and "detail" not in by["notion"]


def test_all_resolved_settles_on_connected_or_skipped_only():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher)
    operation.transition("gmail", c.LegState.failed, c.Actor.backend_watcher, detail="oauth denied")
    operation.transition("notion", c.LegState.skipped, c.Actor.user)
    assert operation.settle_if_all_resolved() is False
    operation.transition("gmail", c.LegState.skipped, c.Actor.user)
    assert operation.settle_if_all_resolved() is True
    assert operation.settled_by == c.SettleReason.all_resolved.value
    assert operation.settle_reason == c.SettleReason.all_resolved


def test_leg_keeps_the_link_and_the_mint_detail_across_transitions():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher, connect_url="https://link/gmail", detail="")
    operation.transition("gmail", c.LegState.failed, c.Actor.backend_watcher, detail="vendor said no")
    snap = operation.leg("gmail").snapshot()
    assert snap["connect_url"] == "https://link/gmail"
    assert snap["detail"] == "vendor said no"


def test_snapshot_carries_the_live_leg_snapshot():
    operation = op.ConnectionOperation([op.Leg("gmail", "connector", "reconnect")], tool_call_id="call-1")
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher, connect_url="https://l/gmail")
    payload = operation.snapshot()
    (leg,) = payload["legs"]
    assert leg == {"name": "gmail", "kind": "connector", "action": "reconnect", "state": "initiated", "connect_url": "https://l/gmail"}
    assert payload == operation.request_payload()
    assert payload["tool_call_id"] == "call-1"
    assert "reason" not in payload


def test_seq_rises_on_every_write_and_appears_in_both_payloads():
    operation = op.ConnectionOperation(_two())
    assert operation.request_payload()["seq"] == 0
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher)
    minted = operation.request_payload()["seq"]
    operation.refresh("gmail", connect_url=None, detail="the link died")
    refreshed = operation.result()["seq"]
    operation.settle(c.SettleReason.continue_)
    assert 0 < minted < refreshed < operation.result()["seq"]


def test_every_emitted_frame_carries_the_state_its_seq_names(monkeypatch):
    operation = op.ConnectionOperation(_two())
    frames = []

    def hook(_operation, _change, snapshot):
        frames.append(snapshot)
        if len(frames) == 1:
            operation.transition("notion", c.LegState.initiated, c.Actor.backend_watcher)

    monkeypatch.setattr(op.ConnectionOperation, "on_change", staticmethod(hook))
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher)

    by_seq = {snap["seq"]: {leg["name"]: leg["state"] for leg in snap["legs"]} for snap in frames}
    assert by_seq[1] == {"gmail": "initiated", "notion": "pending"}
    assert by_seq[2] == {"gmail": "initiated", "notion": "initiated"}


def test_a_write_after_settlement_keeps_the_settle_frame_seq(monkeypatch):
    operation = op.ConnectionOperation(_two())
    frames = []
    monkeypatch.setattr(op.ConnectionOperation, "on_change", staticmethod(lambda _o, _c, snapshot: frames.append(snapshot)))
    operation.transition("gmail", c.LegState.initiated, c.Actor.backend_watcher)
    operation.settle(c.SettleReason.continue_)
    settled_seq = frames[-1]["seq"]

    operation.refresh("gmail", connect_url=None, detail="the provider answered late")

    assert frames[-1]["seq"] == settled_seq
    assert operation.request_payload()["seq"] == settled_seq
