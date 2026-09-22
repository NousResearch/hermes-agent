"""Connection operation: transitions checked against the contract, exactly-once settlement,
server-owned deadline as a constant, wake on every change."""

import pytest

from tools.connectors import contract as c
from tools.connectors import operation as op


def _two(kind="connector"):
    return [op.Target("gmail", kind, "connect"), op.Target("notion", kind, "connect")]


def test_deadline_is_a_constant_not_a_config_key():
    operation = op.ConnectionOperation(_two())
    assert operation.deadline_at == pytest.approx(operation.created_at + op.OPERATION_DEADLINE_SECONDS)
    assert op.OPERATION_DEADLINE_SECONDS == 300
    assert not hasattr(op, "resolve_wait_timeout")


def test_transition_enforces_the_contract_and_names_the_actor():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher)
    with pytest.raises(op.IllegalTransition):
        operation.transition("gmail", c.TargetState.connected, c.Actor.user)  # a card cannot claim connected
    with pytest.raises(op.IllegalTransition):
        operation.transition("gmail", c.TargetState.pending, c.Actor.backend_watcher)  # no edge back
    operation.transition("gmail", c.TargetState.connected, c.Actor.backend_watcher)
    assert operation.target("gmail").state == c.TargetState.connected


def test_transition_returns_the_change_and_wakes_the_waiter():
    operation = op.ConnectionOperation(_two())
    assert not operation.wake.is_set()
    change = operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher, detail="link minted")
    assert change == {"target": "gmail", "from": "pending", "to": "initiated", "actor": "backend_watcher", "detail": "link minted"}
    assert operation.wake.is_set()


def test_same_state_transition_is_a_no_op_not_an_error():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher)
    operation.wake.clear()
    assert operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher) is None
    assert not operation.wake.is_set()


def test_settle_is_exactly_once_and_stamps_not_connected():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher)
    operation.transition("gmail", c.TargetState.connected, c.Actor.backend_watcher)
    assert operation.settle(c.SettleReason.continue_) is True
    frozen = operation.result()
    assert operation.settle(c.SettleReason.deadline) is False
    assert operation.result() == frozen
    by = {t["name"]: t for t in frozen["targets"]}
    assert by["gmail"]["state"] == "connected"
    # The settle reason is on the operation, never copied into a row's detail (the card showed it as red text).
    assert by["notion"]["state"] == "not_connected" and "detail" not in by["notion"]


def test_all_resolved_settles_on_connected_or_skipped_only():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher)
    operation.transition("gmail", c.TargetState.failed, c.Actor.backend_watcher, detail="oauth denied")
    operation.transition("notion", c.TargetState.skipped, c.Actor.user)
    assert operation.settle_if_all_resolved() is False  # failed keeps the op open
    operation.transition("gmail", c.TargetState.skipped, c.Actor.user)
    assert operation.settle_if_all_resolved() is True
    assert operation.settled_by == c.SettleReason.all_resolved


def test_target_keeps_the_link_and_the_mint_detail_across_transitions():
    operation = op.ConnectionOperation(_two())
    operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher,
                         connect_url="https://link/gmail", detail="")
    operation.transition("gmail", c.TargetState.failed, c.Actor.backend_watcher, detail="vendor said no")
    snap = operation.target("gmail").snapshot()
    assert snap["connect_url"] == "https://link/gmail"
    assert snap["detail"] == "vendor said no"


def test_request_payload_carries_the_live_target_snapshot():
    operation = op.ConnectionOperation(
        [op.Target("gmail", "connector", "reconnect", instructions="Finish setup")],
        tool_call_id="call-1")
    operation.transition("gmail", c.TargetState.initiated, c.Actor.backend_watcher, connect_url="https://l/gmail")
    payload = operation.request_payload()
    (target,) = payload["targets"]
    assert target == {"name": "gmail", "kind": "connector", "action": "reconnect", "state": "initiated",
                      "instructions": "Finish setup", "connect_url": "https://l/gmail"}
    # The model's own id keys the card to its tool row; a later op for the same apps gets a new one.
    assert payload["tool_call_id"] == "call-1"
    assert "reason" not in payload
