"""Connection contract states and allowed transitions."""

import json

import pytest

from tools.connectors import contract as c
from tools.connectors.legs.mcp import apply_answer
from tools.connectors.operation import ConnectionOperation, IllegalTransition, Leg


def test_every_state_is_reachable_from_pending_in_some_kind():
    reachable = set()
    for kind in c.KINDS:
        assert (kind, c.LegState.pending) in c.TRANSITIONS
        frontier = [c.LegState.pending]
        seen = {c.LegState.pending}
        while frontier:
            state = frontier.pop()
            for nxt in c.TRANSITIONS.get((kind, state), {}):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        reachable |= seen
    assert reachable | {c.LegState.not_connected} == set(c.LegState)


def test_resolved_states_end_the_leg_and_are_never_left():
    for state in c.RESOLVED_STATES:
        for kind in c.KINDS:
            assert not c.TRANSITIONS.get((kind, state)), (kind, state)


def test_only_the_backend_watcher_may_report_connected():
    for (kind, _from), edges in c.TRANSITIONS.items():
        actor = edges.get(c.LegState.connected)
        if actor is not None:
            assert actor == c.Actor.backend_watcher, (kind, _from)
        assert edges.get(c.LegState.skipped) in {None, c.Actor.user}, (kind, _from)


def test_no_actor_but_the_backend_watcher_can_move_an_mcp_leg_to_connected():
    for actor in c.Actor:
        operation = ConnectionOperation([Leg("paper", "mcp", "enable")])
        operation.transition("paper", c.LegState.initiated, c.Actor.backend_watcher)
        if actor is c.Actor.backend_watcher:
            operation.transition("paper", c.LegState.connected, actor)
            assert operation.leg("paper").state == c.LegState.connected
            continue
        with pytest.raises(IllegalTransition):
            operation.transition("paper", c.LegState.connected, actor)


@pytest.mark.parametrize("claim", ["connected", "initiated", "failed", "expired", "not_connected"])
def test_the_card_word_never_transitions_a_leg(claim):
    operation = ConnectionOperation([Leg("paper", "mcp", "enable")])
    apply_answer(operation, json.dumps({"legs": [{"name": "paper", "status": claim}]}))
    assert operation.leg("paper").state == c.LegState.pending


def test_allowed_is_a_pure_lookup():
    assert c.allowed("connector", c.LegState.initiated, c.LegState.connected) == c.Actor.backend_watcher
    assert c.allowed("connector", c.LegState.initiated, c.LegState.expired) == c.Actor.clock
    assert c.allowed("mcp", c.LegState.initiated, c.LegState.connected) == c.Actor.backend_watcher
    assert c.allowed("mcp", c.LegState.failed, c.LegState.initiated) == c.Actor.user
    assert c.allowed("connector", c.LegState.connected, c.LegState.pending) is None


@pytest.mark.parametrize("value", ["active", "pending ", "CONNECTED", ""])
def test_leg_state_rejects_values_outside_the_enum(value):
    with pytest.raises(ValueError):
        c.LegState(value)
