"""Connection contract: target states, who may cause each transition, settle reasons.

The transition table is the single statement of the operation's lifecycle. ``operation.py``
enforces it; the renderer reads a generated copy (see the shared contract rail)."""

import json

import pytest

from tools.connectors import contract as c
from tools.connectors.mcp import apply_answer
from tools.connectors.operation import ConnectionOperation, IllegalTransition, Target


def test_every_state_is_reachable_from_pending_in_some_kind():
    reachable = set()
    for kind in c.KINDS:
        assert (kind, c.TargetState.pending) in c.TRANSITIONS
        frontier = [c.TargetState.pending]
        seen = {c.TargetState.pending}
        while frontier:
            state = frontier.pop()
            for nxt in c.TRANSITIONS.get((kind, state), {}):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        reachable |= seen
    # not_connected is stamped by settle(), never transitioned to.
    assert reachable | {c.TargetState.not_connected} == set(c.TargetState)


def test_resolved_states_end_the_target_and_are_never_left():
    for state in c.RESOLVED_STATES:
        for kind in c.KINDS:
            assert not c.TRANSITIONS.get((kind, state)), (kind, state)


def test_only_the_backend_watcher_may_report_connected():
    # The card renders the operation; it never witnesses an outcome, whatever the kind.
    for (kind, _from), edges in c.TRANSITIONS.items():
        actor = edges.get(c.TargetState.connected)
        if actor is not None:
            assert actor == c.Actor.backend_watcher, (kind, _from)
        assert edges.get(c.TargetState.skipped) in {None, c.Actor.user}, (kind, _from)


def test_no_actor_but_the_backend_watcher_can_move_an_mcp_target_to_connected():
    """Enforcement, not the table: an install, an enable and an OAuth flow are all witnessed by the
    backend, so every other actor is refused the move."""
    for actor in c.Actor:
        operation = ConnectionOperation([Target("paper", "mcp", "enable")])
        operation.transition("paper", c.TargetState.initiated, c.Actor.backend_watcher)
        if actor is c.Actor.backend_watcher:
            operation.transition("paper", c.TargetState.connected, actor)
            assert operation.target("paper").state == c.TargetState.connected
            continue
        with pytest.raises(IllegalTransition):
            operation.transition("paper", c.TargetState.connected, actor)


@pytest.mark.parametrize("claim", ["connected", "initiated", "failed", "expired", "not_connected"])
def test_the_card_s_word_never_transitions_a_target(claim):
    """The card renders the operation. Approved, skipped and Continue are its whole vocabulary;
    any other claim in ``connection.respond`` moves nothing."""
    operation = ConnectionOperation([Target("paper", "mcp", "enable")])
    apply_answer(operation, json.dumps({"targets": [{"name": "paper", "status": claim}]}))
    assert operation.target("paper").state == c.TargetState.pending


def test_allowed_is_a_pure_lookup():
    assert c.allowed("connector", c.TargetState.initiated, c.TargetState.connected) == c.Actor.backend_watcher
    assert c.allowed("connector", c.TargetState.initiated, c.TargetState.expired) == c.Actor.clock
    assert c.allowed("mcp", c.TargetState.initiated, c.TargetState.connected) == c.Actor.backend_watcher
    assert c.allowed("mcp", c.TargetState.failed, c.TargetState.initiated) == c.Actor.user
    assert c.allowed("connector", c.TargetState.connected, c.TargetState.pending) is None


@pytest.mark.parametrize("value", ["active", "pending ", "CONNECTED", ""])
def test_target_state_rejects_values_outside_the_enum(value):
    with pytest.raises(ValueError):
        c.TargetState(value)
