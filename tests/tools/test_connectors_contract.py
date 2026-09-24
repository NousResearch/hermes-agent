"""Connection contract: target states, who may cause each transition, settle reasons.

The transition table is the single statement of the operation's lifecycle. ``operation.py``
enforces it; the renderer reads a generated copy (see the shared contract rail)."""


from tools.connectors import contract as c


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


def test_allowed_is_a_pure_lookup():
    assert c.allowed("connector", c.TargetState.initiated, c.TargetState.connected) == c.Actor.backend_watcher
    assert c.allowed("connector", c.TargetState.initiated, c.TargetState.expired) == c.Actor.clock
    assert c.allowed("mcp", c.TargetState.initiated, c.TargetState.connected) == c.Actor.backend_watcher
    assert c.allowed("mcp", c.TargetState.failed, c.TargetState.initiated) == c.Actor.user
    assert c.allowed("connector", c.TargetState.connected, c.TargetState.pending) is None


