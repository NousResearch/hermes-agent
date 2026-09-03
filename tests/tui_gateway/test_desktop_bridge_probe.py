"""A desktop client that cannot answer a bridged tool request must not cost a
full bridge timeout per call — and a merely slow one must not be condemned.

The renderer's handlers ship in the desktop bundle; the tools are offered by
the backend. An app build older than the tool has no branch for the event, so
nothing ever responds and the agent blocks for the whole deadline — once per
action the model tries. See tui_gateway.server._probed_bridge_request.

Both bridges (tour, screen annotations) run through that one helper, so the
contract is pinned for both call paths here.
"""

import json

import pytest

import tui_gateway.server as server

# (label, entry point, state key, probe deadline, full deadline, wire event).
BRIDGES = [
    pytest.param(
        server._tour_request,
        "tour_bridge",
        server._TOUR_PROBE_TIMEOUT_S,
        server._TOUR_TIMEOUT_S,
        "tour.request",
        id="tour",
    ),
    pytest.param(
        server._annotate_screen_request,
        "screen_annotate_bridge",
        server._ANNOTATE_SCREEN_PROBE_TIMEOUT_S,
        server._ANNOTATE_SCREEN_TIMEOUT_S,
        "screen.annotate.request",
        id="annotate",
    ),
    pytest.param(
        server._subtitles_control_request,
        "subtitles_bridge",
        server._SUBTITLES_CONTROL_PROBE_TIMEOUT_S,
        server._SUBTITLES_CONTROL_TIMEOUT_S,
        "subtitles.control.request",
        id="subtitles",
    ),
]


@pytest.fixture
def session(monkeypatch):
    record = {}
    monkeypatch.setitem(server._sessions, "s1", record)
    return record


@pytest.fixture
def bridge(monkeypatch):
    """Record every _block call and serve canned answers."""
    calls = []

    def fake_block(event, sid, payload, timeout=None, **_kw):
        answer = fake_block.answers.pop(0) if fake_block.answers else ""
        calls.append({"event": event, "sid": sid, "payload": payload, "timeout": timeout})
        return answer

    fake_block.answers = []
    fake_block.calls = calls
    monkeypatch.setattr(server, "_block", fake_block)
    return fake_block


OK = json.dumps({"success": True})


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_first_action_is_probed_on_a_short_deadline(
    session, bridge, request_fn, state_key, probe_s, full_s, event
):
    bridge.answers = [OK]
    request_fn("s1", {"action": "targets"})

    assert bridge.calls[0]["event"] == event
    assert bridge.calls[0]["timeout"] == probe_s
    assert probe_s < full_s


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_a_client_that_answers_gets_the_full_deadline_back(
    session, bridge, request_fn, state_key, probe_s, full_s, event
):
    """The generous deadline exists for a real first-time cost (injecting the
    tour engine, spawning the overlay); only an unproven client is held to the
    probe."""
    bridge.answers = [OK, OK]
    request_fn("s1", {"action": "targets"})
    request_fn("s1", {"action": "show", "selector": "#composer"})

    assert bridge.calls[1]["timeout"] == full_s


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_one_missed_probe_asks_for_a_retry_rather_than_blaming_the_build(
    session, bridge, request_fn, state_key, probe_s, full_s, event
):
    """The probe deadline judges the very call that pays the first-time cost,
    so a single miss is not evidence the app is too old. Saying so would hand
    the user a fix they cannot apply and disable the tool for the session."""
    result = json.loads(request_fn("s1", {"action": "targets"}))

    assert result["success"] is False
    assert "update" not in result["error"].lower()
    assert "once more" in result["error"].lower()
    assert session.get(state_key) != "dead"


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_a_slow_first_call_still_gets_a_second_chance(
    session, bridge, request_fn, state_key, probe_s, full_s, event
):
    """The regression this guards: one cold-start miss latching the bridge dead
    for the whole session on a perfectly capable client."""
    assert json.loads(request_fn("s1", {"action": "targets"}))["success"] is False

    bridge.answers = [OK]
    assert json.loads(request_fn("s1", {"action": "targets"}))["success"] is True
    assert len(bridge.calls) == 2
    assert bridge.calls[1]["timeout"] == probe_s


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_a_client_that_answers_clears_its_strike(
    session, bridge, request_fn, state_key, probe_s, full_s, event
):
    """Strikes count consecutive misses; a proven client starts clean, so a
    later miss cannot combine with an old one to condemn it."""
    request_fn("s1", {"action": "targets"})
    bridge.answers = [OK]
    request_fn("s1", {"action": "targets"})

    # Miss again on the now-proven client, then confirm it still bridges.
    assert json.loads(request_fn("s1", {"action": "next"}))["success"] is False
    bridge.answers = [OK]
    assert json.loads(request_fn("s1", {"action": "next"}))["success"] is True
    assert len(bridge.calls) == 4


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_two_unanswered_probes_explain_the_real_problem(
    session, bridge, request_fn, state_key, probe_s, full_s, event
):
    request_fn("s1", {"action": "targets"})
    result = json.loads(request_fn("s1", {"action": "targets"}))

    assert result["success"] is False
    assert "desktop" in result["error"].lower()
    assert "update" in result["error"].lower()


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_later_actions_short_circuit_instead_of_stalling_again(
    session, bridge, request_fn, state_key, probe_s, full_s, event
):
    """The original regression: one turn stacks a timeout per action."""
    for action in ("targets", "show"):
        assert json.loads(request_fn("s1", {"action": action}))["success"] is False
    for action in ("start", "next", "stop", "show"):
        assert json.loads(request_fn("s1", {"action": action}))["success"] is False

    assert len(bridge.calls) == server._BRIDGE_PROBE_STRIKES


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_a_proven_client_is_never_condemned(
    session, bridge, request_fn, state_key, probe_s, full_s, event
):
    """A live renderer that misses repeatedly is having action-level trouble,
    not a missing handler — it keeps the bridge."""
    bridge.answers = [OK]
    request_fn("s1", {"action": "targets"})

    for _ in range(server._BRIDGE_PROBE_STRIKES + 2):
        assert json.loads(request_fn("s1", {"action": "next"}))["success"] is False

    assert session[state_key] == "answered"
    bridge.answers = [OK]
    assert json.loads(request_fn("s1", {"action": "next"}))["success"] is True


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_a_new_session_reprobes(bridge, monkeypatch, request_fn, state_key, probe_s, full_s, event):
    """The verdict lives on the session record, so it dies with the session."""
    monkeypatch.setitem(server._sessions, "dead", {})
    monkeypatch.setitem(server._sessions, "fresh", {})

    for _ in range(server._BRIDGE_PROBE_STRIKES):
        request_fn("dead", {"action": "targets"})
    bridge.answers = [OK]

    assert json.loads(request_fn("fresh", {"action": "targets"}))["success"] is True


@pytest.mark.parametrize("request_fn,state_key,probe_s,full_s,event", BRIDGES)
def test_a_session_with_no_record_still_bridges(
    bridge, request_fn, state_key, probe_s, full_s, event
):
    """Detached callers have no session dict to write a verdict onto, so they
    re-probe on every call and can never latch themselves off."""
    for _ in range(server._BRIDGE_PROBE_STRIKES + 1):
        assert json.loads(request_fn("gone", {"action": "targets"}))["success"] is False

    bridge.answers = [OK]
    assert json.loads(request_fn("gone", {"action": "targets"}))["success"] is True
    assert len(bridge.calls) == server._BRIDGE_PROBE_STRIKES + 2
