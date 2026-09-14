"""A desktop client that cannot answer ``tour.request`` must not cost a full
bridge timeout per call.

The renderer's handler ships in the desktop bundle; the tool is offered by the
backend. An app build older than the tour tool has no branch for the event, so
nothing ever calls ``tour.respond`` and the agent blocks for the whole deadline
— once per action the model tries. See tui_gateway.server._tour_request.
"""

import json

import pytest

import tui_gateway.server as server


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


def test_first_action_is_probed_on_a_short_deadline(session, bridge):
    bridge.answers = [json.dumps({"success": True})]
    server._tour_request("s1", {"action": "targets"})

    assert bridge.calls[0]["event"] == "tour.request"
    assert bridge.calls[0]["timeout"] == server._TOUR_PROBE_TIMEOUT_S
    assert server._TOUR_PROBE_TIMEOUT_S < server._TOUR_TIMEOUT_S


def test_a_client_that_answers_gets_the_full_deadline_back(session, bridge):
    """The generous deadline exists for a preview tour injecting into a live
    page; only an unproven client is held to the probe."""
    bridge.answers = [json.dumps({"success": True}), json.dumps({"success": True})]
    server._tour_request("s1", {"action": "targets"})
    server._tour_request("s1", {"action": "show", "selector": "#composer"})

    assert bridge.calls[1]["timeout"] == server._TOUR_TIMEOUT_S


def test_unanswered_probe_explains_the_real_problem(session, bridge):
    result = json.loads(server._tour_request("s1", {"action": "targets"}))

    assert result["success"] is False
    assert "desktop" in result["error"].lower()
    assert "update" in result["error"].lower()


def test_later_actions_short_circuit_instead_of_stalling_again(session, bridge):
    """The regression: a "give me a tour" turn stacks one timeout per action."""
    server._tour_request("s1", {"action": "targets"})
    for action in ("show", "start", "next", "stop"):
        assert json.loads(server._tour_request("s1", {"action": action}))["success"] is False

    assert len(bridge.calls) == 1


def test_a_proven_client_is_not_condemned_by_one_slow_action(session, bridge):
    """A transient miss on a live renderer must not disable the tour."""
    bridge.answers = [json.dumps({"success": True})]
    server._tour_request("s1", {"action": "targets"})
    server._tour_request("s1", {"action": "show", "selector": "#composer"})

    bridge.answers = [json.dumps({"success": True})]
    assert json.loads(server._tour_request("s1", {"action": "next"}))["success"] is True
    assert len(bridge.calls) == 3


def test_a_new_session_reprobes(bridge, monkeypatch):
    """The verdict lives on the session record, so it dies with the session."""
    monkeypatch.setitem(server._sessions, "dead", {})
    monkeypatch.setitem(server._sessions, "fresh", {})

    server._tour_request("dead", {"action": "targets"})
    bridge.answers = [json.dumps({"success": True})]

    assert json.loads(server._tour_request("fresh", {"action": "targets"}))["success"] is True


def test_a_session_with_no_record_still_bridges(bridge):
    """Detached callers have no session dict; they keep the plain bridge."""
    bridge.answers = [json.dumps({"success": True})]
    assert json.loads(server._tour_request("gone", {"action": "targets"}))["success"] is True
