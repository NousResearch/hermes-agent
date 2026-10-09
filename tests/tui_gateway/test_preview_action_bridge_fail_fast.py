"""A desktop client that cannot answer the ``preview.act`` server request must not
cost a full bridge timeout per drive_preview / annotate_preview call.

The renderer's handler ships in the desktop bundle; the tools are offered by the
backend. An app build older than the tools has no handler for the request, so
nothing ever answers and the agent blocks for the whole 45s deadline — once per
action the model tries, stacking per turn exactly like the tour timeouts
(#94272). Same ladder as ``tui_gateway.server._tour_request``, plus a
cooldown-gated reprobe: unlike tour support, a preview pane can OPEN later in
the session, so an unanswered probe condemns the bridge for a cooldown window
only, with a single in-flight reprobe per session.

A window-owned request nobody can answer settles at once once every attached
window declines (#119333, main e33f1a0faf): that refusal arrives as a
non-empty answer, so the probe ladder must let it through and not mask it.
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
    """Record every ``_ask`` (server request) call and serve canned answers."""
    calls = []

    def fake_block(event, sid, payload, timeout=None, **_kw):
        answer = fake_block.answers.pop(0) if fake_block.answers else ""
        calls.append({"event": event, "sid": sid, "payload": payload, "timeout": timeout})
        return answer

    fake_block.answers = []
    fake_block.calls = calls
    monkeypatch.setattr(server, "_ask", fake_block)
    return fake_block


@pytest.fixture(autouse=True)
def no_leftover_reprobe():
    yield
    server._preview_action_reprobe.clear()


def test_first_action_is_probed_on_a_short_deadline(session, bridge):
    bridge.answers = [json.dumps({"success": True})]
    server._preview_action_request("s1", {"action": "elements"})

    assert bridge.calls[0]["event"] == "preview.act"
    assert bridge.calls[0]["timeout"] == server._PREVIEW_ACTION_PROBE_TIMEOUT_S
    assert server._PREVIEW_ACTION_PROBE_TIMEOUT_S < server._PREVIEW_ACTION_TIMEOUT_S


def test_a_client_that_answers_gets_the_full_deadline_back(session, bridge):
    """The generous deadline exists for a slow page; only an unproven client
    is held to the probe."""
    bridge.answers = [json.dumps({"success": True}), json.dumps({"success": True})]
    server._preview_action_request("s1", {"action": "elements"})
    server._preview_action_request("s1", {"action": "click", "ref": "btn-sign-in"})

    assert bridge.calls[1]["timeout"] == server._PREVIEW_ACTION_TIMEOUT_S


def test_unanswered_probe_returns_an_actionable_error(session, bridge):
    """The regression (#94272): every action against an absent renderer burned
    the full 45s and then reported an error that blamed a closed tab."""
    result = json.loads(server._preview_action_request("s1", {"action": "elements"}))

    assert result["success"] is False
    assert "Update the Hermes Desktop app" in result["error"]


def test_repeat_actions_short_circuit_instead_of_stalling_again(session, bridge):
    server._preview_action_request("s1", {"action": "elements"})
    for action in ("click", "type", "scroll", "press"):
        assert json.loads(
            server._preview_action_request("s1", {"action": action, "ref": "btn"}))["success"] is False

    assert len(bridge.calls) == 1


def test_cooldown_expiry_reprobes_once(session, bridge, monkeypatch):
    """After the cooldown the bridge is retried — a pane may have opened (or
    the app launched) since the probe failed; the retry keeps the probe
    deadline until something answers."""
    monkeypatch.setattr(server, "_PREVIEW_ACTION_REPROBE_COOLDOWN_S", 0.0)
    server._preview_action_request("s1", {"action": "elements"})
    assert len(bridge.calls) == 1

    bridge.answers = [json.dumps({"success": True})]
    assert json.loads(
        server._preview_action_request("s1", {"action": "elements"}))["success"] is True
    assert bridge.calls[1]["timeout"] == server._PREVIEW_ACTION_PROBE_TIMEOUT_S

    # Answered once: the full deadline is back.
    server._preview_action_request("s1", {"action": "click", "ref": "btn"})
    assert bridge.calls[2]["timeout"] == server._PREVIEW_ACTION_TIMEOUT_S


def test_within_cooldown_a_second_caller_fails_fast(session, bridge):
    """Only one in-flight reprobe per session: a concurrent sibling gets the
    refusal instead of stacking another 10s wait."""
    server._preview_action_request("s1", {"action": "elements"})

    # Simulate the cooldown expiring while the owner's reprobe is in flight:
    # the state says "unanswered, past retry_at", and the token is held.
    session["preview_action_bridge_retry_at"] = 0.0
    server._preview_action_reprobe["s1"] = object()
    assert len(bridge.calls) == 1
    assert json.loads(
        server._preview_action_request("s1", {"action": "scroll"}))["success"] is False
    assert len(bridge.calls) == 1


def test_a_new_session_reprobes(bridge, monkeypatch):
    """The verdict lives on the session record, so it dies with the session."""
    monkeypatch.setitem(server._sessions, "dead", {})
    monkeypatch.setitem(server._sessions, "fresh", {})

    server._preview_action_request("dead", {"action": "elements"})
    bridge.answers = [json.dumps({"success": True})]

    assert json.loads(
        server._preview_action_request("fresh", {"action": "elements"}))["success"] is True


def test_a_session_with_no_record_still_bridges(bridge):
    """Detached callers have no session dict; they keep the plain bridge."""
    bridge.answers = [json.dumps({"success": True})]
    assert json.loads(
        server._preview_action_request("gone", {"action": "elements"}))["success"] is True


def test_a_window_decline_answer_passes_through(session, bridge):
    """The fast "no window is showing this chat" refusal (#119333) settles as
    a non-empty answer: it must reach the tool verbatim, not be re-labelled
    bridge-unavailable."""
    decline = json.dumps({"success": False, "error": "No Hermes Desktop window is showing this chat."})
    bridge.answers = [decline]

    assert server._preview_action_request("s1", {"action": "elements"}) == decline


def test_a_proven_client_is_not_condemned_by_one_slow_action(session, bridge):
    """A transient miss on a live renderer must not disable preview actions
    (unlike tour: the pane may just be busy); it re-arms the cooldown, and the
    next call after the cooldown gets a real probe again."""
    bridge.answers = [json.dumps({"success": True})]
    server._preview_action_request("s1", {"action": "elements"})

    # Proven client goes quiet (cancelled / one-off miss): stays full-deadline
    # and is NOT flipped to unanswered.
    assert json.loads(
        server._preview_action_request("s1", {"action": "click", "ref": "btn"}))["success"] is False
    assert session["preview_action_bridge"] == "answered"
    server._preview_action_request("s1", {"action": "scroll"})
    assert bridge.calls[2]["timeout"] == server._PREVIEW_ACTION_TIMEOUT_S
