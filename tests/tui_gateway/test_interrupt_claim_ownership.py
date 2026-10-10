"""A delayed Stop or admission cannot change a replacement turn."""
import types
import pytest
from tui_gateway import server_requests
from tui_gateway import server
from tests.tui_gateway.test_turn_claim_interrupt import _idle_session


@pytest.mark.parametrize("replacement_phase", ["hook", "agent", "delegation", "review", "pending", "approvals"])
def test_interrupt_stops_after_a_callback_replaces_the_claim(monkeypatch, replacement_phase):
    session = _idle_session()
    session.update(running=True, _turn_claim=1)
    actions = []
    phases = ["hook", "agent", "delegation", "review", "pending", "approvals"]
    successor_requests = []

    def phase(which):
        actions.append(which)
        if which != replacement_phase:
            return
        with session["history_lock"]:
            session["running"] = False
            server._claim_session_turn(session)
        session["agent"] = types.SimpleNamespace(
            session_id="new", interrupt=lambda: actions.append("successor-agent"))

    session["agent"].interrupt = lambda: phase("agent")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **k: phase("hook"))
    monkeypatch.setattr("tools.async_delegation.interrupt_for_session", lambda *a, **k: phase("delegation"))
    monkeypatch.setattr("agent.background_review.cancel_background_review_for_live_turn",
                        lambda *a, **k: phase("review"))
    clear_pending = server._clear_pending
    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server_requests, "_write", lambda _frame: None)
    monkeypatch.setattr(server_requests, "_emit", lambda *_a: None)

    def clear_captured_requests(*args, **kwargs):
        phase("pending")
        if replacement_phase == "pending":
            server_requests.send_async("sudo", "probe", {}, lambda _result: None)
            successor_requests.extend(request["id"] for request in server_requests.open_requests("probe"))
        clear_pending(*args, **kwargs)

    monkeypatch.setattr(server, "_clear_pending", clear_captured_requests)
    monkeypatch.setattr(server, "_announce_cancelled_gateway_approvals", lambda *a, **k: phase("approvals"))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a: False)
    try:
        server._interrupt_session_turn("probe", session)
        outcome = {
            "claim": session["_turn_claim"], "actions": actions,
            "pending": [request["id"] for request in server_requests.open_requests("probe")],
        }
    finally:
        server_requests.cancel("probe")
    assert outcome == {
        "claim": 2, "actions": phases[:phases.index(replacement_phase) + 1],
        "pending": successor_requests,
    }


def test_stale_admission_does_not_release_successor_lease(monkeypatch):
    session = _idle_session()
    session.pop('active_session_lease')
    session.update(running=True, _turn_claim=1)
    successor = types.SimpleNamespace(released=False, enabled=True, track_liveness=False)
    successor.release = lambda: setattr(successor, 'released', True)

    def replace_during_lease(*args):
        with session['history_lock']:
            session['running'] = False
            server._claim_session_turn(session)
            session.update(active_session_lease=successor, _closing=True)
        return None

    monkeypatch.setattr(server, '_ensure_active_session_slot', replace_during_lease)
    assert server._admit_prompt_turn('probe', session, 'old', None, None, None, None, turn_claim=1) is None
    assert {'lease': session.get('active_session_lease'), 'released': successor.released} == {
        'lease': successor, 'released': False}
