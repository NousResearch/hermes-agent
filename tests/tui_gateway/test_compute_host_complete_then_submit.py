"""A prompt sent right after an isolated turn's ``message.complete`` must not be lost.

Under ``dashboard.turn_isolation`` the child's ``message.complete`` is relayed before
its ``turn.done`` frame flips ``running`` off in the parent. A client that submits its
next prompt on ``message.complete`` is queued in that window; when the text equals the
finished turn's prompt (``continue``, ``yes``, a scripted retry) the parent's stale
``inflight_turn`` mirror made ``_enqueue_prompt`` drop it as a self-duplicate while the
RPC still answered ``queued``. The inline runner clears the mirror before emitting
``message.complete``; the relay now does the same.
"""

import threading

import pytest

from tui_gateway import server


def _isolated_running_session(sid: str, prompt: str) -> dict:
    session = dict(agent=None, agent_ready=threading.Event(), session_key=sid,
                   history=[], history_version=0, history_lock=threading.Lock(),
                   running=True, transport=server._detached_ws_transport,
                   attached_images=[], cols=80, source="desktop", inflight_turn=None,
                   _compute_host_active=True, _compute_host_turn_id="turn-1")
    server._start_inflight_turn(session, prompt)
    return session


def _complete_frame(sid: str, status: str = "complete") -> dict:
    return {"jsonrpc": "2.0", "method": "event",
            "params": {"type": "message.complete", "session_id": sid,
                       "payload": {"text": "done", "status": status}}}


@pytest.fixture
def relay(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(server, "write_json", lambda msg: sent.append(msg) or True)
    return sent


def test_same_text_after_isolated_complete_is_queued_not_dropped(monkeypatch):
    session = _isolated_running_session("s1", "go")
    monkeypatch.setattr(server, "_sessions", {"s1": session})
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    replies, dispatched = [], []

    def receive_complete(message):
        assert message["params"]["type"] == "message.complete"
        assert session["running"] is True  # turn.done has not reached the parent.
        replies.append(server._handle_busy_submit(
            "next", "s1", session, "go", server._detached_ws_transport, queued=True))
        return True

    def dispatch(rid, sid, owning_session, text, **kwargs):
        dispatched.append((rid, sid, owning_session is session, text))
        return {"result": {"status": "streaming"}}

    monkeypatch.setattr(server, "write_json", receive_complete)
    monkeypatch.setattr(server, "_submit_prompt_to_compute_host", dispatch)
    monkeypatch.setattr(server, "_compute_host_session_info", lambda session: {})
    assert server._relay_compute_host_rpc(_complete_frame("s1")) is True
    assert replies[0]["result"]["status"] == "queued"
    assert session.get("queued_prompt") is not None
    assert session["queued_prompt"]["text"] == "go"
    server._on_compute_host_turn_done("next", "s1", session, {
        "type": "turn.done", "session_info_emitted": True})
    assert dispatched == [("next", "s1", True, "go")]
    assert session.get("queued_prompt") is None


@pytest.mark.parametrize("isolated,status", [(True, "error"), (False, "complete")])
def test_unsettled_complete_keeps_the_retained_inflight_turn(monkeypatch, relay, isolated, status):
    session = _isolated_running_session("s2", "go")
    if not isolated:
        session.pop("_compute_host_turn_id")
    monkeypatch.setattr(server, "_sessions", {"s2": session})
    server._relay_compute_host_rpc(_complete_frame("s2", status=status))
    assert server._ac_inflight_original(session) == "go"
