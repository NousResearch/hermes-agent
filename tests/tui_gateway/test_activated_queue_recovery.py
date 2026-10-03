"""Activation recovers an idle accepted queue without crossing an orphan claim."""

import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class Peer:
    def __init__(self, closed=False):
        self._closed = closed

    def write(self, _frame):
        return not self._closed


class ResumeLockWitness:
    def __init__(self, before_drain=None):
        self.lock = threading.Lock()
        self.held = False
        self.entries = 0
        self.before_drain = before_drain

    def __enter__(self):
        assert not self.held, "activation held the resume lock into its drain"
        self.lock.acquire()
        self.held = True
        self.entries += 1
        if self.entries == 2 and self.before_drain:
            self.before_drain()
        return self

    def __exit__(self, *_exc):
        self.held = False
        self.lock.release()


def preserved_session(monkeypatch, tmp_path):
    sid = "queue-recovery"
    dead = Peer(closed=True)
    envelope = {
        "text": "preserved follow-up",
        "image_paths": ["/queue/original.png"],
        "turn_author": {"name": "Original sender", "user_id": "person-7"},
        "transport": dead,
    }
    session = {
        "transport": server._detached_ws_transport,
        "running": False,
        "history_lock": threading.Lock(),
        "history": [],
        "session_key": "stored-queue",
        "cwd": str(tmp_path),
        "profile_home": str(tmp_path),
        "agent": SimpleNamespace(model="test"),
        "queued_prompt": envelope,
    }
    monkeypatch.setattr(server, "_sessions", {sid: session})
    # A preserved detached session already owns a reap timer. The external
    # timer handle is inert; rebind/cancel and admission remain production code.
    monkeypatch.setattr(server, "_pending_ws_reaps", {sid: SimpleNamespace(cancel=lambda: None)})
    return sid, session, envelope, dead


def activate(sid, peer):
    token = bind_transport(peer)
    try:
        return server.handle_request({
            "id": "activate", "method": "session.activate",
            "params": {"session_id": sid, "omit_messages": True},
        })
    finally:
        reset_transport(token)


@pytest.mark.parametrize("compute_host", [False, True])
def test_activation_dispatches_preserved_idle_envelope_once_outside_resume_lock(monkeypatch, tmp_path, compute_host):
    sid, session, envelope, dead = preserved_session(monkeypatch, tmp_path)
    live = Peer()
    witness = ResumeLockWitness()
    monkeypatch.setattr(server, "_session_resume_lock", witness)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: compute_host)
    dispatched = []

    def execute(_rid, target, owner, text, **kwargs):
        assert not witness.held, "external dispatch held the process-wide resume lock"
        assert witness.entries == 2, "queue claim did not reacquire the orphan-reap lock"
        assert owner is session and target == sid
        assert owner["transport"] is live
        assert server._transport_is_dead(dead)
        dispatched.append((text, kwargs))
        return {"result": {"status": "started"}}

    boundary = "_submit_prompt_to_compute_host" if compute_host else "_run_prompt_submit"
    monkeypatch.setattr(server, boundary, execute)
    first = activate(sid, live)
    assert "result" in first, first
    assert first["result"]["running"] is True
    assert session["queued_prompt"] is None
    assert len(dispatched) == 1
    text, kwargs = dispatched[0]
    assert text == "preserved follow-up"
    assert kwargs["image_paths"] == ["/queue/original.png"]
    if not compute_host:
        assert kwargs["turn_author"] == {"name": "Original sender", "user_id": "person-7"}
    assert envelope["transport"] is dead  # no dead pin was revived or rewritten
    second = activate(sid, live)
    assert "result" in second, second
    assert len(dispatched) == 1


@pytest.mark.parametrize("refusal", ["busy", "dead", "stale", "closing", "claimed"])
def test_activation_drain_refuses_lost_admission_and_keeps_envelope(monkeypatch, tmp_path, refusal):
    sid, session, envelope, _dead = preserved_session(monkeypatch, tmp_path)
    live = Peer()
    raced = []

    def orphan_wins():
        raced.append(refusal)
        if refusal == "busy":
            session["running"] = True
        elif refusal == "dead":
            live._closed = True
        elif refusal == "stale":
            server._sessions[sid] = dict(session)
        elif refusal == "closing":
            session["_closing"] = True
        else:
            session["_client_gone_interrupt_requested"] = True

    witness = ResumeLockWitness(orphan_wins)
    monkeypatch.setattr(server, "_session_resume_lock", witness)
    dispatched = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: dispatched.append((a, kw)))
    response = activate(sid, live)
    assert "result" in response, response
    assert raced == [refusal], "activation never attempted the guarded post-rebind drain"
    assert session["queued_prompt"] is envelope
    assert not dispatched
