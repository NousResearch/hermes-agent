"""Compression in flight demotes busy-mode steer/interrupt to queue (#61042).

A follow-up delivered to the provider mid-compression aborts the compression
(``explicit_interrupt``) — the user's message kills the turn that would have
answered it. The channel-side busy path already demoted interrupt→queue for
this reason (gateway/run_busy.py, #56391); these tests pin the same contract
for the local RPC busy path (prompt.submit / session.steer / session.redirect)
so TUI, Desktop and classic chat share the Discord-gateway behavior: the
follow-up queues and drains when compression finishes.
"""

from __future__ import annotations

import threading
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "transport": None,
        "attached_images": [],
        **extra,
    }


# ── _session_compression_in_flight ────────────────────────────────────────

def _patch_db(monkeypatch, holder):
    class _Db:
        def get_compression_lock_holder(self, sid):
            return holder

    import contextlib

    @contextlib.contextmanager
    def _fake_db(session):
        yield _Db()

    monkeypatch.setattr(server, "_session_db", _fake_db)


def test_compression_in_flight_true_when_lock_held(monkeypatch):
    _patch_db(monkeypatch, holder="compressor-1")
    session = _session()
    session["agent"].session_id = "session-key"

    assert server._session_compression_in_flight(session) is True


def test_compression_in_flight_false_when_lock_free(monkeypatch):
    _patch_db(monkeypatch, holder=None)

    assert server._session_compression_in_flight(_session()) is False


def test_compression_in_flight_false_when_db_errors(monkeypatch):
    import contextlib

    @contextlib.contextmanager
    def _boom(session):
        raise RuntimeError("db gone")
        yield  # pragma: no cover

    monkeypatch.setattr(server, "_session_db", _boom)

    # A failed check must fail OPEN (no demotion): compression protection is
    # best-effort here, unlike ownership which fails closed.
    assert server._session_compression_in_flight(_session()) is False


# ── _handle_busy_submit demotion ──────────────────────────────────────────

def test_busy_interrupt_mode_queues_while_compressing(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda session: True)
    seen = []

    def _boom(text):
        seen.append(text)
        return True

    agent = types.SimpleNamespace(_supports_active_turn_redirect=True, redirect=_boom)
    session = _session(agent=agent)

    resp = server._handle_busy_submit("r1", "sid", session, "follow-up", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert seen == []  # never redirected — the compression survives
    assert session["queued_prompt"]["text"] == "follow-up"


def test_busy_steer_mode_queues_while_compressing(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda session: True)
    seen = []
    agent = types.SimpleNamespace(steer=lambda text: seen.append(text) or True)
    session = _session(agent=agent)

    resp = server._handle_busy_submit("r1", "sid", session, "follow-up", "ws-1")

    assert resp["result"]["status"] == "queued"
    assert seen == []
    assert session["queued_prompt"]["text"] == "follow-up"


def test_busy_interrupt_mode_still_redirects_without_compression(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda session: False)
    seen = []
    agent = types.SimpleNamespace(
        _supports_active_turn_redirect=True,
        redirect=lambda text: seen.append(text) or True,
    )
    session = _session(agent=agent)

    resp = server._handle_busy_submit("r1", "sid", session, "correction", "ws-1")

    assert resp["result"]["status"] == "redirected"
    assert seen == ["correction"]


def test_busy_queued_drain_forces_queue_regardless(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda session: False)
    session = _session()
    session["queued_prompt"] = {"text": "earlier", "transport": "ws-1"}

    resp = server._handle_busy_submit("r1", "sid", session, "next", "ws-1", queued=True)

    assert resp["result"]["status"] == "queued"


# ── session.steer / session.redirect demotion ─────────────────────────────

def test_session_steer_queues_while_compressing(monkeypatch):
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda session: True)
    seen = []
    agent = types.SimpleNamespace(steer=lambda text: seen.append(text) or True)
    session = _session(agent=agent)
    monkeypatch.setitem(server._sessions, "sid", session)
    try:
        resp = server.handle_request(
            {"id": "r1", "method": "session.steer", "params": {"session_id": "sid", "text": "follow-up"}})
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["status"] == "queued"
    assert seen == []
    assert session["queued_prompt"]["text"] == "follow-up"


def test_session_redirect_queues_while_compressing(monkeypatch):
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda session: True)
    seen = []
    agent = types.SimpleNamespace(_supports_active_turn_redirect=True, redirect=lambda t: seen.append(t) or True)
    session = _session(agent=agent)
    monkeypatch.setitem(server._sessions, "sid", session)
    try:
        resp = server.handle_request(
            {"id": "r1", "method": "session.redirect", "params": {"session_id": "sid", "text": "follow-up"}})
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"]["status"] == "queued"
    assert seen == []


# ── manual compaction holds the session busy (#133504) ────────────────────

def test_submit_during_manual_compress_is_queued_and_reply_persists(monkeypatch):
    """A prompt sent while session.compress runs its LLM summary used to be admitted on the idle
    session, snapshot history_version N, and lose its reply when the compaction committed N+1."""
    import agent.conversation_compression_manual as ccm

    gate, entered = threading.Event(), threading.Event()

    def fake_compress_now(agent, msgs, request, **kw):
        entered.set()
        gate.wait(5)
        return types.SimpleNamespace(status="compressed", removed=4,
                                     after_messages=[{"role": "user", "content": "[summary]"}, *msgs[-2:]])

    def fake_turn(rid, sid, session, text, **kw):  # the drained turn: snapshot, reply, commit
        with session["history_lock"]:
            hist, ver = list(session["history"]), int(session["history_version"])
        result = {"messages": [*hist, {"role": "user", "content": text}, {"role": "assistant", "content": "REPLY"}]}
        server._commit_turn_history(session, result, hist, ver)
        session["running"] = False

    monkeypatch.setattr(ccm, "compress_now", fake_compress_now)
    monkeypatch.setattr(server, "_run_prompt_submit", fake_turn)
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    for name in ("_status_update", "_emit", "_sync_session_key_after_compress", "_persist_queued_user_row",
                 "_replace_queued_user_row_for_turn"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _s: False)
    monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
    agent = types.SimpleNamespace(
        session_id="session-key", _cached_system_prompt="", tools=None, context_compressor=None,
        interrupt=lambda *a, **k: (_ for _ in ()).throw(AssertionError("compaction must not be interrupted")))
    history = [{"role": r, "content": f"m{i}"} for i, r in enumerate(["user", "assistant"] * 3)]
    session = _session(agent=agent, running=False, history=history)
    monkeypatch.setattr(server, "_sess", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))

    rpc = threading.Thread(target=server._methods["session.compress"], args=("r1", {"session_id": "sid"}))
    rpc.start()
    assert entered.wait(5)
    resp = server._handle_busy_submit("r2", "sid", session, "question", "ws-1")
    gate.set()
    rpc.join(5)

    assert resp["result"]["status"] == "queued"
    assert session["history"][-1]["content"] == "REPLY"
    assert session["history"][0]["content"] == "[summary]"
    assert session["running"] is False
