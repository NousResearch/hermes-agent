"""Tests for tui_gateway JSON-RPC protocol plumbing."""

import io
import json
import os
import subprocess
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


@pytest.fixture()
def server():
    # The sys.modules mocks only need to cover the *initial* import — once
    # tui_gateway.server is cached, they are inert. Keeping them active for
    # the whole test poisons any module first imported inside a test body:
    # e.g. hermes_cli.active_sessions would bind the mocked get_hermes_home
    # (a fixed shared path) forever, leaking active-session registry entries
    # across every later test in the process. Scope the patch to the import.
    #
    # Import server_requests (pure stdlib) BEFORE the window: the patch drops every module first imported
    # inside it, so otherwise the module server.py binds its sinks on (write/emit/answerable) would vanish
    # from sys.modules and a test's own ``from tui_gateway import server_requests`` would get a fresh,
    # unbound copy whose default sinks drop frames and treat every client as answerable.
    import tui_gateway.server_requests  # noqa: F401
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")

    # Snapshot the RPC registry: several tests below stub handlers
    # ("slash.exec", "fast.ping", ...) directly in the module-level dict,
    # which is shared with every other test file in the process.
    methods = dict(mod._methods)
    real_stdout = mod._real_stdout
    yield mod
    # Reset module-level state without re-importing. importlib.reload
    # would re-register the module's atexit hooks (ThreadPoolExecutor
    # shutdown, _shutdown_sessions); the duplicates race the stderr
    # buffer at interpreter shutdown and surface as Fatal Python error:
    # _enter_buffered_busy. Restoring the dicts in place gives the next
    # test a clean slate.
    mod._methods.clear()
    mod._methods.update(methods)
    mod._real_stdout = real_stdout
    for sid in list(mod._sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")
    from tui_gateway import server_requests
    server_requests.reset_for_tests()
    mod._live_transports.clear()


def test_shared_fixture_cleanup_uses_full_session_teardown(server, monkeypatch):
    """The cross-file autouse cleanup must close every retained resource."""
    from tests import conftest

    closed = {"worker": 0, "agent": 0, "lease": 0}

    class _Closable:
        def __init__(self, key):
            self.key = key

        def close(self):
            closed[self.key] += 1

    class _Lease:
        def release(self):
            closed["lease"] += 1

    monkeypatch.setattr(server, "_get_db", lambda: None)
    server._sessions["leaked"] = {
        "session_key": "leaked",
        "agent": _Closable("agent"),
        "slash_worker": _Closable("worker"),
        "active_session_lease": _Lease(),
        "history": [],
    }

    conftest._teardown_tui_server_sessions(server)

    assert server._sessions == {}
    assert closed == {"worker": 1, "agent": 1, "lease": 1}


@pytest.fixture()
def capture(server):
    """Redirect server's real stdout to a StringIO and return (server, buf)."""
    buf = io.StringIO()
    server._real_stdout = buf
    return server, buf


# ── JSON-RPC envelope ────────────────────────────────────────────────


def test_unknown_method(server):
    resp = server.handle_request({"id": "1", "method": "bogus"})
    assert resp["error"]["code"] == -32601


def test_ok_envelope(server):
    assert server._ok("r1", {"x": 1}) == {
        "jsonrpc": "2.0", "id": "r1", "result": {"x": 1},
    }


def test_err_envelope(server):
    assert server._err("r2", 4001, "nope") == {
        "jsonrpc": "2.0", "id": "r2", "error": {"code": 4001, "message": "nope"},
    }


@pytest.mark.parametrize("kind", ["legacy", "hard-only", "dynamic-getattr"])
def test_session_interrupt_uses_explicit_stop_compatibility(server, monkeypatch, kind):
    calls = []

    class _Legacy:
        def interrupt(self):
            calls.append("legacy")

    class _HardOnly:
        def hard_interrupt(self):
            calls.append("hard")

    class _Dynamic:
        def interrupt(self):
            calls.append("legacy")

        def __getattr__(self, name):
            if name == "hard_interrupt":
                return lambda: calls.append("fabricated-hard")
            raise AttributeError(name)

    agent = {
        "legacy": _Legacy(),
        "hard-only": _HardOnly(),
        "dynamic-getattr": _Dynamic(),
    }[kind]
    session = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "running": True,
        "queued_prompt": "later",
        "session_key": "session-key",
        "_run_thread": None,
    }
    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(server, "_sess_nowait", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: False)
    monkeypatch.setattr(server, "_clear_pending", lambda _sid: None)
    response = server._methods["session.interrupt"](
        "stop", {"session_id": "ui-session"}
    )

    assert response["result"]["status"] == "interrupted"
    assert calls == ["hard" if kind == "hard-only" else "legacy"]


# ── write_json ────────────────────────────────────────────────


def test_write_json(capture):
    server, buf = capture
    assert server.write_json({"test": True})
    assert json.loads(buf.getvalue()) == {"test": True}


def test_live_session_payload_replays_pending_approval(server, monkeypatch):
    """A reattached client receives the approval that was emitted while detached."""
    from tools import approval
    from tools import approval_gateway_wait

    session = {
        "agent": types.SimpleNamespace(),
        "cols": 80,
        "created_at": 1.0,
        "history": [],
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "stored-session",
    }
    first = {
        "choices": ["once", "deny"],
        "command": "rm -rf /tmp/example",
        "description": "recursive delete",
    }
    second = {"command": "rm -rf /tmp/later", "description": "later"}
    saved_queue = approval._gateway_queues.pop("stored-session", None)
    approval._gateway_queues["stored-session"] = [
        approval_gateway_wait._ApprovalEntry(first),
        approval_gateway_wait._ApprovalEntry(second),
    ]
    monkeypatch.setattr(server, "_approval_request_payload", lambda data: dict(data or {}))

    try:
        payload = server._live_session_payload("runtime-session", session)
    finally:
        approval._gateway_queues.pop("stored-session", None)
        if saved_queue is not None:
            approval._gateway_queues["stored-session"] = saved_queue

    assert payload["pending_approval"] is not first
    replayed = payload["pending_approval"]
    # request_id is injected by _ApprovalEntry so reconnecting clients can
    # correlate their approval.respond with the exact queued request.
    assert replayed.pop("request_id")
    assert replayed == first


def test_live_session_payload_replays_open_requests(server):
    """A reattached client receives the server→client request still blocking the session (a clarify sent
    while detached), scoped to the owning runtime session only, as a snapshot not a live reference."""
    from tui_gateway import server_requests
    session = {
        "agent": types.SimpleNamespace(), "cols": 80, "created_at": 1.0, "history": [],
        "history_lock": threading.Lock(), "running": True, "session_key": "stored-session",
    }
    req = server_requests.ServerRequest("runtime-session", "clarify", {"question": "Which?", "choices": ["a", "b"]})
    server_requests._open[req.id] = req
    try:
        payload = server._live_session_payload("runtime-session", session)
        other = server._live_session_payload("other-session", session)
    finally:
        server_requests._open.pop(req.id, None)

    assert payload["open_requests"] == [{"id": req.id, "method": "clarify",
                                         "params": {"session_id": "runtime-session", "question": "Which?", "choices": ["a", "b"]}}]
    assert payload["open_requests"][0]["params"] is not req.params
    assert "open_requests" not in other


def test_disable_flush_env_var_actually_wires_to_module_constant(monkeypatch):
    """End-to-end: setting `HERMES_TUI_GATEWAY_NO_FLUSH=1` and importing
    `tui_gateway.transport` fresh actually flips `_DISABLE_FLUSH` true.

    Reloads only the transport module — server.py is untouched so its
    atexit hooks/worker pool stay intact."""
    import importlib

    monkeypatch.setenv("HERMES_TUI_GATEWAY_NO_FLUSH", "1")
    transport_mod = importlib.reload(importlib.import_module("tui_gateway.transport"))

    try:
        assert transport_mod._DISABLE_FLUSH is True
    finally:
        # Restore the env-disabled state so other tests see the default.
        monkeypatch.delenv("HERMES_TUI_GATEWAY_NO_FLUSH", raising=False)
        importlib.reload(transport_mod)


# ── _emit ────────────────────────────────────────────────────────────


def test_emit_with_payload(capture):
    server, buf = capture
    server._emit("test.event", "s1", {"key": "val"})
    msg = json.loads(buf.getvalue())

    assert msg["method"] == "event"
    assert msg["params"]["type"] == "test.event"
    assert msg["params"]["session_id"] == "s1"
    assert msg["params"]["payload"]["key"] == "val"


# ── Server→client requests (tui_gateway/server_requests.py) ─────────


def _frames(buf):
    return [json.loads(line) for line in buf.getvalue().splitlines()]


def _wait_open(server_requests, buf=None, timeout=2.0):
    """The open request once its frame has been written (registration precedes the write)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with server_requests._lock:
            req = next(iter(server_requests._open.values()), None)
        if req is not None and (buf is None or req.id in buf.getvalue()):
            return req
        time.sleep(0.01)
    raise AssertionError("server request never registered")


def test_server_request_round_trip_uses_response_frame(capture):
    """The backend sends a real JSON-RPC request (string id, method, params.session_id) and the client's
    response frame — not a method call — resolves it; the response frame itself gets no reply."""
    from tui_gateway import server_requests
    server, buf = capture
    box = {}
    thread = threading.Thread(target=lambda: box.__setitem__("r", server._ask("sudo", "s1", {}, timeout=5)), daemon=True)
    thread.start()
    req = _wait_open(server_requests, buf)
    frame = _frames(buf)[-1]
    assert frame == {"jsonrpc": "2.0", "id": req.id, "method": "sudo", "params": {"session_id": "s1"}}
    assert req.id.startswith("srq-")

    assert server.dispatch({"jsonrpc": "2.0", "id": req.id, "result": {"value": "hunter2"}}) is None
    thread.join(timeout=5)
    assert box["r"] == "hunter2"
    with server_requests._lock:
        assert not server_requests._open


@pytest.mark.parametrize("method, qids, settle, expected", [
    ("sudo", None,
     lambda sr, req: sr.resolve_response({"id": req.id, "result": {"value": "yes"}}) is True,
     {"value": "yes"}),
    # Batch clarify's lock-based resolution follows the same first-settlement rule.
    ("clarify", ["q1"], lambda sr, req: sr.lock_answer(req.id, "q1", "yes") == [], {"answers": {"q1": "yes"}}),
])
def test_settlement_wins_over_a_later_cancel(capture, method, qids, settle, expected):
    """A response and cancellation may race; the first settlement owns the result."""
    from tui_gateway import server_requests

    req = server_requests.ServerRequest("s1", method, {}, qids=qids)
    with server_requests._lock:
        server_requests._open[req.id] = req

    assert settle(server_requests, req)
    assert server_requests.cancel("s1") == 0
    assert req.answered is True
    assert req.result == expected
    assert req.event.is_set()


def test_send_returns_an_answer_committed_after_the_deadline_expired(capture, monkeypatch):
    """An answer accepted by resolve_response is never reported as a timeout (#112548): the
    response frame can land after event.wait() gave up and before send() withdraws the request,
    and the renderer must not get a bogus request.cancel for a card the user just answered."""
    from tui_gateway import server_requests

    cancels: list[dict] = []
    monkeypatch.setattr(server_requests, "_emit", lambda event, sid, payload: cancels.append(payload))

    real_wait = server_requests.threading.Event.wait

    def answered_during_the_gap(event, timeout=None):
        # Deadline expires, then the response frame lands before send() re-enters the lock.
        expired = real_wait(event, timeout)
        rid = next(iter(server_requests._open))
        assert server_requests.resolve_response({"id": rid, "result": {"value": "yes"}})
        return expired

    monkeypatch.setattr(server_requests.threading.Event, "wait", answered_during_the_gap)

    assert server_requests.send("sudo", "s1", {}, timeout=0.001) == {"value": "yes"}
    assert cancels == []
    assert not server_requests._open


def _silent_ws():
    """A WebSocket client build that predates server→client requests: receives the frame, never answers."""
    from tui_gateway.ws import WSTransport

    class _SilentWS(WSTransport):
        def __init__(self):
            self._ws, self._loop, self._peer, self._auth_identity = object(), None, "test", None
            self.frames: list[dict] = []

        def write(self, obj):
            self.frames.append(obj)
            return True

        def close(self):
            pass

    return _SilentWS()


def _ws_session(server, sid, peer):
    server._sessions[sid] = {"session_key": sid, "transport": peer, "history": [], "history_lock": threading.Lock(),
                             "agent_ready": None}


def test_server_request_fails_fast_for_a_ws_client_that_never_advertised(server):
    """A WebSocket client that never sent ``client.capabilities`` cannot answer, so send() returns the
    error-response shape (None) at once instead of stalling the agent for the deadline (#112548).
    The frame is never written; nothing is left open for a reconnect replay."""
    from tui_gateway import server_requests

    peer = _silent_ws()
    _ws_session(server, "ws-old", peer)
    t0 = time.monotonic()
    assert server_requests.send("clarify", "ws-old", {"question": "q?", "choices": None, "multi_select": False},
                                timeout=5) is None
    assert time.monotonic() - t0 < 1
    assert peer.frames == []
    assert server_requests.open_requests("ws-old") == []
    settled = []
    server_requests.send_async("approval", "ws-old", {"request_id": "r1", "command": "rm", "description": "",
                                                      "pattern_key": "", "pattern_keys": []}, settled.append)
    assert settled == [None]


def test_server_request_waits_for_a_ws_client_that_advertised(server):
    """``client.capabilities {server_requests: true}`` on the connection marks it answerable: the frame is
    written and the wait is a real one (here: answered by the response frame)."""
    from tui_gateway import server_requests
    from tui_gateway.transport import bind_transport, reset_transport

    peer = _silent_ws()
    _ws_session(server, "ws-new", peer)
    token = bind_transport(peer)
    try:
        response = server.handle_request({"id": 1, "method": "client.capabilities", "params": {"server_requests": True}})
    finally:
        reset_transport(token)
    assert "clarify" in response["result"]["server_requests"]

    box = {}
    thread = threading.Thread(target=lambda: box.setdefault("r", server_requests.send("sudo", "ws-new", {}, timeout=5)),
                              daemon=True)
    thread.start()
    req = _wait_open(server_requests)
    assert peer.frames[-1]["id"] == req.id
    assert server.dispatch({"jsonrpc": "2.0", "id": req.id, "result": {"value": "yes"}}) is None
    thread.join(timeout=5)
    assert box["r"] == {"value": "yes"}
    # Disconnect forgets the advertisement; the next connection must advertise again.
    server.unregister_live_transport(peer)
    assert server_requests.answers_requests(peer) is False


def test_server_request_error_response_fails_fast(server):
    """A client that advertised but has no handler for the method answers -32601: that error frame settles
    send() to None at once instead of the agent waiting for the deadline."""
    from tui_gateway import server_requests

    box = {}
    thread = threading.Thread(target=lambda: box.setdefault("result", server_requests.send("sudo", "s1", {}, timeout=5)),
                              daemon=True)
    thread.start()
    req = _wait_open(server_requests)
    t0 = time.monotonic()
    assert server_requests.resolve_response({"id": req.id, "error": {"code": -32601}})
    thread.join(timeout=1)
    assert not thread.is_alive() and time.monotonic() - t0 < 1
    assert box["result"] is None


@pytest.mark.parametrize("method", ["secret", "sudo", "terminal.read", "tour"])
def test_server_request_timeout_emits_one_request_cancel(capture, method):
    from tui_gateway import server_requests
    server, buf = capture
    assert server_requests.send(method, "s1", {}, timeout=0) is None
    request, cancel = _frames(buf)
    assert request["method"] == method
    assert cancel["params"]["type"] == "request.cancel"
    assert cancel["params"]["session_id"] == "s1"
    assert cancel["params"]["payload"] == {"id": request["id"], "method": method, "reason": "timeout"}


def test_late_response_and_lock_are_dropped_quietly(server):
    """A response for an id that already ended is ignored (no reply, no error); a late clarify.lock
    answers `expired` instead of a protocol error a card would surface verbatim."""
    assert server.dispatch({"jsonrpc": "2.0", "id": "srq-gone", "result": {"value": ""}}) is None
    response = server.handle_request({"id": "late", "method": "clarify.lock",
                                      "params": {"request_id": "srq-gone", "question_id": "q0", "answer": ""}})
    assert response["result"] == {"status": "expired"}


def _start_batch_clarify(server, buf, qids, timeout=None):
    from tui_gateway import server_requests
    box = {}
    normalized = [{"qid": q, "id": "", "question": q, "choices": None, "choices_offered": [], "multi_select": False}
                  for q in qids]
    if timeout is not None:
        server._clarify_timeout_seconds = lambda: timeout
    thread = threading.Thread(
        target=lambda: box.__setitem__("answer", server._clarify_block("s1", "", None, questions=normalized)), daemon=True)
    thread.start()
    return thread, box, _wait_open(server_requests, buf)


def test_clarify_batch_locks_resolve_in_order_and_keep_partial_on_timeout(capture):
    """Per-question locks (clarify.lock) are editable until every qid is locked; the last lock resolves the
    request with the full answer set. Only wire fields reach the renderer."""
    server, buf = capture
    thread, box, req = _start_batch_clarify(server, buf, ["q0", "q1"])
    sent = _frames(buf)[-1]["params"]["questions"][0]
    assert set(sent) == {"qid", "question", "choices", "multi_select"}

    first = server.handle_request({"id": "a1", "method": "clarify.lock",
                                   "params": {"request_id": req.id, "question_id": "q0", "answer": "x"}})
    assert first["result"] == {"status": "ok", "remaining": ["q1"]}
    redo = server.handle_request({"id": "a1b", "method": "clarify.lock",
                                  "params": {"request_id": req.id, "question_id": "q0", "answer": "y"}})
    assert redo["result"] == {"status": "ok", "remaining": ["q1"]}
    bad = server.handle_request({"id": "bad", "method": "clarify.lock",
                                 "params": {"request_id": req.id, "question_id": "nope", "answer": ""}})
    assert bad["error"]["code"] == 4002
    assert server._open_requests("s1")[0]["params"]["answers"] == {"q0": "y"}
    last = server.handle_request({"id": "a2", "method": "clarify.lock",
                                  "params": {"request_id": req.id, "question_id": "q1", "answer": ""}})
    assert last["result"] == {"status": "ok", "remaining": []}
    thread.join(timeout=5)
    assert json.loads(box["answer"]) == {"answers": {"q0": "y", "q1": ""}}

    # Deadline: locked answers survive, timed_out flagged, one request.cancel.
    original_timeout = server._clarify_timeout_seconds
    try:
        thread, box, req = _start_batch_clarify(server, buf, ["q0", "q1"], timeout=1.5)
        locked = server.handle_request({"id": "b1", "method": "clarify.lock",
                                        "params": {"request_id": req.id, "question_id": "q0", "answer": "kept"}})
        assert locked["result"]["status"] == "ok"
        thread.join(timeout=5)
    finally:
        server._clarify_timeout_seconds = original_timeout
    assert json.loads(box["answer"]) == {"answers": {"q0": "kept"}, "timed_out": True}
    cancels = [f for f in _frames(buf) if f.get("method") == "event" and f["params"]["type"] == "request.cancel"]
    assert [c["params"]["payload"]["id"] for c in cancels] == [req.id]


def test_clarify_batch_cancel_all_is_a_response_without_answers(capture):
    server, buf = capture
    thread, box, req = _start_batch_clarify(server, buf, ["q0", "q1"])
    server.dispatch({"jsonrpc": "2.0", "id": req.id, "result": {}})
    thread.join(timeout=5)
    assert box["answer"] == ""


def test_clear_pending_cancels_only_that_session(capture):
    from tui_gateway import server_requests
    server, buf = capture
    a = threading.Thread(target=lambda: server_requests.send("sudo", "sid-a", {}, timeout=None), daemon=True)
    b = threading.Thread(target=lambda: server_requests.send("sudo", "sid-b", {}, timeout=None), daemon=True)
    a.start(); b.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and len(server_requests._open) < 2:
        time.sleep(0.01)
    assert server._session_pending_kind("sid-a") == "sudo"
    server._clear_pending("sid-a")
    a.join(timeout=2)
    assert not a.is_alive() and b.is_alive()
    assert server._session_pending_kind("sid-a") == "" and server._session_pending_kind("sid-b") == "sudo"
    server._clear_pending()
    b.join(timeout=2)
    assert not b.is_alive()
    reasons = [f["params"]["payload"]["reason"] for f in _frames(buf) if f.get("method") == "event"]
    assert reasons == ["interrupted", "shutdown"]


def test_approval_pending_replays_unresolved_requests(server, monkeypatch):
    from tools import approval

    server._sessions["ui-1"] = {"session_key": "agent-1", "history": []}
    pending = [{"request_id": "req-1", "command": "danger"}]
    monkeypatch.setattr(approval, "list_gateway_approvals", lambda key: pending if key == "agent-1" else [])

    response = server.handle_request(
        {"id": "r1", "method": "approval.pending", "params": {"session_id": "ui-1"}}
    )

    assert response["result"] == {"approvals": pending}


def test_approval_received_acknowledges_exact_request(server, monkeypatch):
    from tools import approval

    server._sessions["ui-1"] = {"session_key": "agent-1", "history": []}
    calls = []
    monkeypatch.setattr(
        approval,
        "ack_gateway_approval",
        lambda key, request_id: calls.append((key, request_id)) or True,
    )

    response = server.handle_request(
        {
            "id": "r2",
            "method": "approval.received",
            "params": {"session_id": "ui-1", "request_id": "req-1"},
        }
    )

    assert response["result"] == {"acknowledged": True}
    assert calls == [("agent-1", "req-1")]


def test_approval_response_correlates_request_id(server, monkeypatch):
    from tools import approval

    server._sessions["ui-1"] = {"session_key": "agent-1", "history": []}
    calls = []
    monkeypatch.setattr(
        approval,
        "resolve_gateway_approval",
        lambda key, choice, **kwargs: calls.append((key, choice, kwargs)) or 1,
    )

    response = server.handle_request(
        {
            "id": "r3",
            "method": "approval.respond",
            "params": {"session_id": "ui-1", "request_id": "req-1", "choice": "once"},
        }
    )

    assert response["result"] == {"resolved": 1}
    assert calls == [("agent-1", "once", {"resolve_all": False, "request_id": "req-1"})]


def test_approval_respond_falls_back_to_request_id_lookup(server, monkeypatch):
    """A stale live sid must not 4001 an approval answer when the request_id
    resolves to a live session (durable-identity fallback, #91684)."""
    from tools import approval

    live = {"session_key": "agent-live", "history": []}
    server._sessions["ui-live"] = live
    calls = []
    monkeypatch.setattr(
        approval,
        "list_gateway_approvals",
        lambda key: [{"request_id": "req-91684"}] if key == "agent-live" else [],
    )
    monkeypatch.setattr(
        approval,
        "resolve_gateway_approval",
        lambda key, choice, **kwargs: calls.append((key, choice, kwargs)) or 1,
    )

    response = server.handle_request(
        {
            "id": "r-fallback",
            "method": "approval.respond",
            "params": {
                "session_id": "gone-sid",
                "request_id": "req-91684",
                "choice": "once",
            },
        }
    )

    assert response["result"] == {"resolved": 1}
    assert calls == [
        ("agent-live", "once", {"resolve_all": False, "request_id": "req-91684"})
    ]


def test_approval_respond_falls_back_to_stored_session_id(server, monkeypatch):
    """session_id holding a STORED id maps to the live runtime record."""
    from tools import approval

    live = {"session_key": "stored-91684", "history": []}
    server._sessions["ui-stored"] = live
    calls = []
    monkeypatch.setattr(approval, "list_gateway_approvals", lambda key: [])
    monkeypatch.setattr(
        approval,
        "resolve_gateway_approval",
        lambda key, choice, **kwargs: calls.append((key, choice, kwargs)) or 1,
    )

    response = server.handle_request(
        {
            "id": "r-stored",
            "method": "approval.respond",
            "params": {"session_id": "stored-91684", "choice": "deny"},
        }
    )

    assert response["result"] == {"resolved": 1}
    assert calls == [
        ("stored-91684", "deny", {"resolve_all": False, "request_id": None})
    ]


def test_approval_respond_4001_when_nothing_resolves(server, monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "list_gateway_approvals", lambda key: [])
    response = server.handle_request(
        {
            "id": "r-nope",
            "method": "approval.respond",
            "params": {"session_id": "nope", "request_id": "req-x", "choice": "once"},
        }
    )

    assert response["error"]["code"] == 4001


# ── Session lookup ───────────────────────────────────────────────────


def test_sess_missing(server):
    _, err = server._sess({"session_id": "nope"}, "r1")
    assert err["error"]["code"] == 4001


# ── session.resume payload ────────────────────────────────────────────


def test_session_resume_returns_hydrated_messages(server, monkeypatch):
    class _DB:
        def get_session(self, _sid):
            return {"id": "20260409_010101_abc123"}

        def get_session_by_title(self, _title):
            return None

        def reopen_session(self, _sid):
            return None

        def get_resume_conversations(self, session_id):
            return (
                self.get_messages_as_conversation(session_id, repair_alternation=True),
                self.get_messages_as_conversation(session_id, include_ancestors=True),
            )

        def get_ancestor_display_prefix(self, _sid):
            return []

        def get_messages_as_conversation(self, _sid, include_ancestors=False, repair_alternation=False):
            return [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "yo", "reasoning": "thoughts"},
                {"role": "tool", "content": "searched"},
                {"role": "assistant", "content": "   "},
                {"role": "assistant", "content": None},
                {"role": "narrator", "content": "skip"},
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, session_id=None, session_db=None, **_kwargs: object())
    monkeypatch.setattr(server, "_init_session", lambda sid, key, agent, history, cols=80, **_kwargs: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: {"model": "test/model"})

    resp = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            # eager_build: exercise the synchronous build path (this test
            # monkeypatches _make_agent/_init_session/_session_info).
            "params": {"session_id": "20260409_010101_abc123", "cols": 100, "eager_build": True},
        }
    )

    assert "error" not in resp
    assert resp["result"]["message_count"] == 3
    assert resp["result"]["messages"] == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "yo", "reasoning": "thoughts"},
        {"role": "tool", "name": "tool", "context": ""},
    ]


def test_session_resume_rejects_runaway_transcript_before_history_load(
    server, monkeypatch
):
    class _DB:
        def get_session(self, sid):
            return {"id": sid, "message_count": 20_001}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, sid):
            return sid

        def reopen_session(self, _sid):
            raise AssertionError("oversized session must be rejected before reopen")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    response = server.handle_request(
        {
            "id": "r1",
            "method": "session.resume",
            "params": {
                "session_id": "runaway-session",
                "omit_messages": True,
            },
        }
    )

    assert response["error"]["code"] == 4130
    assert "limit 20000" in response["error"]["message"]
    assert "hermes sessions export" in response["error"]["message"]


def test_session_resume_deferred_and_omitted_paths_guard_the_tip_only(server, monkeypatch):
    """A deep compression lineage behind a small tip must open on Desktop.

    Desktop's cold resume sends ``defer_history`` + ``omit_messages`` and pages
    the transcript over REST, so the process only ever holds the tip segment.
    Counting the whole lineage there returned 4130 for the healthiest sessions
    (85 compaction segments / ~29k rows / ~700-row tip: Bot Chat stuck on
    "Waking up…"). The guard must count what each path loads.
    """
    calls = []

    class _DB:
        def get_session(self, sid):
            return {"id": sid, "message_count": 28_730}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, sid):
            return sid

        def assert_resume_safe(self, sid, max_messages=None, *, tip_only=False):
            calls.append(tip_only)
            if not tip_only:
                from hermes_state import SessionResumeTooLargeError

                raise SessionResumeTooLargeError(20_001, 20_000)
            return 666

        def reopen_session(self, _sid):
            raise RuntimeError("stop before history load")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)

    for params in (
        {"defer_history": True, "omit_messages": True, "source": "desktop"},
        {"omit_messages": True},
        {"lazy": True},
    ):
        calls.clear()
        response = server.handle_request(
            {
                "id": "r-tip",
                "method": "session.resume",
                "params": {"session_id": "deep-lineage", **params},
            }
        )
        err = response.get("error") or {}
        assert err.get("code") != 4130, params
        assert calls == [True], params

    # The non-deferred, non-omitted resume materializes the full lineage in
    # memory, so it keeps the lineage-wide bound.
    calls.clear()
    response = server.handle_request(
        {"id": "r-full", "method": "session.resume", "params": {"session_id": "deep-lineage"}}
    )
    assert response["error"]["code"] == 4130
    assert calls == [False]


def test_deferred_hydration_falls_back_to_tip_when_lineage_exceeds_limit(server, monkeypatch):
    """The hydration worker never loads a lineage the guard would refuse."""
    import threading

    from hermes_state import SessionResumeTooLargeError

    tip = [{"role": "user", "content": "tip"}]
    reads = []

    class _DB:
        def reopen_session(self, _sid):
            return True

        def assert_resume_safe(self, sid, max_messages=None, *, tip_only=False):
            if not tip_only:
                raise SessionResumeTooLargeError(20_001, 20_000)
            return 1

        def get_resume_conversations(self, _sid):
            reads.append("lineage")
            raise AssertionError("must not materialize the runaway lineage")

        def get_ancestor_display_prefix(self, _sid):
            reads.append("prefix")
            raise AssertionError("must not materialize the runaway lineage")

        def get_messages_as_conversation(self, sid, **kwargs):
            reads.append(("tip", kwargs.get("repair_alternation")))
            return list(tip)

    built = threading.Event()
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: built.set())
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *_a, **_k: None)

    session = server._deferred_session_record(
        "deep-lineage", cols=80, cwd="/tmp", history=[], lease=None
    )
    session["resume_history_ready"] = threading.Event()
    session["resume_hydrating"] = True
    session["resume_message_count"] = 28_730
    server._sessions["hyd"] = session
    try:
        server._schedule_resume_hydration("hyd", "deep-lineage", _DB())
        assert session["resume_history_ready"].wait(timeout=5)
        assert built.wait(timeout=5)
        assert session.get("resume_history_error") is None
        assert session["history"] == tip
        assert session["display_history_prefix"] == []
        assert session["resume_message_count"] == 1
        assert reads == [("tip", True)]
    finally:
        server._sessions.pop("hyd", None)


def test_session_resume_guard_failure_fails_open(server, monkeypatch):
    """A transient guard error must not block resume (fail open, log only)."""
    reopened = []

    class _DB:
        def get_session(self, sid):
            return {"id": sid}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, sid):
            return sid

        def assert_resume_safe(self, _sid):
            raise RuntimeError("database is locked")

        def reopen_session(self, sid):
            reopened.append(sid)
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    response = server.handle_request(
        {
            "id": "r-open",
            "method": "session.resume",
            "params": {
                "session_id": "transient-guard-session",
                "omit_messages": True,
            },
        }
    )

    # The guard must not block: no 4130, and any downstream failure must not
    # be the guard's own "resume safety check failed" error. Reopen being
    # attempted proves execution moved past the guard.
    err = response.get("error") or {}
    assert err.get("code") != 4130
    assert "resume safety check failed" not in str(err.get("message", ""))
    assert reopened == ["transient-guard-session"]


def test_session_resume_active_turn_payload_matches_desktop_fixture(server, monkeypatch):
    """A live resume serializes the exact timer payload consumed by Desktop."""
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "session-resume-active-turn.json").read_text(
            encoding="utf-8"
        )
    )

    class _DB:
        def get_session(self, session_id):
            return {"id": session_id}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, session_id):
            return session_id

    active_turn = {
        "assistant": "partial answer",
        "started_at": fixture["turn_started_at"],
        "streaming": True,
        "user": "current prompt",
    }
    server._sessions[fixture["session_id"]] = {
        "agent": types.SimpleNamespace(session_id=fixture["session_key"]),
        "created_at": fixture["started_at"],
        "history": [{"content": "earlier prompt", "role": "user"}],
        "history_lock": threading.Lock(),
        "inflight_turn": active_turn,
        "running": True,
        "session_key": fixture["session_key"],
    }
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session=None: fixture["info"])

    # JSON round-trip the real RPC envelope: the desktop fixture must stay
    # faithful to what the gateway actually serializes, not a copied shape.
    response = json.loads(
        json.dumps(
            server.handle_request(
                {
                    "id": "resume-running",
                    "method": "session.resume",
                    "params": {"session_id": fixture["session_key"]},
                }
            )
        )
    )
    result = response["result"]

    assert result["running"] is True
    assert result["turn_started_at"] == active_turn["started_at"]
    assert result == fixture


def test_enforce_session_cap_evicts_oldest_detached_only(server, monkeypatch):
    """The LRU cap frees the least-recently-active DETACHED sessions when over
    the limit, and never a live-transport / running / mid-build one."""

    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_live_sessions": 2})
    evicted: list[str] = []
    monkeypatch.setattr(
        server,
        "_close_session_by_id",
        lambda sid, end_reason=None, predicate=None: evicted.append(sid),
    )

    def _ready() -> threading.Event:
        ev = threading.Event()
        ev.set()
        return ev

    detached = server._detached_ws_transport
    live = object()  # no _closed attr -> live transport, never evictable

    server._sessions.clear()
    server._sessions.update(
        {
            "old_detached": {"transport": detached, "last_active": 100.0, "agent_ready": _ready()},
            "new_detached": {"transport": detached, "last_active": 300.0, "agent_ready": _ready()},
            "running_detached": {
                "transport": detached,
                "last_active": 50.0,
                "running": True,
                "agent_ready": _ready(),
            },
            "focused_live": {"transport": live, "last_active": 200.0, "agent_ready": _ready()},
        }
    )

    server._enforce_session_cap()

    # 4 sessions, cap 2 -> evict 2. Only detached+idle+built are eligible, oldest
    # first; the running one and the live-transport one are exempt.
    assert evicted == ["old_detached", "new_detached"]


@pytest.mark.parametrize("closed_transport", [False, True])
def test_idle_reaper_rearms_missing_ws_orphan_timer(server, monkeypatch, tmp_path, closed_transport):
    """A detached lane cannot keep its lease forever if initial timer setup was lost."""
    from hermes_cli.active_sessions import (
        active_session_registry_snapshot,
        try_acquire_active_session,
    )

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    sid = "detached-without-reaper"
    sibling_sid = "live-sibling"
    orphan_lease, message = try_acquire_active_session(
        session_id=sid,
        surface="desktop",
        config={},
        registry_home=home,
        track_liveness=True,
    )
    assert orphan_lease is not None and message is None
    sibling_lease, message = try_acquire_active_session(
        session_id=sibling_sid,
        surface="desktop",
        config={},
        registry_home=home,
        track_liveness=True,
    )
    assert sibling_lease is not None and message is None

    def _session(session_key, lease, transport):
        return {
            "active_session_lease": lease,
            "created_at": time.time(),
            "history": [],
            "history_lock": threading.Lock(),
            "last_active": time.time(),
            "session_key": session_key,
            "source": "tui",
            "transport": transport,
        }

    server._sessions.clear()
    class ClosedTransport:
        _closed = True

    dead_transport = ClosedTransport() if closed_transport else server._detached_ws_transport
    server._sessions.update({
        sid: _session(sid, orphan_lease, dead_transport),
        sibling_sid: _session(sibling_sid, sibling_lease, object()),
    })
    server._pending_ws_reaps.clear()
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.05)
    monkeypatch.setattr(server, "_SESSION_TTL_S", 3600.0)
    monkeypatch.setattr(server, "_flush_dirty_sessions", lambda: 0)
    monkeypatch.setattr(server, "_enforce_session_cap", lambda: None)

    server._reap_idle_sessions()

    deadline = time.monotonic() + 2.0
    while (sid in server._sessions or not orphan_lease.released) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sid not in server._sessions
    assert orphan_lease.released is True
    assert sibling_sid in server._sessions
    assert [entry["session_id"] for entry in active_session_registry_snapshot(home)] == [sibling_sid]

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH", "")) if part
    )
    successor = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from hermes_cli.active_sessions import try_acquire_active_session; "
                f"lease, refusal = try_acquire_active_session(session_id={sid!r}, surface='desktop', "
                "config={}, track_liveness=True); "
                "assert lease is not None and refusal is None, refusal; lease.release()"
            ),
        ],
        cwd=repo_root,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert successor.returncode == 0, successor.stderr
    assert [entry["session_id"] for entry in active_session_registry_snapshot(home)] == [sibling_sid]


def test_sync_session_key_after_compress_reanchors_active_session_lease(
    server, monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.active_sessions import (
        active_session_registry_snapshot,
        try_acquire_active_session,
    )

    lease, message = try_acquire_active_session(
        session_id="session-old",
        surface="tui",
        config={"max_concurrent_sessions": 1},
        metadata={"live_session_id": "ui-1"},
    )
    assert message is None
    assert lease is not None

    session = {
        "active_session_lease": lease,
        "agent": types.SimpleNamespace(session_id="session-new"),
        "session_key": "session-old",
    }
    fake_approval = types.SimpleNamespace(
        disable_session_yolo=lambda *_args, **_kwargs: None,
        enable_session_yolo=lambda *_args, **_kwargs: None,
        is_session_yolo_enabled=lambda *_args, **_kwargs: False,
        register_gateway_notify=lambda *_args, **_kwargs: None,
        unregister_gateway_notify=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args, **_kwargs: None)

    with patch.dict(sys.modules, {"tools.approval": fake_approval}):
        server._sync_session_key_after_compress("ui-1", session)

    snapshot = active_session_registry_snapshot()
    assert session["session_key"] == "session-new"
    assert lease.session_id == "session-new"
    assert [entry["session_id"] for entry in snapshot] == ["session-new"]
    lease.release()


def test_make_agent_accepts_list_system_prompt(server, monkeypatch):
    captured = {}

    class _Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model", "")

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=_Agent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        types.SimpleNamespace(
            resolve_runtime_provider=lambda **_kwargs: {
                "provider": "test",
                "base_url": None,
                "api_key": None,
                "api_mode": None,
            }
        ),
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {"agent": {"system_prompt": ["one", "two"]}})
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("test/model", "test"))
    monkeypatch.setattr(server, "_get_db", lambda: None)

    server._make_agent("sid", "session-key", session_id="session-key")

    assert captured["ephemeral_system_prompt"] == "one\ntwo"


# ── Config I/O ───────────────────────────────────────────────────────


def test_config_roundtrip(server, tmp_path):
    server._hermes_home = tmp_path
    server._save_cfg({"model": "test/model"})
    assert server._load_cfg()["model"] == "test/model"


# ── _cli_exec_blocked ────────────────────────────────────────────────


@pytest.mark.parametrize("argv", [
    [],
    ["setup"],
    ["gateway"],
    ["sessions", "browse"],
    ["config", "edit"],
])
def test_cli_exec_blocked(server, argv):
    assert server._cli_exec_blocked(argv) is not None


# ── slash.exec skill command interception ────────────────────────────


def test_slash_exec_rejects_skill_commands(server):
    """slash.exec must reject skill commands so the TUI falls through to command.dispatch."""
    # Register a mock session
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid, "agent": None}

    # Mock scan_skill_commands to return a known skill
    fake_skills = {"/hermes-agent-dev": {"name": "hermes-agent-dev", "description": "Dev workflow"}}

    with patch("agent.skill_commands.get_skill_commands", return_value=fake_skills):
        resp = server.handle_request({
            "id": "r1",
            "method": "slash.exec",
            "params": {"command": "hermes-agent-dev", "session_id": sid},
        })

    # Should return an error so the TUI's .catch() fires command.dispatch
    assert "error" in resp
    assert resp["error"]["code"] == 4018
    assert "skill command" in resp["error"]["message"]


def test_slash_exec_scopes_skill_lookup_to_session_profile(server, tmp_path):
    """slash.exec must resolve get_skill_commands() against the session's own
    profile_home rather than the gateway process's ambient HERMES_HOME
    (#88023). A Desktop session that switches profiles mid-session shares
    the same gateway process, so a skill declared only under the new
    profile's skills.external_dirs must still be recognized here — else the
    command falls through to the slash-worker dead path instead of routing
    to command.dispatch.
    """
    import agent.skill_commands as sc_mod

    empty_local_dir = tmp_path / "no-local-skills"
    empty_local_dir.mkdir()

    profile_b = tmp_path / "profile_b"
    external_b = tmp_path / "external_b"
    profile_b.mkdir()
    skill_dir = external_b / "b-only"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: b-only\ndescription: Only in profile b.\n---\n\n# b-only\n\nDo the thing.\n"
    )
    (profile_b / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_b}\n"
    )

    sid = "test-session-profile-b"
    server._sessions[sid] = {
        "session_key": sid,
        "agent": None,
        "profile_home": str(profile_b),
    }

    with (
        patch("tools.skills_tool.SKILLS_DIR", empty_local_dir),
        patch.object(sc_mod, "_skill_commands", {}),
        patch.object(sc_mod, "_skill_commands_platform", None),
        patch.object(sc_mod, "_skill_commands_home", None),
    ):
        resp = server.handle_request({
            "id": "r1",
            "method": "slash.exec",
            "params": {"command": "b-only", "session_id": sid},
        })

    # The gateway's own HERMES_HOME (the test-isolation tempdir, no
    # skills.external_dirs) has no "b-only" skill — the only way this
    # resolves is by scoping the lookup to the session's profile_home.
    assert "error" in resp
    assert resp["error"]["code"] == 4018
    assert "skill command" in resp["error"]["message"]


def test_command_dispatch_scopes_skill_lookup_to_session_profile(server, tmp_path):
    """command.dispatch must load a skill that exists only in the session profile."""
    import agent.skill_commands as sc_mod

    empty_local_dir = tmp_path / "no-local-skills"
    empty_local_dir.mkdir()

    profile_b = tmp_path / "profile_b"
    external_b = tmp_path / "external_b"
    profile_b.mkdir()
    skill_dir = external_b / "b-only"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: b-only\ndescription: Only in profile b.\n---\n\n# b-only\n\nDo the thing.\n"
    )
    (profile_b / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_b}\n"
    )

    sid = "test-session-profile-b-dispatch"
    server._sessions[sid] = {
        "session_key": sid,
        "agent": None,
        "profile_home": str(profile_b),
    }

    with (
        patch("tools.skills_tool.SKILLS_DIR", empty_local_dir),
        patch.object(sc_mod, "_skill_commands", {}),
        patch.object(sc_mod, "_skill_commands_platform", None),
        patch.object(sc_mod, "_skill_commands_home", None),
    ):
        resp = server.handle_request({
            "id": "r1",
            "method": "command.dispatch",
            "params": {"name": "b-only", "arg": "with an argument", "session_id": sid},
        })

    assert "error" not in resp
    assert resp["result"]["type"] == "skill"
    assert resp["result"]["name"] == "b-only"


def test_slash_exec_routes_a_secondary_only_bundle_to_dispatch(server, tmp_path, monkeypatch):
    """A skill bundle that exists only under the session profile's ``skill-bundles/`` must be
    resolved (and routed to command.dispatch) against that profile, not the launch home (#110695)."""
    import agent.skill_bundles as sb_mod
    import agent.skill_commands as sc_mod

    monkeypatch.delenv("HERMES_BUNDLES_DIR", raising=False)
    profile_b = tmp_path / "profile_b"
    external_b = tmp_path / "external_b"
    for name in ("one", "two"):
        d = external_b / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name}.\n---\n\n# {name}\n")
    (profile_b / "skill-bundles").mkdir(parents=True)
    (profile_b / "skill-bundles" / "b-pack.yaml").write_text("name: b-pack\nskills: [one, two]\n")
    (profile_b / "config.yaml").write_text(f"skills:\n  external_dirs:\n    - {external_b}\n")
    sid = "test-session-profile-b-bundle"
    server._sessions[sid] = {"session_key": sid, "agent": None, "profile_home": str(profile_b)}

    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path / "no-local-skills"),
        patch.object(sb_mod, "_bundles_cache", {}),
        patch.object(sb_mod, "_bundles_cache_mtime", None),
        patch.object(sc_mod, "_skill_commands", {}),
        patch.object(sc_mod, "_skill_commands_home", None),
    ):
        resp = server.handle_request({
            "id": "r1", "method": "slash.exec", "params": {"command": "/b-pack go", "session_id": sid}})

    assert "error" not in resp, resp
    assert resp["result"]["type"] == "send" and "b-pack" in resp["result"]["notice"]
def test_slash_exec_routes_quick_command_alias_to_skill(server, monkeypatch):
    """A quick-command alias whose target is a skill must reach command.dispatch,
    not the slash worker. On the worker path the CLI queues the loaded skill onto
    _pending_input, which only the interactive REPL drains, so it is silently
    dropped on desktop/TUI (#106063)."""
    import agent.skill_commands as sc_mod

    sid = "test-alias-skill"
    # No slash_worker: if the alias fell through to the worker path this would
    # try to spawn one; asserting a skill result proves it never got there.
    server._sessions[sid] = {"session_key": sid, "agent": None}

    monkeypatch.setattr(
        server, "_load_cfg",
        lambda: {"quick_commands": {"tr": {"type": "alias", "target": "/trading-tr-hermes"}}},
    )
    fake_cmds = {"/trading-tr-hermes": {"name": "trading-tr-hermes", "description": "Trading skill"}}
    monkeypatch.setattr(sc_mod, "get_skill_commands", lambda: fake_cmds)
    monkeypatch.setattr(sc_mod, "scan_skill_commands", lambda: fake_cmds)
    monkeypatch.setattr(
        sc_mod, "build_skill_invocation_message",
        lambda key, arg, task_id="": f"[IMPORTANT: run {key.lstrip('/')}] {arg}".strip(),
    )

    resp = server.handle_request({
        "id": "r1",
        "method": "slash.exec",
        "params": {"command": "tr EURUSD", "session_id": sid},
    })

    assert "error" not in resp, resp
    result = resp["result"]
    assert result["type"] == "skill"
    assert result["name"] == "trading-tr-hermes"
    # The user's argument reaches the resolved skill.
    assert "EURUSD" in result["message"]


def test_alias_skill_target_resolution(server, monkeypatch):
    """_alias_skill_target resolves only alias→skill, prepends the target's own
    args, and returns None for non-alias / exec / non-skill targets."""
    import agent.skill_commands as sc_mod

    session = {"session_key": "s", "agent": None}
    cfg = {"quick_commands": {
        "tr": {"type": "alias", "target": "/trading-tr-hermes preset"},
        "sh": {"type": "alias", "target": "/some-helper"},
        "run": {"type": "exec", "command": "echo hi"},
    }}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    # Only /trading-tr-hermes is a skill; /some-helper is not.
    monkeypatch.setattr(
        sc_mod, "get_skill_commands",
        lambda: {"/trading-tr-hermes": {"name": "trading-tr-hermes"}},
    )

    # alias→skill: target's embedded arg is prepended to the user arg.
    assert server._alias_skill_target(session, "tr", "EURUSD") == ("trading-tr-hermes", "preset EURUSD")
    # alias→skill with no user arg keeps just the target's embedded arg.
    assert server._alias_skill_target(session, "tr", "") == ("trading-tr-hermes", "preset")
    # alias whose target is not a skill → None (worker/plugin path unchanged).
    assert server._alias_skill_target(session, "sh", "") is None
    # exec-type quick command → None.
    assert server._alias_skill_target(session, "run", "") is None
    # unknown base → None.
    assert server._alias_skill_target(session, "nope", "") is None


def test_command_dispatch_queue_sends_message(server):
    """command.dispatch /queue returns {type: 'send', message: ...} for the TUI."""
    sid = "test-session"
    server._sessions[sid] = {"session_key": sid}

    resp = server.handle_request({
        "id": "r1",
        "method": "command.dispatch",
        "params": {"name": "queue", "arg": "tell me about quantum computing", "session_id": sid},
    })

    assert "error" not in resp
    result = resp["result"]
    assert result["type"] == "send"
    assert result["message"] == "tell me about quantum computing"


def test_skills_manage_search_uses_tools_hub_sources(server):
    result = type("Result", (), {
        "description": "Build better terminal demos",
        "name": "showroom",
    })()
    auth = MagicMock(return_value="auth")
    router = MagicMock(return_value=["source"])
    search = MagicMock(return_value=[result])
    fake_search = types.SimpleNamespace(create_source_router=router, unified_search=search)
    fake_github = types.SimpleNamespace(GitHubAuth=auth)

    with patch.dict(sys.modules, {"tools.skills_hub_search": fake_search, "tools.skills_hub_github": fake_github}):
        resp = server.handle_request({
            "id": "skills-search",
            "method": "skills.manage",
            "params": {"action": "search", "query": "showroom"},
        })

    assert "error" not in resp
    assert resp["result"] == {
        "results": [{"description": "Build better terminal demos", "name": "showroom"}]
    }
    auth.assert_called_once_with()
    router.assert_called_once_with("auth")
    search.assert_called_once_with("showroom", ["source"], source_filter="all", limit=20)


# ── dispatch(): pool routing for long handlers (#12546) ──────────────


def test_dispatch_runs_short_handlers_inline(server):
    """Non-long handlers return their response synchronously from dispatch()."""
    server._methods["fast.ping"] = lambda rid, params: server._ok(rid, {"pong": True})

    resp = server.dispatch({"id": "r1", "method": "fast.ping", "params": {}})

    assert resp == {"jsonrpc": "2.0", "id": "r1", "result": {"pong": True}}


@pytest.mark.parametrize("completion_method", ["complete.path", "complete.slash"])
def test_completion_handlers_are_pool_routed(completion_method, server):
    """complete.path/complete.slash must run on the pool, never the reader thread.

    Regression for #21123: completion ran inline, so a slow git ls-files /
    skill-scan blocked prompt.submit and froze the TUI for the 120s RPC timeout.
    """
    assert completion_method in server._LONG_HANDLERS


@pytest.mark.parametrize(
    "voice_or_wake_method",
    ["voice.toggle", "voice.record", "voice.tts", "wake.start", "wake.status"],
)
def test_voice_and_wake_handlers_are_pool_routed(voice_or_wake_method, server):
    """Voice and wake RPCs must run on the pool, never the WS reader thread.

    Regression: voice.toggle (status) triggers check_voice_requirements() →
    STT provider auto-detect → a SYNCHRONOUS faster-whisper lazy install (uv/pip
    subprocess, up to a 300s timeout). Inline on the WS reader loop it blocked
    prompt.submit / session.list frames queued behind it — the desktop showed
    sent messages that never reached the agent. Same bug class as #21123 /
    #50005: anything that can stall for seconds must stay off the reader thread.

    wake.start and wake.status share the same STT lazy-install path via
    check_wake_word_requirements() → _stt_ready() → _get_provider(), and
    wake.start additionally calls lazy_deps.ensure() for wake-word engine deps.
    The desktop polls wake.status on every gateway-ready.
    """
    assert voice_or_wake_method in server._LONG_HANDLERS


def test_skin_live_switch_end_to_end(server, tmp_path, monkeypatch):
    """Real config + skin files: activating a skin (as `hermes config set` does)
    makes the per-tool reconcile broadcast skin.changed with the resolved palette.
    Exercises _load_cfg → _skin_sig → resolve_skin → _emit with no mocks in between."""
    import hermes_cli.skin_engine as skin_engine

    (tmp_path / "skins").mkdir()
    (tmp_path / "skins" / "midnight.yaml").write_text(
        "name: midnight\ndescription: t\ncolors:\n  banner_title: '#00ffcc'\n  background: '#001010'\n"
    )
    monkeypatch.setattr(skin_engine, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_last_skin_sig", None, raising=False)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload=None: emitted.append((ev, payload)))

    # Baseline (default) — seeds the signature.
    (tmp_path / "config.yaml").write_text("display:\n  skin: default\n", encoding="utf-8")
    server._broadcast_skin_if_changed()
    emitted.clear()

    # Activate midnight, as `hermes config set display.skin midnight` would.
    time.sleep(0.01)  # ensure the config mtime moves
    (tmp_path / "config.yaml").write_text("display:\n  skin: midnight\n", encoding="utf-8")
    server._broadcast_skin_if_changed()

    assert [ev for ev, _ in emitted] == ["skin.changed"]
    assert emitted[0][1]["name"] == "midnight"
    assert emitted[0][1]["colors"]["banner_title"] == "#00ffcc"


def test_broadcast_skin_if_changed_on_any_signature_move(server, monkeypatch):
    """A skin the agent changes mid-turn goes live once per real move: a name
    switch (incl. switch-then-revert) OR an in-place color edit to the active skin
    (same name, new file mtime). An unchanged signature never re-broadcasts."""
    emitted = []
    # switch, no-op, switch, then a color edit (same name, bumped mtime).
    sigs = iter([("neon", 1.0), ("neon", 1.0), ("forest", 1.0), ("forest", 2.0)])
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload=None: emitted.append((ev, payload)))
    monkeypatch.setattr(server, "_last_skin_sig", None, raising=False)
    monkeypatch.setattr(server, "_skin_sig", lambda: next(sigs))
    monkeypatch.setattr(server, "resolve_skin", lambda: {"name": "x", "colors": {}})

    for _ in range(4):
        server._broadcast_skin_if_changed()

    assert [ev for ev, _ in emitted] == ["skin.changed"] * 3


# ── global-event broadcast (session-less events reach every WS client) ──


class _RecordingTransport:
    """Minimal Transport stand-in that records the frames written to it."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    def write(self, obj: dict) -> bool:
        self.frames.append(obj)
        return True

    def close(self) -> None:
        pass


def test_unregister_live_transport_stops_delivery(capture):
    """A disconnected peer (unregistered in the ws finally block) receives nothing
    — and a stale write is never attempted against its closed socket."""
    server, buf = capture
    a = _RecordingTransport()
    server.register_live_transport(a)
    server.unregister_live_transport(a)

    server._broadcast_global_event("skin.changed", {"name": "x"})

    assert a.frames == []
    # No live transports left → fell back to stdio.
    assert json.loads(buf.getvalue())["params"]["type"] == "skin.changed"


def test_approval_for_a_ws_client_that_never_advertised_settles_the_queue_entry(server, monkeypatch):
    """The approval wait is owned by ``tools.approval``'s queue, not by ``server_requests``. When the request
    cannot be sent (the only client predates server→client requests) the queue entry must be withdrawn too,
    otherwise ``_await_gateway_decision`` idles for the whole approvals.timeout with no prompt anywhere
    (#112548). The decision is a withdrawal (``cancelled`` cause), never a user deny."""
    from tools import approval as approval_mod
    from tools import approval_gateway_wait as wait_mod

    peer = _silent_ws()
    _ws_session(server, "ws-old-approval", peer)
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 3)
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    approval_mod.register_gateway_notify("ws-old-approval", lambda data: server._emit_approval_request("ws-old-approval", data))
    try:
        t0 = time.monotonic()
        decision = wait_mod._await_gateway_decision(
            "ws-old-approval", approval_mod._gateway_notify_cbs["ws-old-approval"],
            {"command": "rm -rf build", "description": "", "pattern_key": "dangerous", "pattern_keys": ["dangerous"]})
        waited = time.monotonic() - t0
    finally:
        approval_mod.unregister_gateway_notify("ws-old-approval")
    assert waited < 1, decision
    assert decision["choice"] is None and decision["cancelled"]
    assert peer.frames == []
    assert "ws-old-approval" not in approval_mod._gateway_queues
