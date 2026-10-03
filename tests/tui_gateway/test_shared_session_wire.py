"""Execution observations over real WebSockets, with only inference gated."""

import json
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from websockets.sync.client import connect

from tui_gateway import server
from tui_gateway.ws import handle_ws


@pytest.fixture
def turn_env(monkeypatch, tmp_path):
    # These are unrelated provider, audio and background-service startup hooks.
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd",
                 "_sync_session_key_after_compress", "_ensure_skin_watcher",
                 "_start_backend_heartbeat_refresher", "_schedule_startup_orphan_sweep"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "resolve_skin", lambda: {})
    monkeypatch.setattr(server, "_start_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *a: True)
    return tmp_path


class Inference:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = []
        self.callbacks = []
        self.session_id = "shared-wire-key"
        self.error = None

    def clear_interrupt(self):
        pass

    def run_conversation(self, prompt, *, conversation_history=None, stream_callback=None, **kwargs):
        self.calls.append((prompt, conversation_history, kwargs))
        self.callbacks.append(stream_callback)
        self.entered.set()
        assert self.release.wait(10), "test failed to release inference"
        # Exercise the provider callback from a different thread.
        worker = threading.Thread(target=lambda: stream_callback("answer"))
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        result = {"final_response": "answer", "messages": [*(conversation_history or []),
                  {"role": "user", "content": prompt}, {"role": "assistant", "content": "answer"}]}
        if self.error:
            result.update(error=self.error, failed=True)
        return result


def session_for(agent):
    ready = threading.Event()
    ready.set()
    return dict(agent=agent, agent_ready=ready, session_key="shared-wire-key", history=[],
                history_lock=threading.Lock(), history_version=0, running=False, transport=None,
                attached_images=[], image_counter=0, cols=80, slash_worker=None, show_reasoning=False,
                tool_progress_mode="all", inflight_turn=None,
                active_session_lease=SimpleNamespace(release=lambda: None))


class Peer:
    def __init__(self, ws):
        self.ws, self.seen, self.request_id = ws, [], 0
        self.ready = self.until(lambda f: f.get("params", {}).get("type") == "gateway.ready")["params"]["payload"]

    def until(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            frame = json.loads(self.ws.recv(timeout=max(.01, deadline - time.monotonic())))
            self.seen.append(frame)
            if predicate(frame):
                return frame
        raise AssertionError(self.seen)

    def rpc(self, method, **params):
        self.request_id += 1
        self.ws.send(json.dumps(dict(jsonrpc="2.0", id=self.request_id, method=method, params=params)))
        frame = self.until(lambda f: f.get("id") == self.request_id)
        assert "error" not in frame, frame
        return frame["result"]

    def event(self, kind):
        return self.until(lambda f: f.get("params", {}).get("type") == kind)["params"]


@contextmanager
def endpoint():
    app = Starlette(routes=[WebSocketRoute("/ws", handle_ws)])
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        web = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        worker = threading.Thread(target=lambda: web.run(sockets=[listener]), daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not web.started and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            assert web.started
            yield f"ws://127.0.0.1:{port}/ws"
        finally:
            web.should_exit = True
            worker.join(10)
            assert not worker.is_alive()


_SKILL_INPUT = ('[IMPORTANT: The user has invoked the "work" skill. '
                'The full skill content is loaded below.]\nInternal skill body.\n'
                'The user has provided the following instruction alongside the skill invocation: fix it')


@pytest.mark.parametrize("text,display_kind,expected", [
    ("hello", None, {"role": "user", "text": "hello"}),
    ("hello", "hidden", None),
    (_SKILL_INPUT, None, {"role": "user", "text": "/work fix it", "display_kind": "skill_invocation"}),
])
def test_two_clients_observe_start_before_output_and_reconnect(turn_env, monkeypatch, text, display_kind, expected):
    inference = Inference()
    session = session_for(inference)
    monkeypatch.setitem(server._sessions, "wire", session)
    try:
        with endpoint() as url, connect(url) as socket_a, connect(url) as socket_b:
            a, b = Peer(socket_a), Peer(socket_b)
            for peer in (a, b):
                peer.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)
            session["agent_ready"].clear()
            reply = a.rpc("prompt.submit", session_id="wire", text=text, display_kind=display_kind,
                          submission_ref="lost-reply-reference")
            assert not inference.entered.is_set()
            pending = b.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)["inflight"]
            assert pending.get("inputs") == [{"id": reply.get("submission", {}).get("input_id"),
                                              "ref": "lost-reply-reference"}]
            session["agent_ready"].set()
            assert inference.entered.wait(10)
            start = b.event("message.start")
            # Baseline fails here on missing starting input, before any new API helper is called.
            assert start.get("payload", {}).get("input", "missing") == expected
            assert not any(f.get("params", {}).get("type") == "message.delta" for f in b.seen)
            assert start["turn"]["source"] == {"kind": "connection",
                    "socket_id": a.ready["shared_session"]["socket_id"]}
            assert not a.ready.get("session_mirroring", False)
            assert "origin" not in start
            snapshot = b.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)["inflight"]
            assert (snapshot["turn"], snapshot["input"]) == (start["turn"], expected)
            assert snapshot["user"] == ("" if expected is None else server._inflight_text(text))
            assert start["payload"]["inputs"] == pending["inputs"]
            socket_a.close()
            with connect(url) as socket_c:
                c = Peer(socket_c)
                assert c.ready["shared_session"]["socket_id"] != a.ready["shared_session"]["socket_id"]
                resumed = c.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)
                assert resumed["inflight"]["turn"] == start["turn"]
                replay = c.rpc("session.events.since", session_id="wire", last_seen=0)
                assert any(e.get("turn") == start["turn"] for e in replay["events"])
                inference.release.set()
                for peer in (b, c):
                    assert peer.event("message.delta")["turn"] == start["turn"]
                    assert peer.event("message.complete")["turn"] == start["turn"]
                completed = c.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)
                assert any(o["disposition"] == "terminal" and o["input"]["ref"] == "lost-reply-reference"
                           and o["turn"] == start["turn"] for o in completed["submission_state"]["outcomes"])
        session["_run_thread"].join(10)
        assert not session["_run_thread"].is_alive()
        assert inference.calls[0][0] == text
        assert inference.calls[0][1] == []
        assert "turn" not in json.dumps(inference.calls[0][2])
        assert all("turn" not in row for row in session["history"])
    finally:
        inference.release.set()
        session["agent_ready"].set()
        if session.get("_run_thread"):
            session["_run_thread"].join(10)


def test_late_stream_cannot_mutate_next_turn_or_borrow_its_identity(turn_env, monkeypatch):
    first = Inference()
    first.error = "provider failed"
    session = session_for(first)
    monkeypatch.setitem(server._sessions, "wire", session)
    events = []
    session["transport"] = SimpleNamespace(write=lambda frame: events.append(frame) or True)
    first.release.set()
    server._run_prompt_submit(1, "wire", session, "first", turn_source={"kind": "connection", "socket_id": "a"})
    session["_run_thread"].join(10)
    old = next(f["params"]["turn"] for f in events if f["params"]["type"] == "message.start")
    failed = server._inflight_snapshot(session)
    assert failed["status"] == "error"
    first.callbacks[0]("late after failure")
    assert server._inflight_snapshot(session) == failed
    second = Inference()
    session["agent"] = second
    session["running"] = True
    try:
        server._run_prompt_submit(2, "wire", session, "automatic", display_kind="auto_continue")
        assert second.entered.wait(10)
        snapshot = server._inflight_snapshot(session)
        assert snapshot["turn"]["id"] != old["id"]
        assert snapshot["turn"]["source"] == {"kind": "unknown"}
        first.callbacks[0]("stale text")
        assert server._inflight_snapshot(session) == snapshot
        server._emit("session.info", "wire", {})
        assert "turn" not in events[-1]["params"]
        from tui_gateway.turn_observation import make_turn, turn_scope
        with turn_scope(make_turn("wire", session)):
            assert "turn" not in server._event_frame("message.delta", "child", {"text": "child output"})["params"]
            replacement = dict(session)
            monkeypatch.setitem(server._sessions, "wire", replacement)
            assert "turn" not in server._event_frame("message.delta", "wire", {"text": "retired record"})["params"]
            monkeypatch.setitem(server._sessions, "wire", session)
    finally:
        second.release.set()
        session["_run_thread"].join(10)


@pytest.mark.parametrize("other_id,hidden", [("a", False), ("b", False)])
def test_queued_sources_survive_disconnect_and_merge_without_changing_model_text(turn_env, monkeypatch, other_id, hidden):
    inference = Inference()
    inference.release.set()
    session = session_for(inference)
    events = []
    session["transport"] = SimpleNamespace(write=lambda frame: events.append(frame) or True)
    monkeypatch.setitem(server._sessions, "wire", session)
    dead_sender = SimpleNamespace(_closed=True)
    a = {"kind": "connection", "socket_id": "a"}
    b = {"kind": "connection", "socket_id": other_id}
    server._enqueue_prompt(session, "first", dead_sender, turn_source=a)
    server._enqueue_prompt(session, "second", dead_sender, turn_source=b, display_kind="hidden" if hidden else None)
    assert server._drain_queued_prompt(1, "wire", session)
    session["_run_thread"].join(10)
    start = next(f["params"] for f in events if f["params"]["type"] == "message.start")
    assert start["turn"]["source"] == (a if other_id == "a" else {"kind": "mixed"})
    assert start["payload"]["input"] == (None if hidden else {"role": "user", "text": "first\n\nsecond"})
    assert inference.calls[0][0] == "first\n\nsecond"


@pytest.mark.parametrize("hidden_first", [False, True])
def test_hidden_and_visible_queue_entries_keep_their_visibility(hidden_first):
    session = session_for(Inference())
    entries = [("visible words", None), ("private scaffolding", "hidden")]
    if hidden_first:
        entries.reverse()
    for text, kind in entries:
        server._enqueue_prompt(session, text, None, display_kind=kind)
    queued = [session["queued_prompt"], *session.get("queued_prompts", [])]
    assert [(e["text"], e.get("display_kind")) for e in queued] == entries
    snapshot = server._queued_prompt_snapshot(session)
    assert snapshot["user"] == ("" if hidden_first else "visible words")


def test_hidden_busy_input_queues_without_interrupting(monkeypatch):
    session = session_for(Inference())
    session["running"] = True
    interrupted = []
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setattr(server, "_interrupt_busy_session", lambda *a: interrupted.append(a))
    reply = server._handle_busy_submit(1, "wire", session, "private scaffolding", None, display_kind="hidden")
    assert reply["result"]["status"] == "queued"
    assert not interrupted
    assert session["queued_prompt"]["display_kind"] == "hidden"


def test_simultaneous_idle_claim_has_one_winner():
    session = session_for(Inference())
    barrier = threading.Barrier(2)
    results = []

    def claim(text):
        barrier.wait(5)
        results.append(server._lock_in_submit_turn(1, "wire", session, text, {}, False, None, None, None))

    workers = [threading.Thread(target=claim, args=(text,)) for text in ("a", "b")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    assert results.count((None, {})) == 1
    assert results.count((None, None)) == 1


def test_redirect_during_agent_build_captures_the_request_connection(monkeypatch):
    session = session_for(None)
    session["running"] = True
    transport = SimpleNamespace(socket_id="redirect-socket", write=lambda frame: True)
    session["transport"] = transport
    monkeypatch.setitem(server._sessions, "wire", session)
    token = server.bind_transport(transport)
    try:
        response = server._methods["session.redirect"](1, {"session_id": "wire", "text": "follow up"})
    finally:
        server.reset_transport(token)
    assert response["result"]["status"] == "queued"
    assert session["queued_prompt"]["turn_source"] == {"kind": "connection", "socket_id": "redirect-socket"}


def test_hidden_desktop_correction_waits_and_preserves_visible_history(turn_env, monkeypatch):
    inference = Inference()
    inference.interrupt = lambda: pytest.fail("hidden note interrupted the live turn")
    session = session_for(inference)
    monkeypatch.setitem(server._sessions, "wire", session)
    try:
        with endpoint() as url, connect(url) as socket_a, connect(url) as socket_b:
            a, b = Peer(socket_a), Peer(socket_b)
            for peer in (a, b):
                peer.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)
            a.rpc("prompt.submit", session_id="wire", text="visible words")
            assert inference.entered.wait(5)
            first = b.event("message.start")
            reply = a.rpc("session.steer", session_id="wire", text="private scaffolding",
                          input_visibility="hidden", submission_ref="hidden-ref")
            assert reply["status"] == "queued"
            queued = b.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)["queued"]
            assert queued["user"] == "" and queued["input"] is None
            assert queued["inputs"] == [{"id": reply["submission"]["input_id"], "ref": "hidden-ref"}]
            inference.release.set()
            assert b.event("message.complete")["turn"] == first["turn"]
            second = b.event("message.start")
            assert second["turn"]["id"] != first["turn"]["id"]
            assert second["payload"]["input"] is None
            assert second["payload"]["inputs"] == queued["inputs"]
            b.event("message.complete")
            session["_run_thread"].join(10)
            visible = server._history_to_messages(session["history"])
            assert [m["text"] for m in visible if m["role"] == "user"] == ["visible words"]
            assert [call[0] for call in inference.calls] == ["visible words", "private scaffolding"]
    finally:
        inference.release.set()
        if session.get("_run_thread"):
            session["_run_thread"].join(10)


def test_proven_failed_compute_dispatch_reuses_execution_on_inline_fallback(turn_env, monkeypatch):
    from tui_gateway.input_observation import new_input
    inference = Inference()
    inference.release.set()
    session = session_for(inference)
    session["running"] = True
    frames, dispatched = [], []
    session["transport"] = SimpleNamespace(write=lambda frame: frames.append(frame) or True)
    monkeypatch.setitem(server._sessions, "wire", session)

    def fail(frame, **kwargs):
        dispatched.append(frame)
        raise BrokenPipeError("no child received this test dispatch")

    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *a: SimpleNamespace(submit_turn=fail))
    batch = new_input(session, "visible words", "fallback-ref")
    response = server._submit_prompt_to_compute_host(1, "wire", session, "visible words", input_batch=batch)
    assert response.get("error")
    server._run_prompt_submit(1, "wire", session, "visible words", input_batch=batch,
                              turn_context=session["inflight_turn"]["turn"])
    session["_run_thread"].join(10)
    starts = [f["params"] for f in frames if f["params"]["type"] == "message.start"]
    assert len(starts) == 1 and starts[0]["turn"] == dispatched[0]["turn_context"]
    assert starts[0]["payload"]["inputs"][0]["ref"] == "fallback-ref"


def test_observed_submit_adopts_foreign_durable_turn_without_duplicate(turn_env, monkeypatch):
    """Observation must surround current row adoption, never replace its durable/model boundary."""
    from contextlib import nullcontext
    from agent.turn_context import _stage_turn_user_message
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(turn_env / "observed-state.db")
    db.create_session("shared-wire-key", source="desktop")
    for role, content in [("user", "local question"), ("assistant", "same answer")]:
        db.append_message("shared-wire-key", role, content)
    local = db.get_messages_as_conversation("shared-wire-key", include_row_ids=True)
    foreign = [("user", "external question"), ("assistant", "same answer")]
    foreign_ids = [db.append_message("shared-wire-key", role, content) for role, content in foreign]

    class PersistingInference(Inference):
        def run_conversation(self, prompt, *, conversation_history=None, stream_callback=None, **kwargs):
            self.calls.append((prompt, conversation_history, kwargs))
            user, _ = _stage_turn_user_message(self, prompt, kwargs["persist_user_message"], None, None, None, None)
            messages = [*conversation_history, user]
            self._persist_user_message_idx = len(conversation_history)
            self._flush_messages_to_session_db(messages, [])
            stream_callback("new answer")
            messages.append({"role": "assistant", "content": "new answer"})
            self._flush_messages_to_session_db(messages, [])
            return {"final_response": "new answer", "messages": messages}

    agent = PersistingInference()
    agent.__dict__.update(
        _session_db=db, _session_db_created=True, _persist_disabled=False,
        _session_persist_lock=None, _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None)
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    session = session_for(agent)
    session["history"] = local
    frames = []
    session["transport"] = SimpleNamespace(write=lambda frame: frames.append(frame) or True)
    monkeypatch.setitem(server._sessions, "wire", session)
    monkeypatch.setattr(server, "_session_db", lambda _: nullcontext(db))
    try:
        reply = server.handle_request({"id": 11, "method": "prompt.submit", "params": {
            "session_id": "wire", "text": "new question", "submission_ref": "current-row"}})
        assert "error" not in reply, reply
        deadline = time.monotonic() + 10
        while session["running"] and time.monotonic() < deadline:
            session["_run_thread"].join(.05)
        assert not session["running"]
        assert len(agent.calls) == 1
        model_history = agent.calls[0][1]
        assert [(m["role"], m["content"]) for m in model_history] == [
            ("user", "local question"), ("assistant", "same answer"), *foreign]
        assert [m["_row_id"] for m in model_history[-2:]] == foreign_ids
        rows = db.get_messages_as_conversation("shared-wire-key", include_inactive=True)
        assert [(m["role"], m["content"]) for m in rows] == [
            ("user", "local question"), ("assistant", "same answer"), *foreign,
            ("user", "new question"), ("assistant", "new answer")]
        starts = [f["params"] for f in frames if f.get("params", {}).get("type") == "message.start"]
        assert len(starts) == 1
        assert starts[0]["payload"]["inputs"] == [{
            "id": reply["result"]["submission"]["input_id"], "ref": "current-row"}]
        assert all("submission_ref" not in m and "input_id" not in m for m in rows + model_history)
    finally:
        session["_run_thread"].join(10)
        db.close()


def test_plain_submit_projects_start_and_peer_corrections_before_completion(turn_env, monkeypatch):
    """A normal client's submit gives attached peers visible input before inference finishes."""
    inference = Inference()
    corrections = []
    inference.steer = lambda text: corrections.append(text) or True
    session = session_for(inference)
    monkeypatch.setitem(server._sessions, "wire", session)
    try:
        with endpoint() as url, connect(url) as socket_a, connect(url) as socket_b:
            a, b = Peer(socket_a), Peer(socket_b)
            for peer in (a, b):
                peer.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)
            a.rpc("prompt.submit", session_id="wire", text="plain starting input")
            assert inference.entered.wait(10)
            start = b.event("message.start")
            # This normal official RPC succeeds on the base; its missing input is the regression.
            assert start.get("payload", {}).get("input") == {"role": "user", "text": "plain starting input"}
            assert not inference.release.is_set()
            observed = []
            for _ in range(2):
                reply = a.rpc("session.steer", session_id="wire", text="same correction",
                              submission_ref="same-ref", input_visibility="visible")
                event = b.event("message.input")
                assert event["turn"] == start["turn"]
                assert event["payload"]["inputs"][0]["id"] == reply["submission"]["input_id"]
                observed.append(event["payload"])
            assert observed[0]["inputs"] != observed[1]["inputs"]
            assert corrections == ["same correction", "same correction"]
            resumed = b.rpc("session.resume", session_id="shared-wire-key", omit_messages=True)
            assert resumed["inflight"]["input_observations"] == observed
            assert not any(f.get("params", {}).get("type") == "message.complete" for f in b.seen)
            inference.release.set()
            complete = b.event("message.complete")
            assert complete["turn"] == start["turn"]
            assert {i["id"] for i in complete["payload"]["inputs"]} == {
                start["payload"]["inputs"][0]["id"], *(p["inputs"][0]["id"] for p in observed)}
    finally:
        inference.release.set()
        if session.get("_run_thread"):
            session["_run_thread"].join(10)