"""Queued prompts are addressable: stable ids, ``session.queue.get`` / ``session.queue.update``.

A prompt accepted while a turn runs used to be reachable only as the legacy head-only ``queued``
view on resume. A client could not list the follow-ups it queued after a reconnect, edit or
delete one, or promote one into the running turn. These tests drive the real handlers against a
real ``SessionDB`` (the accept-time durable row must follow every edit and delete).
"""

import types

from hermes_state import SessionDB
from tui_gateway import server


def _desktop_session(monkeypatch, db):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    resp = server.handle_request({"id": "c", "method": "session.create", "params": {"cols": 96, "source": "desktop"}})
    assert "result" in resp, resp
    sid = resp["result"]["session_id"]
    server._sessions[sid]["agent"] = types.SimpleNamespace()
    return sid, resp["result"]["stored_session_id"]


def _busy(session, in_flight="prompt A"):
    with session["history_lock"]:
        session["running"] = True
        server._start_inflight_turn(session, in_flight)


def _queue(sid, session, text, queue_id="", client_message_id=""):
    return server._handle_busy_submit("r", sid, session, text, "ws-1", queued=True, display_kind=None,
                                      queue_id=queue_id, client_message_id=client_message_id)


def _rpc(method, **params):
    return server.handle_request({"id": "q", "method": method, "params": params})


def _active_user_texts(db, key):
    return [r["content"] for r in db.get_messages_as_conversation(key, include_row_ids=True) if r["role"] == "user"]


class _Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.db = SessionDB(db_path=tmp_path / "state.db")
        self.sid, self.key = _desktop_session(monkeypatch, self.db)
        self.session = server._sessions[self.sid]
        self.events: list[tuple[str, str, dict]] = []
        self.global_events: list[tuple[str, dict]] = []
        monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: self.events.append((event, sid, payload)))
        monkeypatch.setattr(server, "_broadcast_global_event",
                            lambda event, payload=None: self.global_events.append((event, payload)))
        _busy(self.session)

    def queue_events(self):
        return [payload for event, _sid, payload in self.events if event == "session.queue"]

    def close(self):
        server._sessions.pop(self.sid, None)
        self.db.close()


def test_client_queue_ids_are_stable_separate_and_idempotent(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    try:
        first = _queue(h.sid, h.session, "one", queue_id="item-1", client_message_id="bubble-1")["result"]
        assert first["status"] == "queued"
        assert [(i["id"], i["client_message_id"], i["text"]) for i in first["queue"]["items"]] == [
            ("item-1", "bubble-1", "one")]
        # A second client item must not merge into the first (its id would be lost).
        second = _queue(h.sid, h.session, "two", queue_id="item-2", client_message_id="bubble-2")["result"]
        assert [i["id"] for i in second["queue"]["items"]] == ["item-1", "item-2"]
        # A retry after a dropped socket resubmits the same id: acknowledged, not queued twice.
        retry = _queue(h.sid, h.session, "two", queue_id="item-2", client_message_id="bubble-2")["result"]
        assert retry["queue"] == second["queue"]
        assert _active_user_texts(h.db, h.key).count("two") == 1

        listed = _rpc("session.queue.get", session_id=h.sid)["result"]
        assert listed == second["queue"]
        assert listed["items"][0] == {"id": "item-1", "client_message_id": "bubble-1", "text": "one",
                                      "has_images": False, "editable": True, "steerable": False}
        # Every visible change is published once, with an advancing revision.
        revisions = [p["revision"] for p in h.queue_events()]
        assert revisions == sorted(set(revisions)) and len(revisions) == 2
        assert ("session.queue.changed", {"session_key": h.key, "revision": revisions[-1]}) in h.global_events
    finally:
        h.close()


def test_idless_prompts_keep_merging_but_get_a_server_id(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    try:
        _queue(h.sid, h.session, "first")
        merged = _queue(h.sid, h.session, "second")["result"]["queue"]["items"]
        assert len(merged) == 1 and merged[0]["text"] == "first\n\nsecond" and merged[0]["id"]
        # An id-less arrival never merges into a client-addressed item.
        _queue(h.sid, h.session, "pinned", queue_id="pinned-1")
        _queue(h.sid, h.session, "third")
        items = _rpc("session.queue.get", session_id=h.sid)["result"]["items"]
        assert [i["text"] for i in items] == ["first\n\nsecond", "pinned", "third"]
    finally:
        h.close()


def test_edit_rewrites_the_queued_text_and_its_durable_row(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    try:
        _queue(h.sid, h.session, "draft", queue_id="item-1")
        resp = _rpc("session.queue.update", session_id=h.sid, queue_id="item-1", action="edit", text="  final  ")
        assert resp["result"]["status"] == "edit"
        assert [i["text"] for i in resp["result"]["items"]] == ["final"]
        assert "final" in _active_user_texts(h.db, h.key) and "draft" not in _active_user_texts(h.db, h.key)

        ran = []
        monkeypatch.setattr(server, "_run_prompt_submit", lambda rid, s, sess, text, **kw: ran.append(text))
        with h.session["history_lock"]:
            h.session["running"] = False
            server._clear_inflight_turn(h.session)
        assert server._drain_queued_prompt("d", h.sid, h.session) is True
        assert ran == ["final"]
        assert h.queue_events()[-1]["items"] == []
    finally:
        h.close()


def test_delete_drops_the_item_and_retires_its_row(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    try:
        _queue(h.sid, h.session, "keep", queue_id="keep")
        _queue(h.sid, h.session, "cancel me", queue_id="drop")
        resp = _rpc("session.queue.update", session_id=h.sid, queue_id="drop", action="delete")
        assert [i["id"] for i in resp["result"]["items"]] == ["keep"]
        assert "cancel me" not in _active_user_texts(h.db, h.key)
        every = h.db.get_messages_as_conversation(h.key, include_inactive=True, include_row_ids=True)
        assert any(r["content"] == "cancel me" for r in every)  # kept on disk, inactive
        missing = _rpc("session.queue.update", session_id=h.sid, queue_id="drop", action="delete")
        assert missing["error"]["code"] == 4041
    finally:
        h.close()


def test_steer_promotes_the_item_into_the_running_turn(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    steered = []
    h.session["agent"] = types.SimpleNamespace(steer=lambda text: steered.append(text) or True)
    try:
        _queue(h.sid, h.session, "use the other file", queue_id="item-1")
        assert _rpc("session.queue.get", session_id=h.sid)["result"]["items"][0]["steerable"] is True
        resp = _rpc("session.queue.update", session_id=h.sid, queue_id="item-1", action="steer")
        assert resp["result"] == {"status": "steer", "revision": resp["result"]["revision"], "items": []}
        assert steered == ["use the other file"]
        assert "use the other file" in (h.session["inflight_turn"].get("corrections") or [])
        assert "use the other file" not in _active_user_texts(h.db, h.key)
    finally:
        h.close()


def test_rejected_steer_puts_the_item_back_in_place(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    h.session["agent"] = types.SimpleNamespace(steer=lambda text: False)
    try:
        for queue_id in ("a", "b", "c"):
            _queue(h.sid, h.session, f"text {queue_id}", queue_id=queue_id)
        resp = _rpc("session.queue.update", session_id=h.sid, queue_id="b", action="steer")
        assert resp["error"]["code"] == 4009
        assert [i["id"] for i in _rpc("session.queue.get", session_id=h.sid)["result"]["items"]] == ["a", "b", "c"]
    finally:
        h.close()


def test_attachment_and_foreign_items_are_not_editable(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    try:
        with h.session["history_lock"]:
            server._enqueue_prompt(h.session, "see image", None, image_paths=["/tmp/x.png"], queue_id="img")
            server._enqueue_prompt(h.session, "from a bot", None, turn_author={"name": "bot"}, queue_id="bot")
        items = {i["id"]: i for i in _rpc("session.queue.get", session_id=h.sid)["result"]["items"]}
        assert items["img"]["has_images"] and not items["img"]["editable"]
        assert not items["bot"]["editable"] and not items["bot"]["steerable"]
        for queue_id in ("img", "bot"):
            err = _rpc("session.queue.update", session_id=h.sid, queue_id=queue_id, action="edit", text="x")
            assert err["error"]["code"] == 4004
    finally:
        h.close()


def test_stop_clears_and_publishes_the_queue_and_resume_carries_it(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    try:
        _queue(h.sid, h.session, "later", queue_id="item-1")
        live = server._live_session_payload(h.sid, h.session, omit_messages=True)
        assert [i["id"] for i in live["queue"]["items"]] == ["item-1"]
        monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a, **_k: False)
        server._interrupt_session_turn(h.sid, h.session)
        assert h.queue_events()[-1]["items"] == []
    finally:
        h.close()


def test_isolated_host_queue_events_do_not_reach_clients(monkeypatch):
    """The child process holds only its own leftovers under an independent revision counter; the
    parent owns the client's queue, so the child's snapshot must never be relayed."""
    forwarded = []
    monkeypatch.setattr(server, "write_json", lambda frame: forwarded.append(frame) or True)
    monkeypatch.setattr(server, "_broadcast_global_event", lambda *a, **k: forwarded.append(a))
    child_queue = {"jsonrpc": "2.0", "method": "event", "params": {
        "type": "session.queue", "session_id": "sid", "payload": {"revision": 99, "items": []}}}
    child_signal = {"jsonrpc": "2.0", "method": "event", "params": {
        "type": "session.queue.changed", "session_id": "", "payload": {"session_key": "k", "revision": 99}}}
    assert server._relay_compute_host_rpc(child_queue) is True
    assert server._relay_compute_host_rpc(child_signal) is True
    assert forwarded == []
