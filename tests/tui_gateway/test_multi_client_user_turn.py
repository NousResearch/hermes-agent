"""Canonical user turns reach every attached client without a second transcript path."""
import threading

import pytest

from hermes_state import SessionDB
from tui_gateway import server
from tests.tui_gateway.test_queued_prompt_persistence import _desktop_session, _busy
from tests.tui_gateway.test_shared_session_delivery import Peer


@pytest.mark.parametrize("busy", [False, True])
def test_submitted_user_row_reaches_all_attached_clients_before_its_reply(tmp_path, monkeypatch, busy):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        sid, key = _desktop_session(monkeypatch, db)
        session = server._sessions[sid]
        class TranscriptPeer(Peer):
            def __init__(self):
                super().__init__()
                self.complete = threading.Event()

            def write(self, frame):
                result = super().write(frame)
                if frame.get("params", {}).get("type") == "message.complete":
                    self.complete.set()
                return result

        first, sender = TranscriptPeer(), TranscriptPeer()
        session["transport"] = first
        server._attach_session_transport(session, sender)
        monkeypatch.setattr(server, "_typed_stop_phrase_response", lambda *a: None)
        monkeypatch.setattr(server, "_start_agent_build", lambda *a: None)
        monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *a: True)
        done = threading.Event()

        def run(_rid, _sid, _session, text, *args, **kwargs):
            server._emit("message.start", sid)
            server._emit("message.complete", sid, {"text": "reply to " + text})
            done.set()

        monkeypatch.setattr(server, "_run_after_agent_ready", run)
        monkeypatch.setattr(server, "_run_prompt_submit", run)
        if busy:
            _busy(session)
        response = server.handle_request({"id": "external", "method": "prompt.submit", "params": {
            "session_id": sid, "text": "follow up", "queued": True}})
        assert "error" not in response, response
        if busy:
            assert not done.is_set()
            accepted = session["queued_prompt"]["_submit_user_row"].copy()
            db.append_message(key, "assistant", content="old reply")
            with session["history_lock"]:
                session["running"] = False
                server._clear_inflight_turn(session)
            server._drain_queued_prompt("drain", sid, session)
        assert done.wait(5)
        assert first.complete.wait(5)
        assert sender.complete.wait(5)
        assert first.frames == sender.frames
        frames = [frame["params"] for frame in first.frames]
        users = [f for f in frames if f["type"] == "message.user"]
        assert len(users) == 1
        payload = users[0]["payload"]
        assert payload["client_message_ids"] == []
        row = [m for m in db.get_messages(key) if m["role"] == "user"][0]
        assert payload["message"]["row_id"] == row["id"]
        assert payload["message"]["message_uid"] == row["message_uid"]
        assert payload["message"]["text"] == row["content"]
        if busy:
            assert row["id"] != accepted["_row_id"]
            assert row["message_uid"] == accepted["message_uid"]
        assert [f["type"] for f in frames][-3:] == ["message.user", "message.start", "message.complete"]
        replay = server.handle_request({"id": "replay", "method": "session.events.since", "params": {
            "session_id": sid, "last_seen": 0}})
        assert "error" not in replay, replay
        assert any(e["type"] == "message.user" for e in replay["result"]["events"])
        session["running"] = False
        server._release_active_session_slot(session)
        server._sessions.pop(sid, None)


def test_merged_queue_keeps_correlations_and_hidden_rows_stay_hidden(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        sid, key = _desktop_session(monkeypatch, db)
        session = server._sessions[sid]
        _busy(session)
        for text, client_id in [("one", "c1"), ("two", "c2")]:
            server._handle_busy_submit("q", sid, session, text, None, queued=True, client_message_id=client_id)
        queued = session["queued_prompt"]
        original_uid = queued["_submit_user_row"]["message_uid"]
        server._replace_queued_user_row_for_turn(session, queued, is_dispatching=True)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
        server._emit_submit_user_row(sid, session)
        assert emitted[0][2]["client_message_ids"] == ["c1", "c2"]
        assert emitted[0][2]["message"]["text"] == "one\n\ntwo"
        assert emitted[0][2]["message"]["message_uid"] == original_uid
        server._persist_submit_user_row(session, "private", "hidden")
        server._emit_submit_user_row(sid, session)
        assert len(emitted) == 1
        session["running"] = False
        server._sessions.pop(sid, None)


def test_compute_worker_projects_the_row_its_turn_adopts(tmp_path, monkeypatch):
    import io
    import json
    from tui_gateway.compute_host import ComputeHost
    from tests.tui_gateway.test_queued_prompt_persistence import _run_turn

    with SessionDB(db_path=tmp_path / "state.db") as db:
        sid, key = _desktop_session(monkeypatch, db)
        session = server._sessions[sid]
        session["transport"] = None
        monkeypatch.setattr(server, "_session_info", lambda *args: {})

        def run(_rid, _sid, current, text, **kwargs):
            server._emit("message.start", sid)
            _run_turn(current, db, key, text, "worker reply")
            server._emit("message.complete", sid, {"text": "worker reply"})
            current["running"] = False

        monkeypatch.setattr(server, "_run_prompt_submit", run)
        output = io.StringIO()
        host = ComputeHost(stdout=output, heartbeat_secs=0)
        try:
            host._run_real_turn({"sid": sid, "request_id": "worker", "text": "worker input",
                                 "display_metadata": {"client_message_ids": ["optimistic"]}})
            frames = [json.loads(line) for line in output.getvalue().splitlines()]
            assert frames[-1]["type"] == "turn.end", frames
            events = [f["message"]["params"] for f in frames if f["type"] == "rpc"]
            users = [e for e in events if e["type"] == "message.user"]
            assert len(users) == 1
            rows = db.get_messages(key)
            assert [row["role"] for row in rows] == ["user", "assistant"]
            assert users[0]["payload"]["message"]["row_id"] == rows[0]["id"]
            assert users[0]["payload"]["message"]["message_uid"] == rows[0]["message_uid"]
            assert users[0]["payload"]["client_message_ids"] == ["optimistic"]
        finally:
            host.close()
            server._sessions.pop(sid, None)
