"""Only a directly authenticated Desktop submit can start request evidence."""
from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


class LoginSocket:
    auth_identity = {"provider": "basic", "user_id": "alice"}

    def write(self, _frame):
        return True


def test_authenticated_submit_passes_server_stamped_origin_to_turn(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a: False)
    origins = []

    class InlineThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    # The submit claim id and physical key now follow approval_origin.
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: origins.append(args[8]))
    token = bind_transport(LoginSocket())
    try:
        created = server.handle_request({"id": "c", "method": "session.create",
                                         "params": {"source": "desktop", "cwd": str(tmp_path)}})
        sid = created["result"]["session_id"]
        direct = server.handle_request({"id": "p", "method": "prompt.submit",
                                        "params": {"session_id": sid, "text": "Edit src/cli.py"}})
        assert direct["result"]["status"] == "streaming"
        requester, task_id, inflight, request, queued_generation = origins[-1]
        assert requester == "basic:alice" and task_id and inflight is server._sessions[sid]["inflight_turn"]
        assert request == "Edit src/cli.py"
        assert queued_generation is None
        server._sessions[sid]["running"] = False
        hidden = server.handle_request({"id": "h", "method": "prompt.submit",
                                        "params": {"session_id": sid, "text": "system wake", "display_kind": "hidden"}})
        assert hidden["result"]["status"] == "streaming"
        assert origins[-1] is None
    finally:
        reset_transport(token)
        if "sid" in locals():
            server._sessions.pop(sid, None)
        db.close()


def test_isolated_authenticated_submit_carries_only_exact_claimed_instruction(tmp_path, monkeypatch):
    from tui_gateway.compute_host import _isolated_chat_user_instruction

    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {"turn_isolation": True}})
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    frames = []

    class Supervisor:
        def submit_turn(self, frame, *, on_complete=None):
            frames.append(frame)
            return frame["request_id"]

    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda _cfg=None: Supervisor())
    token = bind_transport(LoginSocket())
    try:
        created = server.handle_request({"id": "c", "method": "session.create",
                                         "params": {"source": "desktop", "cwd": str(tmp_path)}})
        sid = created["result"]["session_id"]
        text = "Transfer ABC-1 to this chat"
        sent = server.handle_request({"id": "p", "method": "prompt.submit",
                                      "params": {"session_id": sid, "text": text}})
        assert sent["result"]["status"] == "streaming"
        frame = frames[-1]
        child = {"session_key": frame["session_key"], "profile_home": frame["profile_home"]}
        assert frame["_chat_user_instruction"]["requester"] == "basic:alice"
        assert _isolated_chat_user_instruction(frame, child, frame["request_id"], sid, text) == ("basic", "alice")
        assert _isolated_chat_user_instruction(frame, child, frame["request_id"], sid, "rewritten") is None
        assert _isolated_chat_user_instruction({**frame, "_chat_submit_ids": []}, child,
                                                frame["request_id"], sid, text) is None
        assert _isolated_chat_user_instruction({**frame, "source": "bot_room"}, child,
                                                frame["request_id"], sid, text) is None
        assert _isolated_chat_user_instruction(frame, {**child, "profile_home": "other-profile"},
                                                frame["request_id"], sid, text) is None
        refused = server.handle_request({"id": "d", "method": "prompt.submit",
                                         "params": {"session_id": sid, "text": text,
                                                    "_turn_author": {"id": "delegated"}}})
        assert refused["error"]["code"] == 4124
        assert len(frames) == 1
    finally:
        reset_transport(token)
        if "sid" in locals():
            server._sessions.pop(sid, None)
        db.close()
