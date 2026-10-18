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
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: origins.append(args[-1]))
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
