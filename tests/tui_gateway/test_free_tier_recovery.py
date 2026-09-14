"""TUI admission keeps queued input and login runs on the live event channel."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock


def test_capped_queue_is_retained_and_notice_clears_on_route_recovery(tmp_path, monkeypatch):
    from hermes_cli import auth, free_tier_usage as usage
    from tui_gateway import server
    from hermes_constants import get_hermes_home
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    auth._save_active_provider_state("nous", {"auth_method": "anonymous", "anon_token": "anon_queue_fixture"})
    for _ in range(10):
        usage.record_completed_tool(usage.current_identity())
    agent = SimpleNamespace(base_url="https://welcome-api.nousresearch.com/v1")
    queued = {"text": "after", "image_paths": ["kept.png"]}
    session = {"agent": agent, "history_lock": threading.RLock(), "queued_prompt": queued,
               "running": False, "profile_home": str(get_hermes_home())}
    events = []
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))
    assert server._drain_queued_prompt("r", "s", session)
    assert session["queued_prompt"] is queued and not session["running"]
    assert events[-1][0] == "notification.show" and events[-1][2]["key"] == "free_tier.limit"
    agent.base_url = "http://localhost:11434/v1"
    server._sync_free_tier_notice("s", session)
    assert events[-1] == ("notification.clear", "s", {"key": "free_tier.limit"})


def test_capped_busy_submit_queues_without_interrupting_active_turn(tmp_path, monkeypatch):
    from hermes_cli import auth, free_tier_usage as usage
    from tui_gateway import server
    from hermes_constants import get_hermes_home
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    auth._save_active_provider_state("nous", {"auth_method": "anonymous", "anon_token": "anon_busy_fixture"})
    for _ in range(10):
        usage.record_completed_tool(usage.current_identity())
    agent = SimpleNamespace(base_url="https://welcome-api.nousresearch.com/v1", interrupt=Mock(),
                            redirect=Mock(), _supports_active_turn_redirect=True)
    inflight = {"user": "active", "assistant": "still streaming"}
    history = [{"role": "user", "content": "active"}]
    session = {"agent": agent, "history_lock": threading.RLock(), "history": history,
               "running": True, "inflight_turn": inflight, "attached_images": ["kept.png"],
               "profile_home": str(get_hermes_home())}
    events = []
    monkeypatch.setattr(server, "_sess_nowait", lambda *a: (session, None))
    monkeypatch.setattr(server, "_emit", lambda *a: events.append(a))
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    monkeypatch.setitem(server._sessions, "s", session)
    for queued in (False, True):
        result = server.handle_request({"id": "send", "method": "prompt.submit", "params": {
            "session_id": "s", "text": "later", "queued": queued}})
        assert result["result"]["status"] == "queued"
    assert session["running"] and session["inflight_turn"] is inflight
    assert session["history"] is history
    assert session["queued_prompt"]["text"] == "later"
    assert session["queued_prompt"]["image_paths"] == ["kept.png"]
    assert session["queued_prompts"][0]["text"] == "later"
    assert not events
    agent.interrupt.assert_not_called()
    agent.redirect.assert_not_called()


def test_queue_envelope_survives_quota_crossing_after_pop(tmp_path, monkeypatch):
    from hermes_cli import auth, free_tier_usage as usage
    from run_agent import AIAgent
    from tui_gateway import server
    from hermes_constants import get_hermes_home
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    auth._save_active_provider_state("nous", {"auth_method": "anonymous", "anon_token": "anon_race_fixture"})
    identity = usage.current_identity()
    assert identity is not None
    for _ in range(9):
        usage.record_completed_tool(identity)
    agent = AIAgent(api_key="fixture", provider="nous", model="nous/welcome",
                    base_url="https://welcome-api.nousresearch.com/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    monkeypatch.setattr(agent, "_interruptible_api_call", Mock(side_effect=AssertionError("refused turn must not infer")))
    queued = {"text": "kept request", "image_paths": ["kept.png"], "transport": None,
              "turn_author": {"id": "bot:coder", "name": "coder", "is_bot": True}}
    later = {"text": "later", "transport": None}
    history = [{"role": "user", "content": "before"}, {"role": "assistant", "content": "done"}]
    session = {"agent": agent, "history_lock": threading.RLock(), "queued_prompt": queued,
               "queued_prompts": [later], "running": False, "history": history,
               "session_key": agent.session_id, "profile_home": str(get_hermes_home())}
    events = []
    monkeypatch.setattr(server, "_emit", lambda *a: events.append(a))
    def cross_after_pop(*args):
        usage.record_completed_tool(identity)
        return False
    monkeypatch.setattr(server, "_session_uses_compute_host", cross_after_pop)
    def prepare(sid, session, st, text, images):
        from hermes_constants import set_hermes_home_override
        st.scopes.home = set_hermes_home_override(session["profile_home"])
        st.history = list(session["history"])
        return text, text, 80, None
    monkeypatch.setattr(server, "_prepare_turn_input", prepare)
    for name in ("_ensure_active_session_slot", "_sync_session_key_after_compress",
                 "_publish_session_control_snapshot", "_emit_settled_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **kw: None)
    assert server._drain_queued_prompt("r", "s", session)
    session["_run_thread"].join(timeout=5)
    assert not session["_run_thread"].is_alive()
    assert not session["running"] and session["history"] is history
    assert any(e[0] == "message.complete" and e[2].get("code") == usage.LIMIT_REASON for e in events)
    assert session["queued_prompt"] is queued
    assert session["queued_prompts"] == [later]


def test_login_streams_full_code_and_completion_to_session_owner(tmp_path, monkeypatch):
    from hermes_cli import anon_auth
    from tui_gateway import server
    from tui_gateway.transport import bind_transport, reset_transport
    events = []
    owner = SimpleNamespace(write=lambda obj: events.append(obj) or True)
    caller = Mock()
    session = {"agent": SimpleNamespace(base_url="http://localhost:11434/v1"),
               "profile_home": str(tmp_path), "transport": owner}
    monkeypatch.setattr(server, "_sessions", {"s": session})
    monkeypatch.setattr(server, "_sess_nowait", lambda *a: (session, None))
    code = anon_auth.Code(link="https://example.invalid/consent/" + "long-link-" * 30,
                          code="TEST-CODE", expires_in=600, interval=5)
    monkeypatch.setattr(anon_auth, "run_sign_in", lambda **kw: iter([code, anon_auth.Completed()]))
    token = bind_transport(caller)
    try:
        result = server.handle_request({"id": "login", "method": "slash.exec", "params": {
            "session_id": "s", "command": "/login"}})
        assert "output" in result["result"]
        session["_free_tier_login_thread"].join(timeout=5)
        assert not session["_free_tier_login_thread"].is_alive()
    finally:
        reset_transport(token)
    payloads = [e["params"]["payload"] for e in events if e["params"]["type"] == "notification.show"]
    assert payloads[0]["text"] == f"{code.link}\n{code.code}\n{code.copy_with_wait}"
    assert all(p["key"] == "free_tier.login" for p in payloads)
    assert "Signed in" in payloads[-1]["text"]
    assert all(e["params"]["session_id"] == "s" for e in events)
    caller.write.assert_not_called()
    assert session.get("slash_worker") is None
