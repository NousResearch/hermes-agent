"""Repro: the durable per-session override must only be stamped for a real
session-scoped pick, and the session-aware model read must survive every shape
config.yaml can hand back.

These are deliberately written against the real ``config.set`` / ``config.get``
handlers, not the helper in isolation.
"""
import threading
import types

from tui_gateway import server

GUARDED_MODEL = "muse-spark-1.2-contributor"


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


def _config_set_model(value, **extra_params):
    params = {"session_id": "sid", "key": "model", "value": value}
    params.update(extra_params)
    return server.handle_request({"id": "1", "method": "config.set", "params": params})


def _patch_switch(monkeypatch, scope):
    monkeypatch.setattr(server, "_apply_model_switch", lambda *a, **k: {
        "value": "anthropic/claude-sonnet-4.6", "warning": "",
        "confirm_required": False, "scope": scope})
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *a, **k: None)


def _records_persist(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_persist_session_row_override",
                        lambda *a, **k: calls.append((a, k)) or True)
    return calls


def test_global_switch_does_not_stamp_a_session_override(monkeypatch):
    _patch_switch(monkeypatch, "global")
    calls = _records_persist(monkeypatch)
    server._sessions["sid"] = _session()
    try:
        resp = _config_set_model("anthropic/claude-sonnet-4.6 --global")
        assert not resp.get("error"), resp
        assert calls == [], "--global must not pin the session row"
    finally:
        server._sessions.pop("sid", None)


def test_once_switch_does_not_stamp_a_session_override(monkeypatch):
    _patch_switch(monkeypatch, "once")
    calls = _records_persist(monkeypatch)
    server._sessions["sid"] = _session()
    try:
        resp = _config_set_model("anthropic/claude-sonnet-4.6 --once")
        assert not resp.get("error"), resp
        assert calls == [], "--once must not pin the session row"
    finally:
        server._sessions.pop("sid", None)


def test_unconfirmed_midturn_pick_is_not_persisted(monkeypatch):
    calls = _records_persist(monkeypatch)
    server._sessions["sid"] = _session(running=True)
    try:
        resp = _config_set_model(GUARDED_MODEL)
        assert resp["result"]["confirm_required"] is True, resp
        assert "pending_model_switch" not in server._sessions["sid"]
        assert calls == [], "a pick that was never queued must not be persisted"
    finally:
        server._sessions.pop("sid", None)


def test_config_get_model_survives_a_string_model_config(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": "some/vendor-model"})
    out = server._CONFIG_GETTERS["model"]({"session_id": ""})
    assert isinstance(out, dict), out
    assert out["model"] == "some/vendor-model"


def test_config_get_model_reports_session_model_during_voice_route(monkeypatch):
    agent = types.SimpleNamespace(
        model="voice-model", provider="voice-provider",
        _voice_route_state={
            "snapshot": {"model": "chat-model", "provider": "chat-provider"},
            "reasoning_config": {"enabled": True, "effort": "high"},
        },
    )
    monkeypatch.setitem(server._sessions, "voice-sid", _session(agent=agent))
    result = server._CONFIG_GETTERS["model"]({"session_id": "voice-sid"})
    assert result == {"model": "chat-model", "provider": "chat-provider", "scope": "session"}


def test_config_get_reasoning_preserves_session_effort_during_voice_route(monkeypatch):
    agent = types.SimpleNamespace(
        reasoning_config={"enabled": False},
        _voice_route_state={
            "snapshot": {"model": "chat-model", "provider": "chat-provider"},
            "reasoning_config": {"enabled": True, "effort": "high"},
        },
    )
    monkeypatch.setitem(server._sessions, "voice-sid", _session(agent=agent))
    monkeypatch.setattr(server, "_load_cfg", dict)
    result = server._CONFIG_GETTERS["reasoning"]({"session_id": "voice-sid"})
    assert result["value"] == "high"


def test_non_live_session_pick_is_refused_before_any_persist(monkeypatch):
    # Current upstream binds scoped RPCs to a live session; a detached row
    # must be resumed before changing it, rather than borrowing launch scope.
    monkeypatch.delitem(server._sessions, "sid", raising=False)
    writes = []
    monkeypatch.setattr(server, "_persist_session_row_override", lambda *a, **k: writes.append("row"))
    monkeypatch.setattr(server, "_save_cfg", lambda *a, **k: writes.append("profile"))
    response = _config_set_model("vendor/model --session")
    assert response.get("error", {}).get("code") == 4001, response
    assert writes == []
