"""Fast pins cross the turn frame and apply after model sync, not during frame adoption."""

import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.fast_mode import begin_turn, effective_request_overrides
from tui_gateway import server
from tui_gateway.compute_host import ComputeHost


class _PreparedFast(Exception):
    pass


def test_fast_toggle_crosses_turn_frame_and_updates_reused_agent(monkeypatch):
    parent = {
        "agent": None, "session_key": "fast-key", "_compute_host_active": True,
        "_metadata_mirror": {"model": "gpt-6.1-sol", "provider": "openai-codex", "service_tier": ""},
        "history": [], "history_version": 0, "history_lock": threading.Lock(),
    }
    agent = SimpleNamespace(
        model="gpt-6.1-sol", provider="openai-codex", api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex", service_tier=None,
        request_overrides={"extra_body": {"keep": True}}, session_id="fast-key",
    )
    child = {"agent": agent, "session_key": "fast-key"}
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0, max_workers=1)
    monkeypatch.setattr(server, "_profile_runtime_scope_tokens", lambda home, **kwargs: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: [])
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    for name in ("_sync_agent_model_with_config", "_sync_agent_compression_with_config",
                 "_sync_agent_fallback_with_config", "_sync_bot_capabilities"):
        monkeypatch.setattr(server, name, lambda sid, session: None)

    def stop_after_fast(session):
        raise _PreparedFast()

    monkeypatch.setattr(server, "_adopt_out_of_band_turns", stop_after_fast)

    def prepare():
        from tools.approval_context import reset_current_session_key

        st = server._TurnRun(agent=agent, one_turn_restore=None, terminal_callback=None, receipt_committed=False)
        try:
            with pytest.raises(_PreparedFast):
                server._prepare_turn_input("fast-sid", child, st, "hello", [])
        finally:
            if st.scopes.approval is not None:
                reset_current_session_key(st.scopes.approval)

    try:
        for value, tier, expected in (
            ("on", "priority", {"service_tier": "priority"}),
            ("off", None, {}),
            ("auto", "auto", {"service_tier": "priority"}),
            ("cold", "cold", {"service_tier": "priority"}),
        ):
            with patch.dict(server._sessions, {"fast-sid": parent}, clear=True):
                response = server.handle_request({
                    "id": "edit", "method": "config.set",
                    "params": {"session_id": "fast-sid", "key": "fast", "value": value},
                })
                assert "error" not in response, response
                frame = json.loads(json.dumps(server._compute_host_turn_frame("turn", "fast-sid", parent, "hello")))
            with patch.dict(server._sessions, {"fast-sid": child}, clear=True), \
                    patch.object(server, "_persist_live_session_runtime"), \
                    patch.object(server, "_emit"):
                previous_tier = agent.service_tier
                adopted = host._ensure_server_session(server, frame)
                assert adopted["agent"] is agent
                assert agent.service_tier == previous_tier  # adoption must not mutate an in-flight request
                prepare()
                assert agent.service_tier == tier
                begin_turn(agent, [])
                assert effective_request_overrides(agent) == {"extra_body": {"keep": True}, **expected}
                assert adopted["create_service_tier_override"] == (tier or "")

        # Derive the speed parameter only after the queued switch changes the live route.
        def switch(sid, session, raw, **kwargs):
            agent.model, agent.provider = "claude-opus-5", "anthropic"
            agent.api_mode, agent.base_url = "anthropic_messages", ""
            agent._anthropic_base_url = "https://api.anthropic.com"
            return {"value": agent.model}

        monkeypatch.setattr(server, "_apply_model_switch", switch)
        with patch.dict(server._sessions, {"fast-sid": child}, clear=True), \
                patch.object(server, "_persist_live_session_runtime"), patch.object(server, "_emit"):
            host._ensure_server_session(server, {
                "sid": "fast-sid", "service_tier_override": "priority",
                "pending_model_switch": {"raw": "claude-opus-5"},
            })
            prepare()
            assert effective_request_overrides(agent) == {"extra_body": {"keep": True}, "speed": "fast"}
            host._ensure_server_session(server, {"sid": "fast-sid", "service_tier_override": None})
            prepare()
            assert agent.service_tier == "priority"  # None is not an explicit normal pin
            # A blocked/proxy route must not receive a carried-over fast parameter.
            agent._anthropic_base_url = "https://proxy.example/v1"
            prepare()
            assert effective_request_overrides(agent) == {"extra_body": {"keep": True}}
    finally:
        host._closed.set()
        host._executor.shutdown(wait=True)
