"""Voice detours must not erase the text session's temporary-fallback provenance.

Real voice binding, runtime serialization, SQLite rows and resume construction;
only provider clients/resolution are inert, and the shared fixture forbids network.
"""
import json
from unittest.mock import MagicMock

import pytest

from agent.voice_turn_route import begin_voice_turn_route, end_voice_turn_route
from tests.tui_gateway.test_fallback_recovery_boundaries import (
    FALLBACK, PRIMARY, _build, _request_fallback, runtime_env,
)
from tui_gateway import server


@pytest.mark.parametrize("origin", ["construction", "request", "primary"])
@pytest.mark.parametrize("separate_voice_model", [False, True])
def test_voice_checkpoint_preserves_text_recovery(monkeypatch, runtime_env, origin, separate_voice_model):
    clock, db, resolved, home = runtime_env
    clock["auth_failed"] = origin == "construction"
    agent = _build(home, db, "sid", "key")
    if origin == "request":
        _request_fallback(monkeypatch, agent, home)
    clock["auth_failed"] = False
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    agent.service_tier = "fast"
    db.create_session("key", source="desktop", model=agent.model)
    session = {"agent": agent, "session_key": "key", "profile_home": home}
    cfg = {"reasoning_effort": "none"}
    if separate_voice_model:
        cfg.update(provider="openrouter", model="voice-model", api_mode="chat_completions")
    monkeypatch.setattr("agent.auxiliary_task_config._get_auxiliary_task_config", lambda _: cfg)
    client = MagicMock(base_url="https://voice.invalid", api_key="test-not-a-secret")
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **k: (client, "voice-model"))

    with server._profile_build_scope(home):
        before = server._runtime_model_config(agent)
        agent._voice_turn_pending = True
        begin_voice_turn_route(agent, [{"role": "user", "content": "hello"}], "system")
        try:
            assert agent.model == ("voice-model" if separate_voice_model else before["model"])
            # The request-time Codex fallback has mandatory thinking; its voice
            # effort clamps to low rather than disabling upstream's reasoning floor.
            expected_effort = ({"enabled": True, "effort": "low"}
                               if origin == "request" and not separate_voice_model
                               else {"enabled": False})
            assert agent.reasoning_config == expected_effort
            server._persist_live_session_runtime(session)
            row = db.get_session("key")
            during = json.loads(row["model_config"])
            # Equality includes the fallback identity, historical intent and deadline,
            # not merely the displayed model. No temporary voice route is persisted.
            assert during == before
            assert row["model"] == before["model"]
            assert "test-not-a-secret" not in row["model_config"]
        finally:
            end_voice_turn_route(agent)
        assert server._runtime_model_config(agent) == before
        server._persist_live_session_runtime(session)
        assert json.loads(db.get_session("key")["model_config"]) == before

    resumed = _build(home, db, "resumed", "key", **server._stored_session_runtime_overrides(row))
    assert resumed.model == (PRIMARY if origin == "primary" else FALLBACK)
    assert resumed.reasoning_config == before["reasoning_config"]
    assert resumed.service_tier == "fast"
    with server._profile_build_scope(home):
        assert server._runtime_model_config(resumed) == before
    # Resume the mid-voice checkpoint after cooldown: recover the original primary,
    # not the voice model and not a permanently promoted fallback.
    clock.update(wall=1200.0, mono=300.0)
    recovered = _build(home, db, "recovered", "key", **server._stored_session_runtime_overrides(row))
    assert (recovered.model, recovered.provider) == (PRIMARY, "anthropic")
    with server._profile_build_scope(home):
        assert "fallback_recovery" not in server._runtime_model_config(recovered)
