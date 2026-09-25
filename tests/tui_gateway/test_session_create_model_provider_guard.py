"""``session.create`` rejects a model the selected provider cannot serve (#96817).

Before the gate the session was minted and the FIRST turn died with the provider's 404 — a dead
chat. Custom / unknown providers and same-family or unlisted names stay permissive.
"""

import pytest


@pytest.fixture
def _create(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    (tmp_path / "config.yaml").write_text("model:\n  default: claude-opus-5\n  provider: anthropic\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_profile_home", lambda *a: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *a: None)
    return lambda params: (server._methods["session.create"]("r1", {"cols": 80, **params}), server._sessions)


@pytest.mark.parametrize("params", [
    {"model": "gpt-5.5", "provider": "anthropic"},
    {"model": "gpt-5.5"},  # provider implied by the profile config
    {"model": "deepseek/deepseek-v4-flash-0731", "provider": "openai-codex"},  # the incident pair
])
def test_session_create_rejects_incoherent_model_provider_pair_before_any_state(_create, params):
    response, sessions = _create(params)

    error = response["error"]
    assert error["code"] == -32602
    assert params["model"] in error["message"]
    assert error["data"]["provider"] in error["message"]
    assert error["data"]["model"] == params["model"]
    assert 1 <= len(error["data"]["suggestions"]) <= 5
    assert all(s in error["message"] for s in error["data"]["suggestions"])
    assert sessions == {}


@pytest.mark.parametrize("params", [
    {"model": "claude-opus-5", "provider": "anthropic"},  # coherent
    {"model": "claude-opus-5-20261001", "provider": "anthropic"},  # same family, not (yet) listed
    {"model": "gpt-5.5", "provider": "custom:local"},  # custom endpoint: Hermes cannot know its models
    {"model": "gpt-5.5", "provider": "openrouter"},  # aggregator
])
def test_session_create_keeps_coherent_unlisted_and_custom_pairs(_create, params):
    response, sessions = _create(params)

    assert "error" not in response, response
    session = sessions[response["result"]["session_id"]]
    assert session["model_override"] == {"model": params["model"], "provider": params["provider"]}
