import json
from types import SimpleNamespace

from hermes_cli.model_switch import ModelSwitchResult
from model_tools import handle_function_call
from tools.switch_model_tool import switch_model_for_agent, switch_model_tool


def test_switch_model_tool_requires_agent_loop():
    result = json.loads(switch_model_tool("gpt-5.4"))

    assert result["success"] is False
    assert "agent loop" in result["error"]


def test_handle_function_call_does_not_dispatch_switch_model_directly():
    result = json.loads(handle_function_call("switch_model", {"new_model": "gpt-5.4"}))

    assert "error" in result
    assert "must be handled by the agent loop" in result["error"]


def test_switch_model_for_agent_resolves_and_applies_session_switch(monkeypatch):
    calls = {}

    def fake_load_config():
        return {"providers": {"custom": {"base_url": "https://example.invalid"}}}

    def fake_get_compatible_custom_providers(cfg):
        calls["custom_cfg"] = cfg
        return [{"id": "custom"}]

    def fake_switch_model(**kwargs):
        calls["switch_model"] = kwargs
        return ModelSwitchResult(
            success=True,
            new_model="provider/gpt-5.4",
            target_provider="openrouter",
            api_key="sk-new",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            provider_label="OpenRouter",
        )

    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
    monkeypatch.setattr(
        "hermes_cli.config.get_compatible_custom_providers",
        fake_get_compatible_custom_providers,
    )
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", fake_switch_model)

    applied = {}

    def apply_switch(**kwargs):
        applied.update(kwargs)

    agent = SimpleNamespace(
        model="old-model",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-old",
        switch_model=apply_switch,
    )

    result = json.loads(
        switch_model_for_agent(
            agent,
            "gpt-5.4",
            reason="need stronger reasoning",
            provider="openrouter",
        )
    )

    assert result == {
        "success": True,
        "old_model": "old-model",
        "new_model": "provider/gpt-5.4",
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "reason": "need stronger reasoning",
        "warning": "",
    }
    assert calls["switch_model"]["raw_input"] == "gpt-5.4"
    assert calls["switch_model"]["current_model"] == "old-model"
    assert calls["switch_model"]["current_api_key"] == "sk-old"
    assert calls["switch_model"]["explicit_provider"] == "openrouter"
    assert calls["switch_model"]["user_providers"] == {
        "custom": {"base_url": "https://example.invalid"}
    }
    assert calls["switch_model"]["custom_providers"] == [{"id": "custom"}]
    assert applied == {
        "new_model": "provider/gpt-5.4",
        "new_provider": "openrouter",
        "api_key": "sk-new",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }


def test_switch_model_for_agent_returns_resolution_error(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda _cfg: [])
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: ModelSwitchResult(success=False, error_message="no such model"),
    )

    agent = SimpleNamespace(
        model="old-model",
        provider="openrouter",
        base_url="",
        api_key="sk-old",
        switch_model=lambda **_kwargs: None,
    )

    result = json.loads(switch_model_for_agent(agent, "missing-model"))

    assert result == {"success": False, "error": "no such model"}
