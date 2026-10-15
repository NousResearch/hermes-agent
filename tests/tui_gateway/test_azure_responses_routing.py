"""Resuming an Azure session must keep the model's resolved wire protocol."""

import json

import pytest

from hermes_cli.config import atomic_config_write
from hermes_state import SessionDB
from tui_gateway import server


@pytest.mark.parametrize("route_source", ["desktop", "gateway"])
@pytest.mark.parametrize("model, saved_mode, expected_mode, fresh_mode", [
    ("gpt-5.3-codex", "chat_completions", "codex_responses", "codex_responses"),
    ("gpt-5.3-codex", "anthropic_messages", "anthropic_messages", "codex_responses"),
    ("gpt-4o", "chat_completions", "chat_completions", "chat_completions"),
    ("grok-4.7", "chat_completions", "chat_completions", "chat_completions"),
    ("custom-deployment", "codex_responses", "codex_responses", "chat_completions"),
    ("gpt-5.3-codex", None, "codex_responses", "codex_responses"),
    ("openai/gpt-5.3-codex", "chat_completions", "codex_responses", "codex_responses"),
    ("o3", "chat_completions", "codex_responses", "codex_responses"),
])
def test_resume_azure_route_follows_model_policy(
    tmp_path, monkeypatch, model, saved_mode, expected_mode, fresh_mode, route_source,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "fixture-azure-key")
    base_url = "https://fixture.openai.azure.com/openai/v1"
    monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", base_url)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    atomic_config_write(tmp_path / "config.yaml", {
        "model": {"provider": "azure-foundry", "default": "gpt-4o"},
    })
    fresh_model, fresh_runtime = server._resolve_agent_model_runtime(model, "azure-foundry")
    assert fresh_model == model
    assert fresh_runtime["api_mode"] == fresh_mode

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("azure-resume", source=route_source, model=model)
        route = {
            "provider": "azure-foundry", "base_url": base_url,
            "api_mode": saved_mode,
        }
        metadata = {"model": model}
        metadata.update(route if route_source == "desktop" else {"gateway_runtime": route})
        db.update_session_meta("azure-resume", json.dumps(metadata), model)
        overrides = server._stored_session_runtime_overrides(db.get_session("azure-resume"))
        selected, runtime = server._resolve_agent_model_runtime(
            overrides["model_override"], overrides.get("provider_override"))
    finally:
        db.close()

    assert selected == model
    assert runtime["provider"] == "azure-foundry"
    assert runtime["api_mode"] == expected_mode
    assert runtime["base_url"] == fresh_runtime["base_url"] == base_url
    assert runtime["api_key"] == fresh_runtime["api_key"] == "fixture-azure-key"
