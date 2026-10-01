"""TUI switch: URL-bearing aliases retain their host and never borrow the wrong key."""
from __future__ import annotations

from tui_gateway import model_switch_resolution


def _config():
    return {
        "model": {"provider": "openrouter", "default": "old-model"},
        "model_aliases": {
            "relay": {
                "model": "relay-model", "provider": "custom",
                "base_url": "https://relay.example/v1", "api_key": "alias-secret",
            },
        },
    }


def test_explicit_provider_keeps_alias_endpoint_not_alias_key(monkeypatch):
    seen = {}

    def acquire(**kwargs):
        seen.update(kwargs)
        return {
            "provider": "openrouter", "api_key": "provider-key",
            "base_url": kwargs.get("explicit_base_url") or "https://api.openrouter.ai/v1",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", acquire,
    )
    monkeypatch.setattr(model_switch_resolution, "enrich_model_switch", lambda result, cfg: result)

    result = model_switch_resolution.resolve_tui_model_switch(
        config=_config(), raw_input="relay", current_provider="openrouter",
        current_model="old-model", explicit_provider="openrouter",
    )
    assert result.success
    assert result.new_model == "relay-model"
    assert result.base_url == "https://relay.example/v1"
    assert result.api_key == "provider-key"
    assert seen["explicit_base_url"] == "https://relay.example/v1"
    assert not seen["explicit_api_key"]


def test_implicit_alias_acquires_credential_for_alias_endpoint(monkeypatch):
    seen = {}

    def acquire(**kwargs):
        seen.update(kwargs)
        return {
            "provider": "custom", "api_key": kwargs["explicit_api_key"],
            "base_url": kwargs["explicit_base_url"], "api_mode": "chat_completions",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", acquire,
    )
    monkeypatch.setattr(model_switch_resolution, "enrich_model_switch", lambda result, cfg: result)
    result = model_switch_resolution.resolve_tui_model_switch(
        config=_config(), raw_input="relay", current_provider="openrouter",
        current_model="old-model",
    )
    assert result.success
    assert result.base_url == "https://relay.example/v1"
    assert result.api_key == "alias-secret"
    assert seen["requested"] == "custom"
