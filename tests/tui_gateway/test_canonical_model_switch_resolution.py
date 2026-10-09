"""TUI switch decisions consume canonical selection and routing, not the CLI switcher."""
from __future__ import annotations

import pytest

from tui_gateway import model_switch_resolution as seam


@pytest.fixture(autouse=True)
def cheap_enrichment(monkeypatch):
    monkeypatch.setattr(seam, "enrich_model_switch", lambda result, _config: result)


def _resolve(raw, current="openrouter", *, explicit="", config=None,
             url="", key=""):
    return seam.resolve_tui_model_switch(
        raw_input=raw, explicit_provider=explicit, current_provider=current,
        current_model="old-model", current_base_url=url,
        current_api_key=key, config=config or {},
    )


def test_same_custom_route_reuses_session_credentials_without_acquisition(monkeypatch):
    def forbid(**_kwargs):
        raise AssertionError("unchanged custom provider must reuse live credentials")
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", forbid)
    result = _resolve(
        "next-model", current="custom", url="http://127.0.0.1:8100/v1", key="live-key",
    )
    assert result.target_provider == "custom"
    assert result.new_model == "next-model"
    assert result.api_key == "live-key"
    assert result.base_url == "http://127.0.0.1:8100/v1"


def test_unknown_explicit_provider_rejects_before_credentials(monkeypatch):
    def forbid(**_kwargs):
        raise AssertionError("unknown provider must fail before credential lookup")
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", forbid)
    with pytest.raises(ValueError, match="Unknown provider"):
        _resolve("some-model", explicit="definitely-not-a-provider")


def test_named_custom_provider_preserves_identity_and_acquisition_key(monkeypatch):
    calls = []
    def credentials(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "custom", "base_url": "http://127.0.0.1:4141/v1",
            "api_key": "named-key", "api_mode": "chat_completions",
        }
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", credentials)
    cfg = {"providers": {
        "relay": {"name": "Relay", "base_url": "http://127.0.0.1:4141/v1",
                  "model": "relay-model"},
    }}
    result = _resolve("relay-model", explicit="relay", config=cfg)
    assert calls[0]["requested"] == "relay"
    assert result.target_provider == "custom:relay"
    assert result.api_key == "named-key"
    assert result.new_model == "relay-model"


def test_direct_alias_resolves_its_host_credential(monkeypatch):
    calls = []
    def credentials(**kwargs):
        calls.append(kwargs)
        return {"provider": "custom", "base_url": kwargs["explicit_base_url"],
                "api_key": kwargs["explicit_api_key"]}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", credentials)
    cfg = {"model_aliases": {
        "private-relay": {"model": "relay-model", "provider": "custom",
                          "base_url": "https://relay.example/v1",
                          "api_key": "alias-secret"},
    }}
    result = _resolve("private-relay", config=cfg)
    assert calls[0]["requested"] == "custom"
    assert calls[0]["explicit_base_url"] == "https://relay.example/v1"
    assert calls[0]["explicit_api_key"] == "alias-secret"
    assert result.target_provider == "custom"
    assert result.new_model == "relay-model"


def test_explicit_provider_overrides_alias_without_borrowing_its_key(monkeypatch):
    calls = []
    def credentials(**kwargs):
        calls.append(kwargs)
        return {"provider": "openrouter", "base_url": kwargs["explicit_base_url"],
                "api_key": "explicit-provider-key"}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", credentials)
    cfg = {"model_aliases": {
        "private-relay": {"model": "relay-model", "provider": "custom",
                          "base_url": "https://relay.example/v1",
                          "api_key": "alias-secret"},
    }}
    result = _resolve("private-relay", explicit="openrouter", config=cfg)
    assert calls[0]["requested"] == "openrouter"
    assert calls[0]["explicit_base_url"] == "https://relay.example/v1"
    assert calls[0]["explicit_api_key"] is None
    assert result.api_key == "explicit-provider-key"


def test_model_specific_wire_ignores_stale_legacy_api_mode():
    result = _resolve(
        "deepseek-v4-flash-vision-exp", current="opencode-go",
        url="https://opencode.ai/zen/v1", key="key",
    )
    assert result.api_mode == "chat_completions"
    assert result.base_url == "https://opencode.ai/zen/go/v1"
