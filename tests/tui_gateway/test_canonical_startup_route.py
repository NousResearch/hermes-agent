"""Phase 5.8.6.1: canonical TUI startup and legacy session routing."""
from __future__ import annotations

from tui_gateway.model_startup_route import (
    is_routable_provider, recover_custom_identity, resolve_startup_seed,
)
from tui_gateway.agent_factory import _rederive_per_model_route


def _alias_config():
    return {
        "model": {"default": "main-model", "provider": "openrouter"},
        "model_aliases": {
            "alias-host": {
                "model": "alias-model", "provider": "custom",
                "base_url": "https://alias.example/v1", "api_key": "alias-key",
            },
        },
        "providers": {
            "relay": {"name": "Relay", "base_url": "https://relay.example/v1", "model": "relay-model"},
        },
    }


def test_launch_alias_retains_its_own_endpoint_and_key():
    route = resolve_startup_seed("alias-host", "openrouter", _alias_config())
    assert (route.model, route.provider, route.base_url, route.api_key) == (
        "alias-model", "custom", "https://alias.example/v1", "alias-key",
    )


def test_explicit_provider_overrides_alias_identity_without_borrowing_key():
    route = resolve_startup_seed(
        "alias-host", "openrouter", _alias_config(), explicit_provider="openrouter",
    )
    assert (route.model, route.provider, route.base_url, route.api_key) == (
        "alias-model", "openrouter", "https://alias.example/v1", "",
    )


def test_qualified_named_custom_route_uses_canonical_identity():
    route = resolve_startup_seed("custom:relay:relay-model", "openrouter", _alias_config())
    assert (route.model, route.provider) == ("relay-model", "custom:relay")


def test_restore_custom_identity_from_configured_endpoint():
    cfg = _alias_config()
    assert recover_custom_identity(cfg, base_url="https://relay.example/v1") == "custom:relay"
    assert recover_custom_identity(cfg, model="relay-model") == "custom:relay"
    assert is_routable_provider("custom:relay", cfg)
    assert not is_routable_provider("custom", cfg)
    assert not is_routable_provider("custom:deleted-ages-ago", cfg)
    assert not is_routable_provider("not-registered", cfg)


def test_legacy_mode_cannot_override_model_specific_canonical_route():
    runtime = {
        "provider": "opencode-go", "requested_provider": "opencode-go",
        "base_url": "https://opencode.ai/zen/v1", "api_mode": "anthropic_messages",
    }
    _rederive_per_model_route(
        "deepseek-v4-flash-vision-exp", runtime, acquired_api_mode="chat_completions",
    )
    assert (runtime["api_mode"], runtime["base_url"]) == (
        "chat_completions", "https://opencode.ai/zen/go/v1",
    )
