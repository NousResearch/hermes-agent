"""A named delegation provider keeps its own key on delegation.base_url, never the parent's (#150172)."""

from types import SimpleNamespace

import pytest

from tools.delegate_tool_config import _resolve_delegation_credentials


def _parent(key="parent-provider-key"):
    return SimpleNamespace(api_key=key, request_overrides=None)


def test_reporter_config_uses_xiaomi_key_on_token_plan_url(monkeypatch):
    """The issue's exact config: provider xiaomi + token-plan URL, XIAOMI_API_KEY set, no XIAOMI_BASE_URL."""
    monkeypatch.setenv("XIAOMI_API_KEY", "xiaomi-env-key")
    monkeypatch.delenv("XIAOMI_BASE_URL", raising=False)
    config = {"provider": "xiaomi", "model": "mimo-v2.6-pro",
              "base_url": "https://token-plan-sgp.xiaomimimo.com/v1"}

    resolved = _resolve_delegation_credentials(config, _parent("minimax-parent-key"))

    assert resolved["provider"] == "xiaomi"
    assert resolved["base_url"] == "https://token-plan-sgp.xiaomimimo.com/v1"
    assert resolved["api_key"] == "xiaomi-env-key"


def test_same_provider_alternate_path_keeps_working(monkeypatch):
    """MiniMax parent delegating to MiniMax's /v1 (not its /anthropic default) gets the MiniMax key there."""
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-env-key")
    config = {"provider": "minimax", "model": "MiniMax-M2", "base_url": "https://api.minimax.io/v1"}

    resolved = _resolve_delegation_credentials(config, _parent("minimax-env-key"))

    assert (resolved["provider"], resolved["base_url"], resolved["api_key"]) == (
        "minimax", "https://api.minimax.io/v1", "minimax-env-key")
    # Without a cached models.dev catalog the resolver keeps MiniMax's declared Messages wire on /v1;
    # an explicit delegation.api_mode is the operator's override and must reach the child.
    config["api_mode"] = "chat_completions"
    assert _resolve_delegation_credentials(config, _parent())["api_mode"] == "chat_completions"


def test_parent_key_never_reaches_another_providers_base_url(monkeypatch):
    """Parent on provider A, child pinned to provider B with B's URL: B's key or a refusal, never A's key."""
    monkeypatch.setenv("XIAOMI_API_KEY", "xiaomi-env-key")
    config = {"provider": "xiaomi", "model": "mimo-v2.6-pro", "base_url": "https://api.xiaomimimo.com/v1"}
    assert _resolve_delegation_credentials(config, _parent("minimax-parent-key"))["api_key"] == "xiaomi-env-key"

    monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="XIAOMI_API_KEY|no API key"):
        _resolve_delegation_credentials(config, _parent("minimax-parent-key"))


def test_withheld_key_error_names_delegation_api_key(monkeypatch):
    """OpenRouter on a mirror: host gating withholds OPENROUTER_API_KEY, so the refusal must name the recovery."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-env-key")
    config = {"provider": "openrouter", "model": "m", "base_url": "https://or-mirror.example/api/v1"}
    with pytest.raises(ValueError, match="delegation.api_key"):
        _resolve_delegation_credentials(config, _parent())


def test_resolver_that_ignores_base_url_is_refused(monkeypatch):
    """A provider whose resolver keeps its own endpoint must not silently reroute the child off delegation.base_url."""
    from hermes_cli import runtime_provider
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **_: {
        "provider": "minimax-oauth", "base_url": "https://api.minimax.io/anthropic", "api_key": "oauth-token"})
    for base_url in ("https://llm-proxy.example/anthropic", "https://api.minimax.io/v1"):
        config = {"provider": "minimax-oauth", "model": "MiniMax-M2", "base_url": base_url}
        with pytest.raises(ValueError, match="does not accept delegation.base_url"):
            _resolve_delegation_credentials(config, _parent())


def test_resolver_path_normalisation_is_not_a_reroute(monkeypatch):
    """LM Studio's resolver rewrites /api/v1 to /v1, and Messages-mode resolvers (Azure Foundry, OpenCode-hosted
    providers: entries) strip /v1: that is the configured endpoint, not a reroute."""
    config = {"provider": "lmstudio", "model": "m", "base_url": "http://localhost:1234/api/v1"}
    assert _resolve_delegation_credentials(config, _parent())["base_url"] == "http://localhost:1234/v1"

    from hermes_cli import runtime_provider
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **_: {
        "provider": "azure-foundry", "base_url": "https://res.services.ai.azure.com/anthropic",
        "api_key": "foundry-key", "api_mode": "anthropic_messages"})
    config = {"provider": "azure-foundry", "model": "claude", "base_url": "https://res.services.ai.azure.com/anthropic/v1"}
    assert _resolve_delegation_credentials(config, _parent())["api_key"] == "foundry-key"


@pytest.mark.parametrize("provider", ["", "custom", "auto"])
def test_unnamed_endpoint_still_inherits_parent_key(provider):
    """Bare / custom base_url keeps the documented direct-endpoint behaviour (parent key filled in later)."""
    config = {"provider": provider, "model": "m", "base_url": "https://llm.internal.example/v1"}

    resolved = _resolve_delegation_credentials(config, _parent())

    assert (resolved["provider"], resolved["base_url"], resolved["api_key"]) == (
        "custom", "https://llm.internal.example/v1", None)
