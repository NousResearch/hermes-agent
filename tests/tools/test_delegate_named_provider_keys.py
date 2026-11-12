"""Named delegation providers keep their scoped keys on matching endpoints (#150172)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.delegate_tool_config import _resolve_delegation_credentials


@patch("hermes_cli.runtime_provider.resolve_runtime_provider")
def test_named_provider_base_url_uses_scoped_key(mock_resolve):
    endpoint = "https://token-plan-sgp.xiaomimimo.com/v1"
    mock_resolve.return_value = {
        "provider": "custom",
        "base_url": endpoint,
        "api_key": "xiaomi-scoped-key",
        "api_mode": "chat_completions",
    }
    parent = SimpleNamespace(api_key="minimax-parent-key")
    config = {
        "provider": "xiaomi",
        "model": "mimo-v2.6-pro",
        "base_url": endpoint,
    }

    resolved = _resolve_delegation_credentials(config, parent)
    assert resolved["provider"] == "xiaomi"
    assert resolved["base_url"] == endpoint
    assert resolved["api_key"] == "xiaomi-scoped-key"
    assert resolved["api_key"] != parent.api_key
    mock_resolve.assert_called_once_with(
        requested="xiaomi", target_model="mimo-v2.6-pro"
    )

    # A differently configured host must not receive the scoped provider key.
    with pytest.raises(ValueError, match="differs from provider 'xiaomi'"):
        _resolve_delegation_credentials(
            {**config, "base_url": "https://other-provider.invalid/v1"}, parent
        )


def test_registry_provider_with_its_own_base_url_resolves_its_key(monkeypatch):
    """Real resolution: a registry provider's key_env is used on its endpoint, never the parent's key."""
    monkeypatch.setenv("XIAOMI_API_KEY", "xiaomi-env-key")
    monkeypatch.setenv("XIAOMI_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
    parent = SimpleNamespace(api_key="minimax-parent-key", request_overrides=None)
    config = {"provider": "xiaomi", "model": "mimo-v2.6-pro", "api_mode": "chat_completions",
              "base_url": "https://token-plan-sgp.xiaomimimo.com/v1/"}

    resolved = _resolve_delegation_credentials(config, parent)

    assert (resolved["provider"], resolved["api_key"]) == ("xiaomi", "xiaomi-env-key")
    assert resolved["api_mode"] == "chat_completions"


def test_registry_provider_without_a_key_refuses_instead_of_inheriting(monkeypatch):
    monkeypatch.delenv("XIAOMI_API_KEY", raising=False)
    parent = SimpleNamespace(api_key="minimax-parent-key", request_overrides=None)
    config = {"provider": "xiaomi", "model": "mimo-v2.6-pro", "base_url": "https://api.xiaomimimo.com/v1"}

    with pytest.raises(ValueError, match="XIAOMI_API_KEY"):
        _resolve_delegation_credentials(config, parent)
