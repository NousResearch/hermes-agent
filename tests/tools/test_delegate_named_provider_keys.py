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
    with pytest.raises(ValueError, match="delegation.base_url differs"):
        _resolve_delegation_credentials(
            {**config, "base_url": "https://other-provider.invalid/v1"}, parent
        )
