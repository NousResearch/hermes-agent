"""Explicit direct endpoints fail before provider lookup, child creation or HTTP retries."""

from unittest.mock import patch

import pytest

from tools.delegate_tool import _resolve_delegation_credentials


@pytest.mark.parametrize("provider", [None, "custom", "claude-subscription-directsdk-experimental"])
@pytest.mark.parametrize("base_url", ["process://claude-subscription-directsdk-experimental", "file:///not-an-api", "localhost:8080/v1"])
def test_non_http_direct_endpoint_is_rejected_before_runtime_lookup(provider, base_url):
    route = {"model": "test-model", "base_url": base_url}
    if provider:
        route["provider"] = provider
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolve:
        with pytest.raises(ValueError, match=r"delegation\.base_url.*http://.*https://") as error:
            _resolve_delegation_credentials(route, parent_agent=None)
    assert "remove" in str(error.value).lower()
    resolve.assert_not_called()


@pytest.mark.parametrize("base_url", ["http://localhost:8080/v1", "https://relay.example.test/v1", "HTTPS://relay.example.test/v1"])
def test_http_direct_endpoints_still_resolve(base_url):
    credentials = _resolve_delegation_credentials({"model": "test-model", "base_url": base_url}, None)
    assert credentials["provider"] == "custom"
    assert credentials["base_url"] == base_url


def test_provider_owned_process_address_is_not_an_explicit_direct_endpoint():
    provider = "test-sdk-plugin"
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
        "provider": provider, "base_url": "process://test-sdk-plugin", "api_key": "external-process",
        "api_mode": "test_sdk",
    }):
        credentials = _resolve_delegation_credentials({"provider": provider, "model": "test-model"}, None)
    assert credentials["provider"] == provider
    assert credentials["base_url"] == "process://test-sdk-plugin"
    assert credentials["api_mode"] == "test_sdk"


def test_tool_refuses_a_process_endpoint_before_building_a_child(monkeypatch):
    import importlib
    import json
    from types import SimpleNamespace

    from tools import delegate_tool as dt
    from tools import delegate_tool_routes as routes
    from tools.registry import registry

    backend = importlib.import_module("plugins.delegation.default").DefaultDelegationProvider()
    config = {"backend": "default", "model": "test-model", "base_url": "process://test-sdk-plugin"}
    monkeypatch.setattr(dt, "_load_config", lambda: config)
    monkeypatch.setattr(routes, "_active_provider", lambda config: backend)
    parent = SimpleNamespace(_delegate_depth=0, model="parent-model", provider="parent-provider",
                             _delegation_backend="default", session_id="endpoint-refusal-test")
    with patch.object(dt, "_build_child_preserving_parent_tools") as build:
        output = registry.dispatch("delegate_task", {"tasks": [{"goal": "must not start"}]}, parent_agent=parent)
    if isinstance(output, str):
        output = json.loads(output)
    assert "delegation.base_url" in output["error"]
    assert "http://" in output["error"]
    build.assert_not_called()
