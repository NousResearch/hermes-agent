"""Inherited custom-provider identity with shared gateway URLs."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool_config import _resolve_delegation_credentials


@pytest.fixture
def shared_custom_route(monkeypatch):
    from hermes_cli import runtime_provider as rp

    endpoint = "https://shared.invalid/v1"
    key = "test-custom-shared-route-key"
    provider_b = {"api": endpoint, "api_key": key, "models": ["model-b"]}
    config = {"providers": {
        "provider-a": {"api": endpoint, "api_key": key, "models": ["model-a"]},
        "provider-b": provider_b,
    }}
    monkeypatch.setattr(rp, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(rp, "_get_model_config", dict)
    parent = SimpleNamespace(
        provider="custom", requested_provider=None, model="model-b", base_url=endpoint + "/",
        api_key=key, api_mode="chat_completions",
        request_overrides={"extra_body": {"route": "live", "session": {"value": 1}}},
    )
    return parent, provider_b


def test_recovered_same_route_inherits_live_request_overrides(shared_custom_route):
    parent, _ = shared_custom_route

    creds = _resolve_delegation_credentials({}, parent)

    assert creds["provider"] == "custom:provider-b"
    assert creds["request_overrides"] == parent.request_overrides
    creds["request_overrides"]["extra_body"]["session"]["value"] = 2
    assert parent.request_overrides["extra_body"]["session"]["value"] == 1


def test_recovered_same_route_merges_config_live_and_explicit_overrides(shared_custom_route):
    parent, provider_b = shared_custom_route
    provider_b["extra_body"] = {"route": "configured", "default": True}

    live = _resolve_delegation_credentials({}, parent)
    assert live["request_overrides"] == {"extra_body": {
        "route": "live", "default": True, "session": {"value": 1},
    }}
    explicit = _resolve_delegation_credentials(
        {"request_overrides": {"extra_body": {"route": "delegation"}}}, parent,
    )
    assert explicit["request_overrides"] == {"extra_body": {
        "route": "delegation", "default": True, "session": {"value": 1},
    }}


def test_model_switch_on_shared_endpoint_does_not_inherit_live_overrides(shared_custom_route):
    parent, _ = shared_custom_route

    creds = _resolve_delegation_credentials({"model": "model-a"}, parent)

    assert creds["provider"] == "custom:provider-a"
    assert creds["api_key"] == parent.api_key
    assert creds["request_overrides"] == {}


def test_moved_endpoint_does_not_inherit_live_overrides(shared_custom_route):
    parent, provider_b = shared_custom_route
    provider_b["api"] = "https://shared.invalid/other/v1"
    provider_b["extra_body"] = {"route": "configured"}

    creds = _resolve_delegation_credentials({}, parent)

    assert creds["base_url"] == "https://shared.invalid/other/v1"
    assert creds["request_overrides"] == {"extra_body": {"route": "configured"}}


class TestDelegationCredentialResolution(unittest.TestCase):
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    @patch("hermes_cli.runtime_provider.canonical_custom_identity")
    def test_no_override_disambiguates_shared_custom_url_by_parent_model(
        self, mock_canonical, mock_resolve
    ):
        """A switched custom model must not inherit the previous provider key."""
        mock_canonical.return_value = "custom:provider-b"
        mock_resolve.return_value = {
            "provider": "custom",
            "model": "model-b-default",
            "base_url": "https://shared.invalid/v1",
            "api_key": "provider-b-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        parent.model = "model-b-current"
        parent.provider = "custom:provider-a"
        parent.base_url = "https://shared.invalid/v1"

        creds = _resolve_delegation_credentials(
            {"model": "", "provider": ""}, parent
        )

        self.assertEqual(creds["model"], "model-b-current")
        self.assertEqual(creds["provider"], "custom:provider-b")
        self.assertEqual(creds["api_key"], "provider-b-key")
        mock_canonical.assert_called_once_with(
            base_url="https://shared.invalid/v1",
            config_provider="custom:provider-a",
            model="model-b-current",
        )
        mock_resolve.assert_called_once_with(
            requested="custom:provider-b", target_model="model-b-current"
        )

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    @patch("hermes_cli.runtime_provider.canonical_custom_identity")
    def test_builtin_parent_keeps_ordinary_inheritance_path(
        self, mock_canonical, mock_resolve
    ):
        """A built-in parent must not be remapped by a matching custom URL."""
        parent = _make_mock_parent(depth=0)
        parent.base_url = "https://shared.invalid/v1"

        creds = _resolve_delegation_credentials(
            {"model": "", "provider": ""}, parent
        )

        self.assertIsNone(creds["provider"])
        self.assertIsNone(creds["base_url"])
        self.assertIsNone(creds["api_key"])
        mock_canonical.assert_not_called()
        mock_resolve.assert_not_called()

