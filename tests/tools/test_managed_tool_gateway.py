import os
import json
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest.mock import patch

from tools import managed_gateway_auth


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "managed_tool_gateway.py"
MODULE_SPEC = spec_from_file_location("managed_tool_gateway_test_module", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
managed_tool_gateway = module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = managed_tool_gateway
MODULE_SPEC.loader.exec_module(managed_tool_gateway)
is_managed_tool_gateway_ready = managed_tool_gateway.is_managed_tool_gateway_ready
resolve_managed_tool_gateway = managed_tool_gateway.resolve_managed_tool_gateway


def test_resolve_managed_tool_gateway_derives_vendor_origin_from_shared_domain():
    with patch.dict(
        os.environ,
        {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        },
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        result = resolve_managed_tool_gateway(
            "firecrawl",
            token_reader=lambda: "nous-token",
        )

    assert result is not None
    assert result.gateway_origin == "https://firecrawl-gateway.nousresearch.com"
    assert result.nous_user_token == "nous-token"
    assert result.managed_mode is True


def test_resolve_managed_tool_gateway_uses_vendor_specific_override():
    with patch.dict(
        os.environ,
        {
            "BROWSER_USE_GATEWAY_URL": "http://browser-use-gateway.localhost:3009/",
        },
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        result = resolve_managed_tool_gateway(
            "browser-use",
            token_reader=lambda: "nous-token",
        )

    assert result is not None
    assert result.gateway_origin == "http://browser-use-gateway.localhost:3009"


def test_resolve_managed_tool_gateway_is_inactive_without_nous_token():
    with patch.dict(
        os.environ,
        {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        },
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        result = resolve_managed_tool_gateway(
            "firecrawl",
            token_reader=lambda: None,
        )

    assert result is None


def test_resolve_managed_tool_gateway_is_disabled_without_subscription():
    with patch.dict(os.environ, {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"}, clear=False), \
         patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=False):
        result = resolve_managed_tool_gateway(
            "firecrawl",
            token_reader=lambda: "nous-token",
        )

    assert result is None


def test_read_nous_access_token_refreshes_expiring_cached_token(tmp_path, monkeypatch):
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    (tmp_path / "auth.json").write_text(json.dumps({
        "providers": {
            "nous": {
                "access_token": "stale-token",
                "refresh_token": "refresh-token",
                "expires_at": expires_at,
            }
        }
    }))
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_access_token",
        lambda refresh_skew_seconds=120: "fresh-token",
    )

    assert managed_tool_gateway.read_nous_access_token() == "fresh-token"


def test_is_managed_tool_gateway_ready_skips_refresh_for_expired_cached_token(tmp_path, monkeypatch):
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expired_at = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    (tmp_path / "auth.json").write_text(json.dumps({
        "providers": {
            "nous": {
                "access_token": "expired-token",
                "refresh_token": "refresh-token",
                "expires_at": expired_at,
            }
        }
    }))
    refresh_calls = []

    def _record_refresh(*, refresh_skew_seconds=120, **_kwargs):
        refresh_calls.append(refresh_skew_seconds)
        return "fresh-token"

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_access_token",
        _record_refresh,
    )

    with patch.dict(
        os.environ,
        {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"},
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        assert is_managed_tool_gateway_ready("modal") is True

    assert refresh_calls == []


def test_connector_gateway_origin_pins_the_deployed_connectors_host():
    # The connectors API is its own deployment on its own canonical host, so
    # the default resolution must not land on the media/vendor origin.
    with patch.dict(
        os.environ,
        {"TOOL_GATEWAY_DOMAIN": "nousresearch.com", "TOOL_GATEWAY_SCHEME": "https"},
        clear=False,
    ):
        os.environ.pop("CONNECTOR_GATEWAY_URL", None)
        assert managed_gateway_auth.connector_gateway_origin() == (
            "https://connector-gateway.nousresearch.com"
        )

def test_managed_gateway_origin_honors_the_harness_override():
    # TOOL_GATEWAY_URL pins the full media origin (the e2e harness sets it to a
    # loopback gateway), and the bearer gate must accept exactly that origin.
    with patch.dict(os.environ, {"TOOL_GATEWAY_URL": "http://127.0.0.1:3009/"}, clear=False):
        os.environ.pop("CONNECTOR_GATEWAY_URL", None)
        assert managed_gateway_auth.managed_gateway_origin() == "http://127.0.0.1:3009"
        assert managed_gateway_auth.is_managed_nous_gateway_url(
            "http://127.0.0.1:3009/api/vendorx/generations"
        )
        assert not managed_gateway_auth.is_managed_nous_gateway_url(
            "https://tools.nousresearch.com/api/vendorx/generations"
        )

def test_connector_gateway_origin_honors_its_own_override():
    # CONNECTOR_GATEWAY_URL is the connectors host's own key: it moves the
    # connectors origin without touching the media origin, and the bearer gate
    # accepts the overridden origin.
    with patch.dict(
        os.environ,
        {
            "CONNECTOR_GATEWAY_URL": "http://127.0.0.1:3009/",
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        },
        clear=False,
    ):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        assert managed_gateway_auth.connector_gateway_origin() == "http://127.0.0.1:3009"
        assert managed_gateway_auth.managed_gateway_origin() == (
            "https://tool-gateway.nousresearch.com"
        )
        assert managed_gateway_auth.is_managed_nous_gateway_url(
            "http://127.0.0.1:3009/v1/connectors/search"
        )

def test_default_bearer_gate_accepts_both_deployed_hosts_only():
    # Exact (scheme, netloc) equality against each deployed origin. Both
    # first-party hosts are in; the retired `tools.` host, subdomain cousins,
    # and scheme downgrades are all out.
    with patch.dict(
        os.environ,
        {"TOOL_GATEWAY_DOMAIN": "nousresearch.com", "TOOL_GATEWAY_SCHEME": "https"},
        clear=False,
    ):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        os.environ.pop("CONNECTOR_GATEWAY_URL", None)
        for trusted in (
            "https://connector-gateway.nousresearch.com/v1/connectors/execute",
            "https://tool-gateway.nousresearch.com/api/vendorx/generations",
        ):
            assert managed_gateway_auth.is_managed_nous_gateway_url(trusted)
        for untrusted in (
            "https://tools.nousresearch.com/v1/connectors/execute",
            "https://evil-connector-gateway.nousresearch.com.attacker.dev/v1/connectors",
            "https://connector-gateway.nousresearch.com.attacker.dev/v1/connectors",
            "http://connector-gateway.nousresearch.com/v1/connectors",
            "http://tool-gateway.nousresearch.com/api/vendorx/generations",
        ):
            assert not managed_gateway_auth.is_managed_nous_gateway_url(untrusted)
