import os
import json
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

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


def test_managed_vendor_endpoints_pin_the_deployed_gateway_url():
    """The exact URL an agent may connect to is a code fact, not a lookup.

    Exercises the real ``build_vendor_gateway_url`` (which once resolved a
    typo'd pseudo-vendor to a non-existent host while every other test stubbed
    it): default builder, real deployed host, pinned vendor path.
    """
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True), \
         patch.dict(
             os.environ,
             {"TOOL_GATEWAY_DOMAIN": "nousresearch.com", "TOOL_GATEWAY_SCHEME": "https"},
             clear=False,
         ):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        endpoints = managed_tool_gateway.managed_vendor_endpoints("bfl")

    assert endpoints == {
        "origin": "https://tool-gateway.nousresearch.com",
        "base_url": "https://tool-gateway.nousresearch.com/api/bfl",
        "upload_path": "/api/uploads/bfl",
    }


def test_managed_vendor_endpoints_unreachable_when_managed_tools_disabled():
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=False):
        assert managed_tool_gateway.managed_vendor_endpoints("bfl") is None


def test_managed_gateway_auth_headers_carry_the_bearer():
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        headers = managed_tool_gateway.managed_gateway_auth_headers(
            "https://tool-gateway.example.com/api/bfl/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: "nous-token",
        )

    assert headers == {"Authorization": "Bearer nous-token"}


def test_managed_gateway_auth_headers_reflect_a_rotated_token():
    # Read fresh on every call: a Nous access token expires within the hour,
    # and a long session must not keep presenting a dead bearer.
    tokens = iter(["first-token", "second-token"])
    builder = lambda vendor: f"https://{vendor}-gateway.example.com"
    url = "https://tool-gateway.example.com/api/bfl/generations"

    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        first = managed_tool_gateway.managed_gateway_auth_headers(url, builder, lambda: next(tokens))
        second = managed_tool_gateway.managed_gateway_auth_headers(url, builder, lambda: next(tokens))

    assert first["Authorization"] == "Bearer first-token"
    assert second["Authorization"] == "Bearer second-token"


def test_managed_gateway_auth_headers_refuse_a_url_off_the_gateway_origin():
    # Gated on the URL, never a name: our bearer must never be handed to a
    # host that merely looks managed.
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        assert managed_tool_gateway.managed_gateway_auth_headers(
            "https://attacker.example/api/bfl/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: "nous-token",
        ) == {}


def test_managed_gateway_auth_headers_empty_without_a_token():
    # Empty rather than raising, so a caller can say "sign in" instead of
    # sending an unauthenticated request.
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        assert managed_tool_gateway.managed_gateway_auth_headers(
            "https://tool-gateway.example.com/api/bfl/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: None,
        ) == {}


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
