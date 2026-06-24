"""Tests for tools/managed_tool_gateway.py — Nous-hosted vendor passthrough helpers.

Covers: auth_json_path, _read_nous_provider_state, _parse_timestamp,
_access_token_is_expiring, peek_nous_access_token, read_nous_access_token,
get_tool_gateway_scheme, build_vendor_gateway_url,
resolve_managed_tool_gateway, is_managed_tool_gateway_ready.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.managed_tool_gateway import (
    ManagedToolGatewayConfig,
    _access_token_is_expiring,
    _parse_timestamp,
    _read_nous_provider_state,
    auth_json_path,
    build_vendor_gateway_url,
    get_tool_gateway_scheme,
    is_managed_tool_gateway_ready,
    peek_nous_access_token,
    read_nous_access_token,
    resolve_managed_tool_gateway,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Set HERMES_HOME to a tmp dir and return it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all gateway-related env vars."""
    for key in (
        "TOOL_GATEWAY_USER_TOKEN",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "FIRECRAWL_GATEWAY_URL",
        "BROWSER_USE_GATEWAY_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_auth_file(home: Path, providers: dict) -> None:
    """Write an auth.json with the given providers dict."""
    (home / "auth.json").write_text(json.dumps({"providers": providers}))


# ---------------------------------------------------------------------------
# auth_json_path
# ---------------------------------------------------------------------------


class TestAuthJsonPath:
    def test_returns_hermes_home_auth_json(self, hermes_home):
        assert auth_json_path() == hermes_home / "auth.json"

    def test_respects_hermes_home_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom-home"
        custom.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(custom))
        assert auth_json_path() == custom / "auth.json"


# ---------------------------------------------------------------------------
# _read_nous_provider_state
# ---------------------------------------------------------------------------


class TestReadNousProviderState:
    def test_returns_none_when_no_auth_file(self, hermes_home):
        assert _read_nous_provider_state() is None

    def test_returns_nous_provider_when_present(self, hermes_home):
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "TOK", "expires_at": "2026-01-01T00:00:00Z"},
            },
        )
        result = _read_nous_provider_state()
        assert result == {"access_token": "TOK", "expires_at": "2026-01-01T00:00:00Z"}

    def test_returns_empty_dict_when_no_nous_provider(self, hermes_home):
        """providers.get("nous", {}) returns {} which is a dict → returned as-is."""
        _write_auth_file(hermes_home, {"other": {"access_token": "TOK"}})
        assert _read_nous_provider_state() == {}

    def test_returns_none_when_providers_not_dict(self, hermes_home):
        _write_auth_file(hermes_home, {})
        (hermes_home / "auth.json").write_text(json.dumps({"providers": "not-a-dict"}))
        assert _read_nous_provider_state() is None

    def test_returns_none_when_nous_provider_not_dict(self, hermes_home):
        _write_auth_file(hermes_home, {"nous": "not-a-dict"})
        assert _read_nous_provider_state() is None

    def test_returns_empty_dict_when_nous_is_empty(self, hermes_home):
        _write_auth_file(hermes_home, {"nous": {}})
        assert _read_nous_provider_state() == {}

    def test_returns_none_on_invalid_json(self, hermes_home):
        (hermes_home / "auth.json").write_text("not json {{{")
        assert _read_nous_provider_state() is None

    def test_returns_none_on_read_error(self, hermes_home):
        # Create auth.json as a directory to trigger OSError
        (hermes_home / "auth.json").mkdir()
        assert _read_nous_provider_state() is None

    def test_returns_empty_dict_when_providers_key_missing(self, hermes_home):
        """data.get("providers", {}) returns {} → nous defaults to {} → returned."""
        (hermes_home / "auth.json").write_text(json.dumps({"other_key": 42}))
        assert _read_nous_provider_state() == {}


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    def test_parses_iso_with_z_suffix(self):
        result = _parse_timestamp("2026-01-01T00:00:00Z")
        assert result == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_parses_iso_with_utc_offset(self):
        result = _parse_timestamp("2026-01-01T00:00:00+00:00")
        assert result == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_parses_iso_with_non_utc_offset(self):
        result = _parse_timestamp("2026-01-01T02:00:00+02:00")
        assert result == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_parses_naive_datetime_assumes_utc(self):
        result = _parse_timestamp("2026-01-01T00:00:00")
        assert result == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_returns_none_for_non_string(self):
        assert _parse_timestamp(12345) is None
        assert _parse_timestamp(None) is None
        assert _parse_timestamp([]) is None

    def test_returns_none_for_empty_string(self):
        assert _parse_timestamp("") is None
        assert _parse_timestamp("   ") is None

    def test_returns_none_for_invalid_iso(self):
        assert _parse_timestamp("not-a-date") is None
        assert _parse_timestamp("2026-13-45T99:99:99Z") is None

    def test_strips_whitespace_before_parsing(self):
        result = _parse_timestamp("  2026-01-01T00:00:00Z  ")
        assert result == datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# _access_token_is_expiring
# ---------------------------------------------------------------------------


class TestAccessTokenIsExpiring:
    def test_returns_true_when_expires_is_none(self):
        assert _access_token_is_expiring(None, 120) is True

    def test_returns_true_when_expires_unparseable(self):
        assert _access_token_is_expiring("not-a-date", 120) is True

    def test_returns_true_when_token_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
        assert _access_token_is_expiring(past, 120) is True

    def test_returns_true_when_within_skew(self):
        soon = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
        assert _access_token_is_expiring(soon, 120) is True

    def test_returns_false_when_well_before_skew(self):
        future = (datetime.now(timezone.utc) + timedelta(seconds=3600)).isoformat()
        assert _access_token_is_expiring(future, 120) is False

    def test_returns_true_at_exact_boundary(self):
        """remaining == skew → returns True (<=)."""
        now = datetime.now(timezone.utc)
        boundary = now + timedelta(seconds=120)
        result = _access_token_is_expiring(boundary.isoformat(), 120)
        # Due to execution time, remaining is slightly < 120, so True
        assert result is True

    def test_zero_skew(self):
        future = (datetime.now(timezone.utc) + timedelta(seconds=3600)).isoformat()
        assert _access_token_is_expiring(future, 0) is False

    def test_zero_skew_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        assert _access_token_is_expiring(past, 0) is True

    def test_negative_skew_clamped_to_zero(self):
        """Negative skew is clamped to 0 via max(0, int(skew))."""
        future = (datetime.now(timezone.utc) + timedelta(seconds=3600)).isoformat()
        assert _access_token_is_expiring(future, -100) is False


# ---------------------------------------------------------------------------
# peek_nous_access_token
# ---------------------------------------------------------------------------


class TestPeekNousAccessToken:
    def test_returns_explicit_env_token(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "env-token")
        assert peek_nous_access_token() == "env-token"

    def test_strips_whitespace_from_env_token(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "  env-token  ")
        assert peek_nous_access_token() == "env-token"

    def test_ignores_empty_env_token(self, clean_env, hermes_home):
        _write_auth_file(hermes_home, {"nous": {"access_token": "cached-tok"}})
        # Empty string env token should be ignored
        with patch.dict(os.environ, {"TOOL_GATEWAY_USER_TOKEN": ""}, clear=False):
            assert peek_nous_access_token() == "cached-tok"

    def test_ignores_whitespace_only_env_token(self, clean_env, hermes_home):
        _write_auth_file(hermes_home, {"nous": {"access_token": "cached-tok"}})
        with patch.dict(os.environ, {"TOOL_GATEWAY_USER_TOKEN": "   "}, clear=False):
            assert peek_nous_access_token() == "cached-tok"

    def test_returns_cached_token_from_auth_store(self, clean_env, hermes_home):
        _write_auth_file(hermes_home, {"nous": {"access_token": "cached-tok"}})
        assert peek_nous_access_token() == "cached-tok"

    def test_strips_whitespace_from_cached_token(self, clean_env, hermes_home):
        _write_auth_file(hermes_home, {"nous": {"access_token": "  cached-tok  "}})
        assert peek_nous_access_token() == "cached-tok"

    def test_returns_none_when_no_token_anywhere(self, clean_env, hermes_home):
        _write_auth_file(hermes_home, {"nous": {}})
        assert peek_nous_access_token() is None

    def test_returns_none_when_no_auth_file(self, clean_env, hermes_home):
        assert peek_nous_access_token() is None

    def test_returns_none_when_cached_token_not_string(self, clean_env, hermes_home):
        _write_auth_file(hermes_home, {"nous": {"access_token": 12345}})
        assert peek_nous_access_token() is None

    def test_returns_none_when_cached_token_empty(self, clean_env, hermes_home):
        _write_auth_file(hermes_home, {"nous": {"access_token": ""}})
        assert peek_nous_access_token() is None

    def test_returns_none_when_cached_token_whitespace_only(
        self, clean_env, hermes_home
    ):
        _write_auth_file(hermes_home, {"nous": {"access_token": "   "}})
        assert peek_nous_access_token() is None

    def test_env_token_takes_precedence_over_cached(
        self, clean_env, hermes_home, monkeypatch
    ):
        _write_auth_file(hermes_home, {"nous": {"access_token": "cached-tok"}})
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "env-token")
        assert peek_nous_access_token() == "env-token"


# ---------------------------------------------------------------------------
# read_nous_access_token
# ---------------------------------------------------------------------------


class TestReadNousAccessToken:
    def test_returns_explicit_env_token(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "env-token")
        assert read_nous_access_token() == "env-token"

    def test_strips_whitespace_from_env_token(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "  env-token  ")
        assert read_nous_access_token() == "env-token"

    def test_returns_cached_token_when_not_expiring(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        future = (datetime.now(timezone.utc) + timedelta(seconds=3600)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "cached-tok", "expires_at": future},
            },
        )
        # Should NOT call refresh
        with patch("hermes_cli.auth.resolve_nous_access_token") as refresh:
            assert read_nous_access_token() == "cached-tok"
            refresh.assert_not_called()

    def test_refreshes_expiring_cached_token(self, clean_env, hermes_home, monkeypatch):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "stale-tok", "expires_at": soon},
            },
        )
        with patch(
            "hermes_cli.auth.resolve_nous_access_token", return_value="fresh-token"
        ) as refresh:
            assert read_nous_access_token() == "fresh-token"
            refresh.assert_called_once_with(refresh_skew_seconds=120)

    def test_strips_refreshed_token(self, clean_env, hermes_home, monkeypatch):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "stale-tok", "expires_at": soon},
            },
        )
        with patch(
            "hermes_cli.auth.resolve_nous_access_token", return_value="  fresh-token  "
        ):
            assert read_nous_access_token() == "fresh-token"

    def test_returns_cached_when_refresh_returns_none(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "stale-tok", "expires_at": soon},
            },
        )
        with patch("hermes_cli.auth.resolve_nous_access_token", return_value=None):
            assert read_nous_access_token() == "stale-tok"

    def test_returns_cached_when_refresh_returns_empty(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "stale-tok", "expires_at": soon},
            },
        )
        with patch("hermes_cli.auth.resolve_nous_access_token", return_value=""):
            assert read_nous_access_token() == "stale-tok"

    def test_returns_cached_when_refresh_returns_non_string(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "stale-tok", "expires_at": soon},
            },
        )
        with patch("hermes_cli.auth.resolve_nous_access_token", return_value=12345):
            assert read_nous_access_token() == "stale-tok"

    def test_returns_cached_when_refresh_raises(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "stale-tok", "expires_at": soon},
            },
        )
        with patch(
            "hermes_cli.auth.resolve_nous_access_token",
            side_effect=RuntimeError("refresh failed"),
        ):
            assert read_nous_access_token() == "stale-tok"

    def test_returns_cached_when_refresh_import_fails(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "stale-tok", "expires_at": soon},
            },
        )
        # Make the import inside read_nous_access_token fail
        import builtins

        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "hermes_cli.auth":
                raise ImportError("module not found")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            assert read_nous_access_token() == "stale-tok"

    def test_returns_none_when_no_cached_and_refresh_returns_none(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        _write_auth_file(hermes_home, {"nous": {}})
        with patch("hermes_cli.auth.resolve_nous_access_token", return_value=None):
            assert read_nous_access_token() is None

    def test_returns_none_when_no_auth_file_and_refresh_returns_none(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        with patch("hermes_cli.auth.resolve_nous_access_token", return_value=None):
            assert read_nous_access_token() is None

    def test_returns_fresh_when_no_cached_but_refresh_succeeds(
        self, clean_env, hermes_home, monkeypatch
    ):
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        _write_auth_file(hermes_home, {"nous": {}})
        with patch(
            "hermes_cli.auth.resolve_nous_access_token", return_value="fresh-token"
        ):
            assert read_nous_access_token() == "fresh-token"

    def test_returns_cached_when_expires_at_missing(
        self, clean_env, hermes_home, monkeypatch
    ):
        """No expires_at → _access_token_is_expiring returns True → refresh attempted."""
        monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
        _write_auth_file(hermes_home, {"nous": {"access_token": "cached-tok"}})
        with patch("hermes_cli.auth.resolve_nous_access_token", return_value=None):
            # Refresh returns None → falls back to cached_token
            assert read_nous_access_token() == "cached-tok"


# ---------------------------------------------------------------------------
# get_tool_gateway_scheme
# ---------------------------------------------------------------------------


class TestGetToolGatewayScheme:
    def test_returns_https_by_default(self, clean_env):
        assert get_tool_gateway_scheme() == "https"

    def test_returns_https_when_set(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "https")
        assert get_tool_gateway_scheme() == "https"

    def test_returns_http_when_set(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "http")
        assert get_tool_gateway_scheme() == "http"

    def test_uppercase_normalised_to_lower(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "HTTPS")
        assert get_tool_gateway_scheme() == "https"

    def test_whitespace_stripped(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "  https  ")
        assert get_tool_gateway_scheme() == "https"

    def test_empty_string_returns_default(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "")
        assert get_tool_gateway_scheme() == "https"

    def test_whitespace_only_returns_default(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "   ")
        assert get_tool_gateway_scheme() == "https"

    def test_invalid_scheme_raises_value_error(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "ftp")
        with pytest.raises(ValueError, match="must be 'http' or 'https'"):
            get_tool_gateway_scheme()

    def test_random_string_raises(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "not-a-scheme")
        with pytest.raises(ValueError, match="must be 'http' or 'https'"):
            get_tool_gateway_scheme()


# ---------------------------------------------------------------------------
# build_vendor_gateway_url
# ---------------------------------------------------------------------------


class TestBuildVendorGatewayUrl:
    def test_uses_vendor_specific_override(self, clean_env, monkeypatch):
        monkeypatch.setenv(
            "FIRECRAWL_GATEWAY_URL", "http://firecrawl-gateway.localhost:3009/"
        )
        assert (
            build_vendor_gateway_url("firecrawl")
            == "http://firecrawl-gateway.localhost:3009"
        )

    def test_strips_trailing_slash_from_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_GATEWAY_URL", "http://fc.example.com///")
        assert build_vendor_gateway_url("firecrawl") == "http://fc.example.com"

    def test_derives_from_shared_domain(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        assert (
            build_vendor_gateway_url("firecrawl")
            == "https://firecrawl-gateway.nousresearch.com"
        )

    def test_derives_from_shared_domain_with_http_scheme(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "internal.local")
        monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "http")
        assert (
            build_vendor_gateway_url("browser-use")
            == "http://browser-use-gateway.internal.local"
        )

    def test_strips_trailing_slash_from_shared_domain(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com/")
        assert (
            build_vendor_gateway_url("firecrawl")
            == "https://firecrawl-gateway.nousresearch.com"
        )

    def test_strips_leading_slash_from_shared_domain(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "/nousresearch.com")
        assert (
            build_vendor_gateway_url("firecrawl")
            == "https://firecrawl-gateway.nousresearch.com"
        )

    def test_uses_default_domain_when_no_override(self, clean_env):
        assert (
            build_vendor_gateway_url("firecrawl")
            == "https://firecrawl-gateway.nousresearch.com"
        )

    def test_vendor_with_hyphen_converted_to_underscore_in_env_key(
        self, clean_env, monkeypatch
    ):
        monkeypatch.setenv("BROWSER_USE_GATEWAY_URL", "http://bu.example.com")
        assert build_vendor_gateway_url("browser-use") == "http://bu.example.com"

    def test_vendor_uppercase_in_env_key(self, clean_env, monkeypatch):
        monkeypatch.setenv("MODAL_GATEWAY_URL", "http://modal.example.com")
        assert build_vendor_gateway_url("modal") == "http://modal.example.com"

    def test_empty_vendor_override_falls_through(self, clean_env, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_GATEWAY_URL", "")
        assert (
            build_vendor_gateway_url("firecrawl")
            == "https://firecrawl-gateway.nousresearch.com"
        )

    def test_whitespace_vendor_override_falls_through(self, clean_env, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_GATEWAY_URL", "   ")
        assert (
            build_vendor_gateway_url("firecrawl")
            == "https://firecrawl-gateway.nousresearch.com"
        )


# ---------------------------------------------------------------------------
# resolve_managed_tool_gateway
# ---------------------------------------------------------------------------


class TestResolveManagedToolGateway:
    def test_derives_vendor_origin_from_shared_domain(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = resolve_managed_tool_gateway(
                "firecrawl",
                token_reader=lambda: "nous-token",
            )
        assert result is not None
        assert result.gateway_origin == "https://firecrawl-gateway.nousresearch.com"
        assert result.nous_user_token == "nous-token"
        assert result.managed_mode is True
        assert result.vendor == "firecrawl"

    def test_uses_vendor_specific_override(self, clean_env, monkeypatch):
        monkeypatch.setenv(
            "BROWSER_USE_GATEWAY_URL", "http://browser-use-gateway.localhost:3009/"
        )
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = resolve_managed_tool_gateway(
                "browser-use",
                token_reader=lambda: "nous-token",
            )
        assert result is not None
        assert result.gateway_origin == "http://browser-use-gateway.localhost:3009"

    def test_returns_none_without_nous_token(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = resolve_managed_tool_gateway(
                "firecrawl",
                token_reader=lambda: None,
            )
        assert result is None

    def test_returns_none_without_subscription(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=False
        ):
            result = resolve_managed_tool_gateway(
                "firecrawl",
                token_reader=lambda: "nous-token",
            )
        assert result is None

    def test_returns_none_when_gateway_origin_empty(self, clean_env, monkeypatch):
        """gateway_builder returns empty string → result is None."""
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = resolve_managed_tool_gateway(
                "firecrawl",
                gateway_builder=lambda v: "",
                token_reader=lambda: "nous-token",
            )
        assert result is None

    def test_returns_none_when_token_empty_string(self, clean_env, monkeypatch):
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = resolve_managed_tool_gateway(
                "firecrawl",
                gateway_builder=lambda v: "https://gw.example.com",
                token_reader=lambda: "",
            )
        assert result is None

    def test_uses_default_gateway_builder_when_none(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = resolve_managed_tool_gateway(
                "firecrawl",
                gateway_builder=None,
                token_reader=lambda: "nous-token",
            )
        assert result is not None
        assert result.gateway_origin == "https://firecrawl-gateway.nousresearch.com"

    def test_uses_default_token_reader_when_none(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "env-token")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = resolve_managed_tool_gateway("firecrawl")
        assert result is not None
        assert result.nous_user_token == "env-token"

    def test_custom_gateway_builder_takes_precedence(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = resolve_managed_tool_gateway(
                "firecrawl",
                gateway_builder=lambda v: "https://custom.example.com",
                token_reader=lambda: "nous-token",
            )
        assert result is not None
        assert result.gateway_origin == "https://custom.example.com"


# ---------------------------------------------------------------------------
# is_managed_tool_gateway_ready
# ---------------------------------------------------------------------------


class TestIsManagedToolGatewayReady:
    def test_returns_true_when_gateway_and_token_present(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "env-token")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            assert is_managed_tool_gateway_ready("modal") is True

    def test_returns_false_when_no_token(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = is_managed_tool_gateway_ready(
                "modal",
                token_reader=lambda: None,
            )
        assert result is False

    def test_returns_false_when_subscription_disabled(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=False
        ):
            assert is_managed_tool_gateway_ready("modal") is False

    def test_uses_peek_token_by_default(self, clean_env, hermes_home, monkeypatch):
        """is_managed_tool_gateway_ready defaults to peek (no refresh)."""
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        expired_at = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        _write_auth_file(
            hermes_home,
            {
                "nous": {"access_token": "expired-token", "expires_at": expired_at},
            },
        )
        refresh_calls = []

        def _record_refresh(*, refresh_skew_seconds=120, **_kwargs):
            refresh_calls.append(refresh_skew_seconds)
            return "fresh-token"

        with (
            patch(
                "tools.managed_tool_gateway.managed_nous_tools_enabled",
                return_value=True,
            ),
            patch("hermes_cli.auth.resolve_nous_access_token", _record_refresh),
        ):
            assert is_managed_tool_gateway_ready("modal") is True

        # peek does NOT trigger refresh
        assert refresh_calls == []

    def test_custom_token_reader_takes_precedence(self, clean_env, monkeypatch):
        monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = is_managed_tool_gateway_ready(
                "modal",
                token_reader=lambda: "custom-token",
            )
        assert result is True

    def test_custom_gateway_builder_takes_precedence(self, clean_env, monkeypatch):
        with patch(
            "tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=True
        ):
            result = is_managed_tool_gateway_ready(
                "modal",
                gateway_builder=lambda v: "https://custom.example.com",
                token_reader=lambda: "token",
            )
        assert result is True


# ---------------------------------------------------------------------------
# ManagedToolGatewayConfig dataclass
# ---------------------------------------------------------------------------


class TestManagedToolGatewayConfig:
    def test_is_frozen(self):
        cfg = ManagedToolGatewayConfig(
            vendor="firecrawl",
            gateway_origin="https://gw.example.com",
            nous_user_token="tok",
            managed_mode=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            cfg.vendor = "changed"  # type: ignore[misc]

    def test_fields_preserved(self):
        cfg = ManagedToolGatewayConfig(
            vendor="firecrawl",
            gateway_origin="https://gw.example.com",
            nous_user_token="tok",
            managed_mode=False,
        )
        assert cfg.vendor == "firecrawl"
        assert cfg.gateway_origin == "https://gw.example.com"
        assert cfg.nous_user_token == "tok"
        assert cfg.managed_mode is False
