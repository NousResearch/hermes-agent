"""Tests for tools/managed_tool_gateway.py — Nous-hosted vendor passthrough helpers.

Covers: auth_json_path, _read_nous_provider_state, _parse_timestamp,
_access_token_is_expiring, peek_nous_access_token, read_nous_access_token,
get_tool_gateway_scheme, build_vendor_gateway_url,
resolve_managed_tool_gateway, is_managed_tool_gateway_ready.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import managed_tool_gateway
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


def test_managed_vendor_endpoints_pin_the_deployed_gateway_url():
    """The exact URL an agent may connect to is a code fact, not a lookup.

    Exercises the real ``build_vendor_gateway_url`` (which once resolved a
    typo'd pseudo-vendor to a non-existent host while every other test stubbed
    it): default builder, real deployed host, pinned vendor path.
    """
    with patch.dict(
        os.environ,
        {"TOOL_GATEWAY_DOMAIN": "nousresearch.com", "TOOL_GATEWAY_SCHEME": "https"},
        clear=False,
    ):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        endpoints = managed_tool_gateway.managed_vendor_endpoints("vendorx")

    assert endpoints == {
        "origin": "https://tool-gateway.nousresearch.com",
        "base_url": "https://tool-gateway.nousresearch.com/api/vendorx",
        "upload_path": "/api/uploads/vendorx",
    }


def test_managed_vendor_endpoints_do_not_consult_entitlement():
    """Address resolution, not a policy decision.

    What an account may spend is the gateway's ruling, stated in its refusals.
    Guessing at it here would hide the address from a caller the server would
    have served, so entitlement must not be read on this path at all.
    """
    with patch.dict(os.environ, {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"}, clear=False), \
         patch.object(
             managed_tool_gateway,
             "managed_nous_tools_enabled",
             side_effect=AssertionError("entitlement must not gate address resolution"),
         ):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        endpoints = managed_tool_gateway.managed_vendor_endpoints("vendorx")

    assert endpoints is not None
    assert endpoints["base_url"] == "https://tool-gateway.nousresearch.com/api/vendorx"


def test_managed_vendor_endpoints_are_none_when_no_origin_resolves():
    # A misconfigured scheme leaves nothing to call, and the caller reports
    # that rather than building a URL out of a broken setting.
    with patch.dict(os.environ, {"TOOL_GATEWAY_SCHEME": "ftp"}, clear=False):
        os.environ.pop("TOOL_GATEWAY_URL", None)
        assert managed_tool_gateway.managed_vendor_endpoints("vendorx") is None


def test_managed_vendor_endpoints_are_none_when_builder_returns_empty_origin():
    assert managed_tool_gateway.managed_vendor_endpoints(
        "vendorx", gateway_builder=lambda _vendor: ""
    ) is None


@pytest.mark.parametrize("url", [None, "", "   ", 42])
def test_managed_gateway_url_rejects_non_urls(url):
    assert managed_tool_gateway.is_managed_nous_gateway_url(url) is False


def test_managed_gateway_url_rejects_invalid_builder_url():
    assert managed_tool_gateway.is_managed_nous_gateway_url(
        "https://tool-gateway.example.com/api/vendorx",
        gateway_builder=lambda _vendor: "https://[invalid",
    ) is False


def test_managed_gateway_auth_headers_carry_the_bearer():
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        headers = managed_tool_gateway.managed_gateway_auth_headers(
            "https://tool-gateway.example.com/api/vendorx/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: "nous-token",
        )

    assert headers == {"Authorization": "Bearer nous-token"}


def test_managed_gateway_auth_headers_reflect_a_rotated_token():
    # Read fresh on every call: a Nous access token expires within the hour,
    # and a long session must not keep presenting a dead bearer.
    tokens = iter(["first-token", "second-token"])
    builder = lambda vendor: f"https://{vendor}-gateway.example.com"
    url = "https://tool-gateway.example.com/api/vendorx/generations"

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
            "https://attacker.example/api/vendorx/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: "nous-token",
        ) == {}


def test_managed_gateway_auth_headers_empty_without_a_token():
    # Empty rather than raising, so a caller can say "sign in" instead of
    # sending an unauthenticated request.
    with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        assert managed_tool_gateway.managed_gateway_auth_headers(
            "https://tool-gateway.example.com/api/vendorx/generations",
            gateway_builder=lambda vendor: f"https://{vendor}-gateway.example.com",
            token_reader=lambda: None,
        ) == {}


class TestManagedMediaUploader:
    """The presign -> PUT -> ``nous-upload:<token>`` protocol.

    This is the only way a local image or video reaches a managed vendor, and
    the pieces it gets right are not incidental: the presigned URL signs the
    content type and byte length, so a PUT that disagrees with the presign is
    rejected by storage rather than by us.
    """

    GATEWAY = "https://tool-gateway.example.com"
    BASE_URL = f"{GATEWAY}/api/vendorx"
    UPLOAD_PATH = "/api/uploads/vendorx"

    def _uploader(self, **kwargs):
        return managed_tool_gateway.build_managed_media_uploader(
            kwargs.pop("server_url", self.BASE_URL),
            kwargs.pop("upload_path", self.UPLOAD_PATH),
            gateway_builder=lambda vendor: self.GATEWAY,
            token_reader=kwargs.pop("token_reader", lambda: "nous-token"),
        )

    @staticmethod
    def _response(status_code=200, payload=None):
        class _R:
            def __init__(self):
                self.status_code = status_code

            def json(self):
                if payload is None:
                    raise ValueError("no json")
                return payload

        return _R()

    def _run(self, uploader, data=b"bytes", mime="image/png", presign=None, put=None):
        """Drive one upload with both HTTP legs stubbed; returns the calls made."""
        import httpx

        from tools import url_safety

        calls = {"presign": [], "put": []}
        presign = presign if presign is not None else self._response(
            200, {"uploadUrl": "https://storage.example/put?sig=abc", "token": "tok-1"}
        )
        put = put if put is not None else self._response(200)

        class _PresignClient:
            def __init__(self, **_kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def post(self, url, headers=None, json=None):
                calls["presign"].append({"url": url, "headers": headers, "json": json})
                return presign

        class _PutClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def put(self, url, content=None, headers=None):
                calls["put"].append({"url": url, "content": content, "headers": headers})
                return put

        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True), \
                patch.object(httpx, "AsyncClient", _PresignClient), \
                patch.object(url_safety, "create_ssrf_safe_async_client", lambda **_kw: _PutClient()):
            calls["result"] = asyncio.run(uploader(data, mime))
        return calls

    def test_presign_declares_the_exact_type_and_length_the_put_then_sends(self):
        # Storage validates the PUT against what was signed, so a mismatch
        # between these two is a rejection with no useful error.
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        data = b"\x89PNG\r\n\x1a\n" + b"payload" * 100

        calls = self._run(uploader, data=data, mime="image/png")

        assert calls["presign"][0]["url"] == f"{self.GATEWAY}{self.UPLOAD_PATH}"
        assert calls["presign"][0]["json"] == {
            "contentType": "image/png",
            "contentLength": len(data),
        }
        assert calls["presign"][0]["headers"]["Authorization"] == "Bearer nous-token"
        assert calls["put"][0]["url"] == "https://storage.example/put?sig=abc"
        assert calls["put"][0]["content"] == data
        assert calls["put"][0]["headers"] == {"Content-Type": "image/png"}
        assert calls["result"] == "nous-upload:tok-1"

    def test_the_bytes_go_to_storage_and_never_through_the_gateway(self):
        # The whole point of presigning is that the gateway's request-size
        # ceiling does not apply to a 50MB clip.
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()

        calls = self._run(uploader, data=b"v" * 4096, mime="video/mp4")

        assert len(calls["presign"]) == 1 and len(calls["put"]) == 1
        assert self.GATEWAY not in calls["put"][0]["url"]
        assert calls["presign"][0]["json"]["contentType"] == "video/mp4"

    def test_no_uploader_when_the_url_is_not_a_managed_gateway(self):
        # Refusing to build is what makes the caller say "pass a URL instead"
        # rather than forwarding a raw local path to a third party.
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            assert self._uploader(server_url="https://attacker.example/api/vendorx") is None

    @pytest.mark.parametrize("upload_path", [None, "", "api/uploads/vendorx", 42])
    def test_no_uploader_without_a_rooted_upload_path(self, upload_path):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            assert self._uploader(upload_path=upload_path) is None

    def test_a_missing_credential_fails_before_any_request(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()

        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True), \
                patch.object(managed_tool_gateway, "managed_gateway_auth_headers", return_value={}):
            with pytest.raises(RuntimeError, match="no Nous credential"):
                asyncio.run(uploader(b"x", "image/png"))

    def test_a_gateway_refusal_surfaces_its_own_message(self):
        # Quota and size refusals carry guidance written for the model; a bare
        # status code would throw that away.
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()
        refusal = self._response(
            413, {"error": {"message": "That file is 82MB; the limit for video is 50MB."}}
        )

        with pytest.raises(RuntimeError, match="the limit for video is 50MB"):
            self._run(uploader, presign=refusal)

    def test_an_unreadable_refusal_still_reports_the_status(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()

        with pytest.raises(RuntimeError, match="HTTP 502"):
            self._run(uploader, presign=self._response(502, None))

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"uploadUrl": "https://storage.example/put"},
            {"token": "tok-1"},
            {"uploadUrl": "", "token": "tok-1"},
            {"uploadUrl": "https://storage.example/put", "token": ""},
        ],
    )
    def test_a_malformed_presign_response_is_refused_rather_than_guessed(self, payload):
        # Half a presign must not become a PUT to nowhere or an empty token
        # that later reads as a valid reference.
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()

        with pytest.raises(RuntimeError, match="malformed"):
            self._run(uploader, presign=self._response(200, payload))

    def test_an_unreadable_success_response_is_refused_as_malformed(self):
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()

        with pytest.raises(RuntimeError, match="malformed"):
            self._run(uploader, presign=self._response(200, None))

    def test_a_storage_rejection_is_not_reported_as_a_successful_upload(self):
        # A signature mismatch answers non-200 with an XML body; returning a
        # token here would hand the vendor a reference to nothing.
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()

        with pytest.raises(RuntimeError, match="storage refused the upload"):
            self._run(uploader, put=self._response(403))


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


def test_user_token_override_falls_back_to_env_when_scope_is_uninstalled(
    clean_env, monkeypatch
):
    from agent.secret_scope import UnscopedSecretError

    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "env-token")
    with patch(
        "agent.secret_scope.get_secret",
        side_effect=UnscopedSecretError,
    ):
        assert managed_tool_gateway._read_user_token_override() == "env-token"


def test_user_token_override_falls_back_to_env_when_scope_import_fails(
    clean_env, monkeypatch
):
    import builtins

    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "agent.secret_scope":
            raise ImportError("module not found")
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "env-token")
    with patch("builtins.__import__", side_effect=failing_import):
        assert managed_tool_gateway._read_user_token_override() == "env-token"


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
