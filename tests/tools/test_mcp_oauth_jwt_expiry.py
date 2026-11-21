"""MCP OAuth access-token lifetime is bounded by the JWT's own ``exp`` claim.

Regression for #149017: a token endpoint answering ``expires_in: 3600`` for a JWT whose
``exp - iat`` is 900 had Hermes treat the token as valid for an hour. At +15 min the resource
server 401s while ``is_token_valid()`` still says True, the SDK's 401 branch skips the refresh
token for the authorization-code flow, and a headless server is parked until the stored expiry.

Contract: the lifetime Hermes keeps (in memory and on disk) is
``min(expires_in, exp - now - margin)``; it is never lengthened; a missing ``expires_in`` is
derived from ``exp``; opaque or malformed tokens keep the pre-fix behaviour; and a tokens file
written before the fix is clamped when loaded.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")

# Seconds of slack for wall-clock drift between the code under test and the assertions.
_SLACK = 5


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _jwt(claims) -> str:
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    return f"{header}.{_b64url(json.dumps(claims).encode())}.{_b64url(b'not-a-real-signature')}"


def _margin() -> int:
    from tools.mcp_oauth import JWT_EXP_SAFETY_MARGIN_SECONDS

    return JWT_EXP_SAFETY_MARGIN_SECONDS


def _store(tmp_path, monkeypatch, **token_fields) -> dict:
    """Persist an OAuthToken through HermesTokenStorage.set_tokens; return the on-disk JSON."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from mcp.shared.auth import OAuthToken

    from tools.mcp_oauth import HermesTokenStorage

    token = OAuthToken(token_type="Bearer", refresh_token="r", **token_fields)
    asyncio.run(HermesTokenStorage("srv").set_tokens(token))
    return json.loads((tmp_path / "mcp-tokens" / "srv.json").read_text(encoding="utf-8"))


class TestSetTokensBoundsLifetimeByJwtExp:
    def test_exp_sooner_than_expires_in_is_clamped(self, tmp_path, monkeypatch):
        now = time.time()
        on_disk = _store(tmp_path, monkeypatch, access_token=_jwt({"iat": int(now), "exp": int(now) + 900}),
                         expires_in=3600)
        expected = int(now) + 900 - _margin()
        assert expected - _SLACK <= on_disk["expires_at"] <= expected + _SLACK

    def test_exp_later_than_expires_in_is_not_lengthened(self, tmp_path, monkeypatch):
        now = time.time()
        on_disk = _store(tmp_path, monkeypatch, access_token=_jwt({"exp": int(now) + 7200}), expires_in=3600)
        assert now + 3600 - _SLACK <= on_disk["expires_at"] <= now + 3600 + _SLACK

    def test_missing_expires_in_is_derived_from_exp(self, tmp_path, monkeypatch):
        now = time.time()
        on_disk = _store(tmp_path, monkeypatch, access_token=_jwt({"exp": int(now) + 900}))
        expected = int(now) + 900 - _margin()
        assert expected - _SLACK <= on_disk["expires_at"] <= expected + _SLACK

    def test_opaque_token_is_unchanged(self, tmp_path, monkeypatch):
        now = time.time()
        on_disk = _store(tmp_path, monkeypatch, access_token="opaque-access-token", expires_in=3600)
        assert now + 3600 - _SLACK <= on_disk["expires_at"] <= now + 3600 + _SLACK

    def test_opaque_token_without_expires_in_records_no_expiry(self, tmp_path, monkeypatch):
        on_disk = _store(tmp_path, monkeypatch, access_token="opaque-access-token")
        assert "expires_at" not in on_disk

    @pytest.mark.parametrize("access_token", [
        "a.b.c",  # payload is not valid base64url JSON
        f"x.{_b64url(b'not json')}.y",
        f"x.{_b64url(bytes([0xff, 0xfe]))}.y",  # not UTF-8
        _jwt(["exp", 1]),  # payload is not an object
        _jwt({"exp": "9999999999"}),  # string exp
        _jwt({"exp": True}),  # bool is not a timestamp
        _jwt({"exp": None}),
        _jwt({"sub": "no-exp"}),
        "only.two",
        "a.b.c.d.e",  # JWE compact form: claims are encrypted, nothing to read
    ])
    def test_malformed_jwt_is_unchanged(self, tmp_path, monkeypatch, access_token):
        now = time.time()
        on_disk = _store(tmp_path, monkeypatch, access_token=access_token, expires_in=3600)
        assert now + 3600 - _SLACK <= on_disk["expires_at"] <= now + 3600 + _SLACK

    def test_already_expired_jwt_is_stored_as_expired(self, tmp_path, monkeypatch):
        now = time.time()
        on_disk = _store(tmp_path, monkeypatch, access_token=_jwt({"exp": int(now) - 60}), expires_in=3600)
        assert on_disk["expires_at"] <= now + _SLACK


class TestGetTokensClampsPreFixFile:
    def _write(self, tmp_path, monkeypatch, payload: dict):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "srv.json").write_text(json.dumps(payload), encoding="utf-8")
        from tools.mcp_oauth import HermesTokenStorage

        return asyncio.run(HermesTokenStorage("srv").get_tokens())

    def test_stored_expires_at_later_than_exp_is_clamped_on_load(self, tmp_path, monkeypatch):
        """A file written before the fix recorded ``now + expires_in`` regardless of ``exp``."""
        now = time.time()
        tokens = self._write(tmp_path, monkeypatch, {
            "access_token": _jwt({"exp": int(now) + 900}), "token_type": "Bearer",
            "refresh_token": "r", "expires_in": 3600, "expires_at": now + 3600,
        })
        expected = 900 - _margin()
        assert expected - _SLACK <= tokens.expires_in <= expected + _SLACK

    def test_stored_expires_at_sooner_than_exp_is_kept(self, tmp_path, monkeypatch):
        now = time.time()
        tokens = self._write(tmp_path, monkeypatch, {
            "access_token": _jwt({"exp": int(now) + 7200}), "token_type": "Bearer",
            "refresh_token": "r", "expires_in": 3600, "expires_at": now + 600,
        })
        assert 600 - _SLACK <= tokens.expires_in <= 600

    def test_file_without_any_expiry_is_derived_from_exp(self, tmp_path, monkeypatch):
        now = time.time()
        tokens = self._write(tmp_path, monkeypatch, {
            "access_token": _jwt({"exp": int(now) + 900}), "token_type": "Bearer", "refresh_token": "r",
        })
        expected = 900 - _margin()
        assert expected - _SLACK <= tokens.expires_in <= expected + _SLACK


@pytest.mark.asyncio
async def test_refresh_response_seeds_live_expiry_from_jwt_exp(tmp_path, monkeypatch):
    """The in-memory expiry the SDK's ``is_token_valid()`` reads must be clamped too, not just disk:
    otherwise the running process keeps sending the dead token until ``expires_in`` elapses."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from mcp.shared.auth import OAuthToken

    from tests.tools.test_mcp_oauth_manager import _provider_with_token_endpoint

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch)
    provider.context.current_tokens = OAuthToken(
        access_token="at-1", token_type="Bearer", expires_in=3600, refresh_token="rt")
    provider.context.client_info = SimpleNamespace(client_id="cid")

    now = time.time()
    body = json.dumps({"access_token": _jwt({"iat": int(now), "exp": int(now) + 900}),
                       "token_type": "Bearer", "expires_in": 3600}).encode()
    response = MagicMock(status_code=200, next_request=None,
                         request=SimpleNamespace(url="https://idp.example.com/oauth/token"))

    async def _aread():
        return body

    response.aread = _aread
    assert await provider._handle_refresh_response(response)

    expected = int(now) + 900 - _margin()
    assert expected - _SLACK <= provider.context.token_expiry_time <= expected + _SLACK
