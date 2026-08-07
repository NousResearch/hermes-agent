import asyncio
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
    with patch.dict(
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
        endpoints = managed_tool_gateway.managed_vendor_endpoints("bfl")

    assert endpoints is not None
    assert endpoints["base_url"] == "https://tool-gateway.nousresearch.com/api/bfl"


def test_managed_vendor_endpoints_are_none_when_no_origin_resolves():
    # A misconfigured scheme leaves nothing to call, and the caller reports
    # that rather than building a URL out of a broken setting.
    with patch.dict(os.environ, {"TOOL_GATEWAY_SCHEME": "ftp"}, clear=False):
        os.environ.pop("TOOL_GATEWAY_URL", None)
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


class TestManagedMediaUploader:
    """The presign -> PUT -> ``nous-upload:<token>`` protocol.

    This is the only way a local image or video reaches a managed vendor, and
    the pieces it gets right are not incidental: the presigned URL signs the
    content type and byte length, so a PUT that disagrees with the presign is
    rejected by storage rather than by us.
    """

    GATEWAY = "https://tool-gateway.example.com"
    BASE_URL = f"{GATEWAY}/api/bfl"
    UPLOAD_PATH = "/api/uploads/bfl"

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
            assert self._uploader(server_url="https://attacker.example/api/bfl") is None

    @pytest.mark.parametrize("upload_path", [None, "", "api/uploads/bfl", 42])
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

    def test_a_storage_rejection_is_not_reported_as_a_successful_upload(self):
        # A signature mismatch answers non-200 with an XML body; returning a
        # token here would hand the vendor a reference to nothing.
        with patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
            uploader = self._uploader()

        with pytest.raises(RuntimeError, match="storage refused the upload"):
            self._run(uploader, put=self._response(403))


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
