"""Tests for tools/mcp_oauth_device.py — RFC 8628 device-code flow for MCP OAuth."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from tools.mcp_oauth_device import (
    DEFAULT_POLL_INTERVAL,
    DEVICE_CODE_GRANT,
    DeviceAuthorization,
    DeviceFlowError,
    poll_device_token,
    register_device_client,
    request_device_authorization,
    server_supports_device_flow,
)


def _device_response(**overrides):
    payload = {
        "device_code": "dc_123",
        "user_code": "ABCD-1234",
        "verification_uri": "https://auth.example/activate",
        "verification_uri_complete": "https://auth.example/activate?code=ABCD-1234",
        "expires_in": 900,
        "interval": 5,
    }
    payload.update(overrides)
    return payload


def _sdk_available() -> bool:
    try:
        from mcp.client import auth as _client_auth  # noqa: F401
        from mcp.shared import auth as _shared_auth  # noqa: F401
        return all(hasattr(_shared_auth, name) for name in
                   ("OAuthClientInformationFull", "OAuthToken"))
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# request_device_authorization
# ---------------------------------------------------------------------------

class TestRequestDeviceAuthorization:
    def test_parses_full_response(self):
        calls = []

        def fake_post(url, data):
            calls.append((url, data))
            return _device_response()

        auth = request_device_authorization(
            "https://auth.example/device", "client_1",
            scope="mcp:read", resource="https://mcp.example/mcp", http_post=fake_post)
        assert auth.device_code == "dc_123"
        assert auth.user_code == "ABCD-1234"
        assert auth.verification_uri_complete == "https://auth.example/activate?code=ABCD-1234"
        assert auth.expires_in == 900
        assert auth.interval == 5
        url, data = calls[0]
        assert url == "https://auth.example/device"
        assert data["client_id"] == "client_1"
        assert data["scope"] == "mcp:read"
        assert data["resource"] == "https://mcp.example/mcp"

    def test_defaults_interval_and_expiry(self):
        response = {"device_code": "d", "user_code": "u", "verification_uri": "https://x/y"}
        auth = request_device_authorization(
            "https://auth.example/device", "c", http_post=lambda u, d: response)
        assert auth.interval == DEFAULT_POLL_INTERVAL
        assert auth.expires_in == 1800
        assert auth.verification_uri_complete is None

    def test_refusal_names_error_code(self):
        def fake_post(url, data):
            return {"error": "unauthorized_client",
                    "error_description": "client not registered for device authorization"}

        with pytest.raises(DeviceFlowError, match="unauthorized_client"):
            request_device_authorization("https://auth.example/device", "c", http_post=fake_post)

    def test_missing_fields_rejected(self):
        with pytest.raises(DeviceFlowError, match="missing"):
            request_device_authorization(
                "https://auth.example/device", "c",
                http_post=lambda u, d: {"user_code": "u", "verification_uri": "https://x/y"})


# ---------------------------------------------------------------------------
# poll_device_token
# ---------------------------------------------------------------------------

class TestPollDeviceToken:
    def test_immediate_success_never_sleeps(self):
        sleeps = []
        tokens = poll_device_token(
            "https://auth.example/token", "c", "dc_123",
            http_post=lambda u, d: {"access_token": "at_1", "token_type": "Bearer"},
            sleep=sleeps.append)
        assert tokens["access_token"] == "at_1"
        assert sleeps == []

    def test_pending_then_success_uses_interval(self):
        responses = [
            {"error": "authorization_pending", "error_description": "waiting"},
            {"error": "authorization_pending"},
            {"access_token": "at_2", "refresh_token": "rt_2", "expires_in": 899},
        ]
        sleeps = []
        tokens = poll_device_token(
            "https://auth.example/token", "c", "dc_123", interval=5,
            http_post=lambda u, d: responses.pop(0), sleep=sleeps.append)
        assert tokens["refresh_token"] == "rt_2"
        assert sleeps == [5, 5]

    def test_slow_down_increases_interval(self):
        responses = [{"error": "slow_down"}, {"access_token": "at_3"}]
        sleeps = []
        poll_device_token(
            "https://auth.example/token", "c", "dc_123", interval=5,
            http_post=lambda u, d: responses.pop(0), sleep=sleeps.append)
        assert sleeps == [10]

    def test_denial_is_terminal(self):
        calls = []

        def fake_post(url, data):
            calls.append(data)
            return {"error": "access_denied", "error_description": "user refused"}

        with pytest.raises(DeviceFlowError, match="access_denied"):
            poll_device_token("https://auth.example/token", "c", "dc_123",
                             http_post=fake_post, sleep=lambda s: None)
        assert len(calls) == 1  # no retry after a terminal error

    def test_expiry_without_approval(self):
        def fail_if_called(url, data):  # pragma: no cover — must never be reached
            raise AssertionError("no HTTP once the code has expired")

        with pytest.raises(DeviceFlowError, match="expired"):
            poll_device_token("https://auth.example/token", "c", "dc_123",
                             expires_in=600, timeout=0,
                             http_post=fail_if_called, sleep=lambda s: None)

    def test_unknown_error_surfaces(self):
        with pytest.raises(DeviceFlowError, match="server_broke"):
            poll_device_token(
                "https://auth.example/token", "c", "dc_123",
                http_post=lambda u, d: {"error": "server_broke"},
                sleep=lambda s: None)


# ---------------------------------------------------------------------------
# register_device_client
# ---------------------------------------------------------------------------

class TestRegisterDeviceClient:
    def test_requests_device_grant(self):
        seen = {}

        def fake_post(url, payload):
            seen.update(payload)
            return {"client_id": "lmo_client_1", "redirect_uris": []}

        info = register_device_client(
            "https://auth.example/register", client_name="Hermes Agent",
            scope="mcp:read", http_post=fake_post)
        assert info["client_id"] == "lmo_client_1"
        assert DEVICE_CODE_GRANT in seen["grant_types"]
        assert "authorization_code" in seen["grant_types"]
        assert seen["scope"] == "mcp:read"
        assert seen["token_endpoint_auth_method"] == "none"

    def test_missing_client_id_rejected(self):
        with pytest.raises(DeviceFlowError, match="no client_id"):
            register_device_client(
                "https://auth.example/register",
                http_post=lambda u, p: {"oops": True})


# ---------------------------------------------------------------------------
# server_supports_device_flow
# ---------------------------------------------------------------------------

class TestServerSupportsDeviceFlow:
    def test_dict_with_endpoint(self):
        assert server_supports_device_flow(
            {"device_authorization_endpoint": "https://auth.example/device"}) is True

    def test_dict_without_endpoint(self):
        assert server_supports_device_flow({"token_endpoint": "https://x/t"}) is False

    def test_model_with_endpoint(self):
        model = types.SimpleNamespace(device_authorization_endpoint="https://auth.example/device")
        assert server_supports_device_flow(model) is True

    def test_model_without_endpoint(self):
        assert server_supports_device_flow(types.SimpleNamespace()) is False


# ---------------------------------------------------------------------------
# Contract: the grant string is the RFC 8628 value servers match on.
# ---------------------------------------------------------------------------

def test_device_grant_is_rfc8628_value():
    assert DEVICE_CODE_GRANT == "urn:ietf:params:oauth:grant-type:device_code"


# ---------------------------------------------------------------------------
# Endpoint extraction: the device endpoint comes from the raw ASM document
# because the pinned SDK model drops it.
# ---------------------------------------------------------------------------

class TestDeviceEndpointsFrom:
    def _asm(self):
        return types.SimpleNamespace(
            token_endpoint="https://auth.example/oauth/token",
            registration_endpoint="https://auth.example/oauth/register")

    def test_reads_device_endpoint_from_raw_document(self):
        from tools.mcp_oauth_device import _device_endpoints_from
        endpoints = _device_endpoints_from(
            {"device_authorization_endpoint": "https://auth.example/oauth/device"},
            self._asm())
        assert endpoints is not None
        assert endpoints["device_authorization_endpoint"] == "https://auth.example/oauth/device"
        assert endpoints["token_endpoint"] == "https://auth.example/oauth/token"

    def test_none_when_device_endpoint_absent(self):
        from tools.mcp_oauth_device import _device_endpoints_from
        assert _device_endpoints_from({}, self._asm()) is None


# ---------------------------------------------------------------------------
# Discovery-shim contract: the SDK handlers read status_code + aread() bytes.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _sdk_available(), reason="MCP OAuth SDK types not installed")
class TestDiscoveryShim:
    def test_prm_handler_accepts_shim(self):
        import asyncio
        from mcp.client.auth.utils import handle_protected_resource_response
        from tools.mcp_oauth_device import _adapt_metadata_response

        payload = {
            "resource": "https://mcp.example/mcp",
            "authorization_servers": ["https://auth.example/oauth"],
        }
        prm = asyncio.run(_adapt_metadata_response(payload, 200, handle_protected_resource_response))
        assert prm is not None
        assert str(prm.resource) == "https://mcp.example/mcp"

    def test_asm_handler_accepts_shim(self):
        import asyncio
        from mcp.client.auth.utils import handle_auth_metadata_response
        from tools.mcp_oauth_device import _adapt_metadata_response

        payload = {
            "issuer": "https://auth.example/oauth",
            "authorization_endpoint": "https://auth.example/oauth/authorize",
            "token_endpoint": "https://auth.example/oauth/token",
            "registration_endpoint": "https://auth.example/oauth/register",
            "response_types_supported": ["code"],
        }
        ok, asm = asyncio.run(_adapt_metadata_response(payload, 200, handle_auth_metadata_response))
        assert ok is True
        assert asm is not None
        assert str(asm.token_endpoint) == "https://auth.example/oauth/token"


# ---------------------------------------------------------------------------
# Persistence round-trip through the real storage (temp HERMES_HOME, real SDK).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _sdk_available(), reason="MCP OAuth SDK types not installed")
class TestPersistDeviceState:
    def test_roundtrip(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import HermesTokenStorage
        from tools.mcp_oauth_device import persist_device_state
        import asyncio

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        persist_device_state(
            "device-server",
            {"client_id": "c1", "redirect_uris": [],
             "grant_types": ["urn:ietf:params:oauth:grant-type:device_code"],
             "response_types": ["code"], "token_endpoint_auth_method": "none"},
            {"access_token": "at_9", "token_type": "Bearer", "expires_in": 899},
            hermes_home=tmp_path)
        stored = asyncio.run(HermesTokenStorage("device-server", hermes_home=tmp_path).get_tokens())
        assert stored is not None
