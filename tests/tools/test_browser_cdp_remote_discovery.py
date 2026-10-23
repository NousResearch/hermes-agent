"""Remote CDP discovery must not send clients into their own container."""

from unittest.mock import Mock, patch

import pytest

from tools.browser_session_cdp import resolve_cdp_override


@pytest.mark.parametrize(
    ("endpoint", "advertised", "expected"),
    [
        ("http://browser-service:9222", "ws://127.0.0.1:9223", "ws://browser-service:9222"),
        ("http://browser-service:9222", "ws://localhost:9223", "ws://browser-service:9222"),
        ("http://browser-service:9222", "ws://[::1]:9223", "ws://browser-service:9222"),
        ("http://browser-service:9222", "ws://127.2.3.4:9223", "ws://browser-service:9222"),
        ("https://browser.example", "ws://127.0.0.1:9223", "wss://browser.example"),
        ("https://browser.example:9443/json/version", "ws://localhost:9223", "wss://browser.example:9443"),
        ("http://[2001:db8::1]:9222", "ws://[::1]:9223", "ws://[2001:db8::1]:9222"),
        ("ws://browser-service:9222", "ws://127.0.0.1:9223", "ws://browser-service:9222"),
        ("wss://browser.example:9443", "ws://localhost:9223", "wss://browser.example:9443"),
        ("http://browser-service:9222", "wss://localhost:9443", "wss://browser-service:9222"),
    ],
)
def test_remote_discovery_rebases_loopback_websocket_to_configured_authority(endpoint, advertised, expected):
    suffix = "/devtools/browser/browser-id?token=keep-me"
    response = Mock()
    response.json.return_value = {"webSocketDebuggerUrl": advertised + suffix}
    with patch("requests.get", return_value=response):
        resolved = resolve_cdp_override(endpoint)

    assert resolved == expected + suffix


@pytest.mark.parametrize(
    ("endpoint", "advertised"),
    [
        ("http://localhost:9222", "ws://127.0.0.1:9223/devtools/browser/id"),
        ("http://127.0.0.1:9222", "ws://localhost:9223/devtools/browser/id"),
        ("http://[::1]:9222", "ws://[::1]:9223/devtools/browser/id"),
        ("https://discovery.example", "wss://provider.example:9443/devtools/browser/id?token=keep"),
        ("http://browser-service:9222", "ws://10.0.0.8:9223/devtools/browser/id"),
        ("ws://127.0.0.1:9223/devtools/browser/explicit", None),
    ],
)
def test_preserves_local_discovery_provider_routes_and_explicit_websockets(endpoint, advertised):
    response = Mock()
    response.json.return_value = {"webSocketDebuggerUrl": advertised}
    with patch("requests.get", return_value=response) as get:
        resolved = resolve_cdp_override(endpoint)

    assert resolved == (advertised or endpoint)
    if advertised is None:
        get.assert_not_called()


def test_malformed_advertised_websocket_falls_back_to_configured_endpoint():
    endpoint = "http://browser-service:9222"
    response = Mock()
    response.json.return_value = {"webSocketDebuggerUrl": "ws://[invalid/devtools/browser/id"}
    with patch("requests.get", return_value=response):
        assert resolve_cdp_override(endpoint) == endpoint
