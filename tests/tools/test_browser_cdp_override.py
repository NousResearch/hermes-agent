from unittest.mock import Mock, patch


class TestResolveCdpOverride:
    def test_keeps_full_devtools_websocket_url(self):
        from tools.browser_tool import _resolve_cdp_override

        url = "ws://10.0.0.68:9223/devtools/browser/abc123"
        assert _resolve_cdp_override(url) == url

    def test_resolves_http_discovery_endpoint_to_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "webSocketDebuggerUrl": "ws://10.0.0.68:9223/devtools/browser/abc123"
        }

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override("http://10.0.0.68:9223")

        assert resolved == "ws://10.0.0.68:9223/devtools/browser/abc123"
        mock_get.assert_called_once_with("http://10.0.0.68:9223/json/version", timeout=10)

    def test_resolves_bare_ws_hostport_to_discovery_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "webSocketDebuggerUrl": "ws://10.0.0.68:9223/devtools/browser/abc123"
        }

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override("ws://10.0.0.68:9223")

        assert resolved == "ws://10.0.0.68:9223/devtools/browser/abc123"
        mock_get.assert_called_once_with("http://10.0.0.68:9223/json/version", timeout=10)

    def test_falls_back_to_raw_url_when_discovery_fails(self):
        from tools.browser_tool import _resolve_cdp_override

        with patch("tools.browser_tool.requests.get", side_effect=RuntimeError("boom")):
            assert _resolve_cdp_override("http://10.0.0.68:9223") == "http://10.0.0.68:9223"
