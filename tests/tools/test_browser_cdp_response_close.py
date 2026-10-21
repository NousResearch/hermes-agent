"""Response lifetime regression tests."""

import io
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("status,body,expected", [
    (200, b'{"webSocketDebuggerUrl": "ws://localhost:9222/devtools/browser/test"}',
     "ws://localhost:9222/devtools/browser/test"),
    (503, b"{}", "http://localhost:9222"),
    (200, b"invalid json", "http://localhost:9222"),
])
def test_discovery_closes_response(monkeypatch, status, body, expected):
    from tools.browser_tool_cdp import _resolve_cdp_override

    response = requests.Response()
    response.status_code = status
    response._content = body
    response.raw = io.BytesIO(body)
    response.close = Mock(wraps=response.close)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)

    assert _resolve_cdp_override("http://localhost:9222") == expected
    response.close.assert_called_once_with()
