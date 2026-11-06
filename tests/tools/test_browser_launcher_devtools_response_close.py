"""The launcher closes its response while validating a live Chromium endpoint."""

import io
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("body,expected", [
    (b'{"webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/test"}', True),
    (b"invalid json", False),
])
def test_devtools_endpoint_probe_closes_response(monkeypatch, body, expected):
    from tools.browser_launcher import _devtools_endpoint_live

    response = requests.Response()
    response.status_code = 200
    response._content = body
    response.raw = io.BytesIO(body)
    response.close = Mock(wraps=response.close)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)

    assert _devtools_endpoint_live(9222, "/devtools/browser/test") is expected
    response.close.assert_called_once_with()
