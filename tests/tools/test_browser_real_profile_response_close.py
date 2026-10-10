"""Response lifetime regression tests."""

import io
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("body,expected", [
    (b'{"webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/test"}', "http://127.0.0.1:9222"),
    (b"invalid json", None),
])
def test_profile_probe_closes_response(tmp_path, monkeypatch, body, expected):
    from tools.browser_tool_real_profile import _surviving_chrome_cdp

    (tmp_path / "DevToolsActivePort").write_text("9222\n/devtools/browser/test\n")
    response = requests.Response()
    response.status_code = 200
    response._content = body
    response.raw = io.BytesIO(body)
    response.close = Mock(wraps=response.close)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)

    assert _surviving_chrome_cdp(str(tmp_path)) == expected
    response.close.assert_called_once_with()
