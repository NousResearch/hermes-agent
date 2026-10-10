"""Health probes close their responses on both healthy and unhealthy results."""

import io
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("status", [200, 503])
def test_health_probe_closes_response(monkeypatch, status):
    from tools import browser_camofox

    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    monkeypatch.setattr(browser_camofox, "_vnc_url_checked", False)
    monkeypatch.setattr(browser_camofox, "_vnc_url_by_camofox_url", {})
    response = requests.Response()
    response.status_code = status
    response._content = b'{"ok": true, "vncPort": 6080}'
    response.raw = io.BytesIO(response._content)
    response.close = Mock(wraps=response.close)
    monkeypatch.setattr(browser_camofox.requests, "get", lambda *args, **kwargs: response)

    assert browser_camofox.check_camofox_available() is (status == 200)
    response.close.assert_called_once_with()
