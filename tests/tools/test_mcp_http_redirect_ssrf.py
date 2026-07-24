"""MCP HTTP redirect hooks: salvage #62929 + close redirect SSRF."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.mcp_tool import _make_mcp_http_redirect_hooks


class _FakeURL:
    def __init__(self, url: str):
        from urllib.parse import urlparse

        p = urlparse(url)
        self.scheme = p.scheme
        self.host = p.hostname
        self.port = p.port
        self._url = url

    def __str__(self):
        return self._url


def _run(coro):
    return asyncio.run(coro)


def test_public_mcp_redirect_to_metadata_blocked():
    with patch("tools.url_safety.is_safe_url", return_value=True), patch(
        "tools.url_safety.is_always_blocked_url", return_value=True
    ):
        hooks = _make_mcp_http_redirect_hooks("https://mcp.example.com/v1")
        hook = hooks[0]
        next_req = SimpleNamespace(
            url=_FakeURL("http://169.254.169.254/latest/meta-data/"),
            headers={},
        )
        response = SimpleNamespace(is_redirect=True, next_request=next_req)
        with pytest.raises(ValueError, match="metadata|always-blocked|Blocked MCP redirect"):
            _run(hook(response))


def test_public_mcp_redirect_to_private_blocked():
    def _safe(url: str) -> bool:
        return "127.0.0.1" not in url and "169.254." not in url

    with patch("tools.url_safety.is_safe_url", side_effect=_safe), patch(
        "tools.url_safety.is_always_blocked_url", return_value=False
    ):
        hooks = _make_mcp_http_redirect_hooks("https://mcp.example.com/v1")
        hook = hooks[0]
        next_req = SimpleNamespace(
            url=_FakeURL("http://127.0.0.1:9000/secret"),
            headers={"Authorization": "Bearer leak"},
        )
        response = SimpleNamespace(is_redirect=True, next_request=next_req)
        with pytest.raises(ValueError, match="private/internal"):
            _run(hook(response))
        assert "Authorization" not in next_req.headers


def test_loopback_mcp_redirect_to_loopback_allowed():
    def _safe(url: str) -> bool:
        # Loopback origin is not "public" for SSRF purposes.
        return "127.0.0.1" not in url

    with patch("tools.url_safety.is_safe_url", side_effect=_safe), patch(
        "tools.url_safety.is_always_blocked_url", return_value=False
    ):
        hooks = _make_mcp_http_redirect_hooks("http://127.0.0.1:3100/mcp")
        hook = hooks[0]
        next_req = SimpleNamespace(
            url=_FakeURL("http://127.0.0.1:3100/mcp/v2"),
            headers={},
        )
        response = SimpleNamespace(is_redirect=True, next_request=next_req)
        _run(hook(response))
