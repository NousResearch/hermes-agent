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


def _real_redirect_response(location, url="https://mcp.example.com/v1"):
    """A genuine httpx redirect response whose ``next_request`` is unset.

    Inside an ``httpx.AsyncClient`` response hook, ``next_request`` is frequently
    ``None`` even for a real redirect: httpx only materialises it later, while
    following the redirect.  These tests deliberately do **not** pre-populate
    it, so a guard keyed on ``response.next_request`` fails them.
    """
    import httpx

    response = httpx.Response(
        302,
        headers={"Location": location},
        request=httpx.Request("GET", url),
    )
    assert response.next_request is None, "httpx must not pre-populate next_request"
    return response


def test_location_target_blocked_without_next_request():
    """Regression: safety must not depend on ``response.next_request``.

    A public MCP URL that 302s to cloud metadata must be blocked even when the
    hook sees ``next_request is None``.
    """
    with patch("tools.url_safety.is_safe_url", return_value=True), patch(
        "tools.url_safety.is_always_blocked_url", return_value=True
    ):
        hook = _make_mcp_http_redirect_hooks("https://mcp.example.com/v1")[0]
        response = _real_redirect_response("http://169.254.169.254/latest/meta-data/")
        with pytest.raises(ValueError, match="metadata|always-blocked|Blocked MCP redirect"):
            _run(hook(response))


def test_location_private_target_blocked_without_next_request():
    """Public origin + private ``Location`` target is blocked with no next_request."""
    def _safe(url: str) -> bool:
        return "127.0.0.1" not in url and "169.254." not in url

    with patch("tools.url_safety.is_safe_url", side_effect=_safe), patch(
        "tools.url_safety.is_always_blocked_url", return_value=False
    ):
        hook = _make_mcp_http_redirect_hooks("https://mcp.example.com/v1")[0]
        response = _real_redirect_response("http://127.0.0.1:9000/secret")
        with pytest.raises(ValueError, match="private/internal"):
            _run(hook(response))


def test_relative_location_blocked_without_next_request():
    """A relative ``Location`` is resolved against the response URL, then blocked."""
    with patch("tools.url_safety.is_safe_url", return_value=True), patch(
        "tools.url_safety.is_always_blocked_url", return_value=False
    ):
        hook = _make_mcp_http_redirect_hooks("https://mcp.example.com/v1")[0]
        response = _real_redirect_response("/meta", url="https://mcp.example.com/v1")
        _run(hook(response))  # same-origin relative redirect stays allowed

    with patch("tools.url_safety.is_safe_url", side_effect=lambda u: "169.254." not in u), patch(
        "tools.url_safety.is_always_blocked_url", return_value=False
    ):
        hook = _make_mcp_http_redirect_hooks("https://mcp.example.com/v1")[0]
        response = _real_redirect_response(
            "//169.254.169.254/latest", url="https://mcp.example.com/v1"
        )
        with pytest.raises(ValueError, match="private/internal"):
            _run(hook(response))


def test_loopback_location_redirect_allowed_without_next_request():
    """Intentionally-private local MCP servers may still redirect to loopback."""
    def _safe(url: str) -> bool:
        return "127.0.0.1" not in url

    with patch("tools.url_safety.is_safe_url", side_effect=_safe), patch(
        "tools.url_safety.is_always_blocked_url", return_value=False
    ):
        hook = _make_mcp_http_redirect_hooks("http://127.0.0.1:3100/mcp")[0]
        response = _real_redirect_response(
            "http://127.0.0.1:3100/mcp/v2", url="http://127.0.0.1:3100/mcp"
        )
        _run(hook(response))


def test_non_redirect_response_is_ignored_without_next_request():
    """A plain 200 with no redirect must not trip the guard."""
    import httpx

    with patch("tools.url_safety.is_safe_url", return_value=True), patch(
        "tools.url_safety.is_always_blocked_url", return_value=True
    ):
        hook = _make_mcp_http_redirect_hooks("https://mcp.example.com/v1")[0]
        response = httpx.Response(
            200, request=httpx.Request("GET", "https://mcp.example.com/v1")
        )
        assert response.next_request is None
        _run(hook(response))
