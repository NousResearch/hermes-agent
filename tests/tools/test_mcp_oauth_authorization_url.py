"""MCP OAuth authorization_url scheme allowlist.

``_redirect_handler`` opens the provider URL in a browser (and may publish it
to the dashboard). Only http(s) URLs with a host are accepted so a hostile
``javascript:`` / ``file:`` response cannot be navigated.
"""

from __future__ import annotations

import pytest

from tools.mcp_oauth import _is_valid_authorization_url, _make_redirect_handler


@pytest.mark.parametrize(
    "url",
    [
        "https://auth.example.com/authorize?client_id=1",
        "http://127.0.0.1:8080/authorize",
        "http://localhost:3000/oauth",
        "http://[::1]:8080/authorize",
        "https://[::1]/443/authorize",
    ],
)
def test_accepts_https_and_loopback_http_authorization_urls(url):
    assert _is_valid_authorization_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html,hi",
        "ftp://example.com/auth",
        "http://attacker.example/authorize",  # non-loopback http
        "http://169.254.169.254/latest/meta-data",  # link-local, not loopback
        "https://",  # scheme only, no host
        "not a url",
        "",
        "   ",
    ],
)
def test_rejects_non_https_or_non_loopback_authorization_urls(url):
    assert _is_valid_authorization_url(url) is False


@pytest.mark.asyncio
async def test_redirect_handler_rejects_javascript_scheme():
    handler = _make_redirect_handler(port=0)
    with pytest.raises(ValueError, match="http\\(s\\) URL"):
        await handler("javascript:alert(1)")
