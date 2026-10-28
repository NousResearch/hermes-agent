"""URLs whose authority urllib.parse and WHATWG clients (browsers, Node) read differently.

Every case resolves to a public address, so DNS is never what blocks it: only the parse
does. ``http://127.0.0.1:8080\\@1.1.1.1`` is host 1.1.1.1 to urlsplit but 127.0.0.1:8080
to a browser.
"""

import logging
import socket
from unittest.mock import patch

import pytest

from tools.url_safety import async_is_safe_url, is_always_blocked_url, is_safe_url


@pytest.fixture(autouse=True)
def _public_dns():
    """Every hostname resolves to a public address."""
    answer = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 0))]
    with patch("socket.getaddrinfo", return_value=answer):
        yield


AMBIGUOUS = [
    "http://127.0.0.1:8080\\@1.1.1.1",          # reported: browser dials 127.0.0.1:8080
    "http://127.0.0.1:8080\\\\@1.1.1.1",
    "http://127.0.0.1\\@1.1.1.1/",
    "http://1.1.1.1\\.127.0.0.1/",
    "http://\\127.0.0.1/",
    "http://%31%32%37.0.0.1/",                   # WHATWG percent-decodes the host
    "http://127.0.0.1:8080\t@1.1.1.1",
    "http://127.0.0.1:8080\n@1.1.1.1",
    "http://127.0.0.1:8080\r@1.1.1.1",
    "http://127.0.0.1:8080\x00@1.1.1.1",
    "http://exa\tmple.com/",
    "https://example.com/a\nb",
    "http://exa mple.com/",
    "http://1.1.1.1:99999/",
    "http://1.1.1.1:80x/",
]

USERINFO = [
    "http://127.0.0.1:8080%5C@1.1.1.1",
    "http://user:pass@1.1.1.1/",
    "http://1.1.1.1@1.1.1.1/",
    "https://@example.com/",
]

LEGITIMATE = [
    "https://example.com/image.png",
    "https://example.com/a\\b?q=c\\d",          # backslash after the authority
    "https://example.com/%5C@x",
    "https://example.com/p@q?x=@y#frag@z",       # '@' after the authority
    "https://example.com/r?next=http://127.0.0.1\\@x",
    "https://example.com/path%20with%20spaces?q=a%40b",
    "https://example.com:8443/x",
    "http://[2606:4700:4700::1111]:8080/",
    "HTTPS://EXAMPLE.COM/",
    "  https://example.com/  ",                 # surrounding whitespace is stripped by every parser
]


class TestIsSafeUrlRefusesAmbiguousAuthority:
    @pytest.mark.parametrize("url", AMBIGUOUS + USERINFO)
    def test_refused(self, url):
        assert is_safe_url(url) is False

    @pytest.mark.parametrize("url", LEGITIMATE)
    def test_legitimate_urls_still_allowed(self, url):
        assert is_safe_url(url) is True

    @pytest.mark.asyncio
    async def test_async_variant_refuses_reported_url(self):
        assert await async_is_safe_url("http://127.0.0.1:8080\\@1.1.1.1") is False

    def test_log_omits_userinfo_credentials(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.url_safety"):
            assert is_safe_url("https://alice:hunter2@example.com/") is False
        assert "userinfo" in caplog.text
        assert "hunter2" not in caplog.text


class TestAlwaysBlockedFloorRefusesAmbiguousAuthority:
    """The floor guards browser backends that skip ``is_safe_url``; Chromium reads the WHATWG host."""

    @pytest.mark.parametrize("url", [
        "http://169.254.169.254\\@1.1.1.1/latest/meta-data/",
        "http:\\\\169.254.169.254\\latest",
        "http:///169.254.169.254/latest",
        "http://%31%36%39.254.169.254/",
        "http://169.254.169.254\t@1.1.1.1/",
    ])
    def test_refused(self, url):
        assert is_always_blocked_url(url) is True

    @pytest.mark.parametrize("url", [
        "https://user:pass@example.com/",           # userinfo alone parses the same everywhere
        "https://example.com/a\\b",
        "about:blank",
        "data:text/plain,line1\nline2",
    ])
    def test_unaffected(self, url):
        assert is_always_blocked_url(url) is False
