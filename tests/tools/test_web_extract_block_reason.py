"""web_extract must report WHY a URL was blocked, not always blame a private address.

A hostname that does not resolve (NXDOMAIN) and a hostname that resolves into RFC1918 are
different failures needing different user action ("the address is wrong" vs "the address is
internal"). Both used to surface as the same "targets a private or internal network address"
string, which actively misleads: a typo'd or dead domain reads as an infrastructure/SSRF policy
problem. These tests pin the two messages apart, and pin the reasons to a closed constant set so
resolver detail cannot leak into tool output.
"""
from __future__ import annotations

import asyncio
import json
import socket
from unittest.mock import patch

import tools.web_tools as wt
from tools import url_safety
from tools.url_safety import (
    DNS_FAILURE_REASON,
    INVALID_URL_REASON,
    PRIVATE_ADDRESS_REASON,
    url_block_reason,
)

_PUBLIC = [(socket.AF_INET, None, None, "", ("93.184.216.34", 443))]
_PRIVATE = [(socket.AF_INET, None, None, "", ("10.0.0.5", 443))]


def _extract(url: str) -> dict:
    payload = json.loads(asyncio.run(wt.web_extract_tool([url])))
    return payload["results"][0]


def test_url_block_reason_separates_dns_failure_from_private_address():
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
        assert url_block_reason("https://nonexistent.example.com") == DNS_FAILURE_REASON
    with patch("socket.getaddrinfo", return_value=_PRIVATE):
        assert url_block_reason("https://internal.example.com") == PRIVATE_ADDRESS_REASON
    with patch("socket.getaddrinfo", return_value=_PUBLIC):
        assert url_block_reason("https://example.com") is None


def test_url_block_reason_agrees_with_is_safe_url():
    """The bool gate must stay exactly `reason is None` — no policy drift between the two."""
    cases = [
        ("https://example.com", _PUBLIC, None),
        ("https://internal.example.com", _PRIVATE, socket.gaierror("x")),
    ]
    for url, addrs, _ in cases:
        with patch("socket.getaddrinfo", return_value=addrs):
            assert url_safety.is_safe_url(url) is (url_block_reason(url) is None)
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
        assert url_safety.is_safe_url("https://nope.example.com") is False


def test_reasons_never_leak_resolver_detail():
    """Reasons are a closed constant set: no exception text, hostname or IP in the message."""
    allowed = {DNS_FAILURE_REASON, PRIVATE_ADDRESS_REASON, INVALID_URL_REASON}
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
        assert url_block_reason("https://nonexistent.example.com") in allowed
    with patch("socket.getaddrinfo", return_value=_PRIVATE):
        reason = url_block_reason("https://internal.example.com")
        assert reason in allowed
        assert "10.0.0.5" not in reason
    with patch("socket.getaddrinfo", side_effect=RuntimeError("resolver stack trace detail")):
        reason = url_block_reason("https://boom.example.com")
        assert reason in allowed
        assert "resolver stack trace detail" not in reason
    assert url_block_reason("file:///etc/passwd") == INVALID_URL_REASON
    assert url_block_reason("https://") == INVALID_URL_REASON


def test_extract_reports_dns_failure_not_private_address():
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
        entry = _extract("https://nonexistent.example.com/x")
    assert DNS_FAILURE_REASON in entry["error"]
    assert PRIVATE_ADDRESS_REASON not in entry["error"]


def test_extract_still_reports_private_address_for_internal_target():
    with patch("socket.getaddrinfo", return_value=_PRIVATE):
        entry = _extract("https://internal.example.com/x")
    assert PRIVATE_ADDRESS_REASON in entry["error"]


def test_extract_resolves_each_url_once():
    """The blocked path must not resolve twice: a second lookup doubles the timeout on exactly
    the failure this patch targets, and could report a reason from a different DNS answer."""
    calls: list[str] = []

    def _fake_getaddrinfo(host, *a, **k):
        calls.append(host)
        raise socket.gaierror("no such host")

    with patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo):
        entry = _extract("https://nonexistent.example.com/x")

    assert DNS_FAILURE_REASON in entry["error"]
    assert calls.count("nonexistent.example.com") == 1, calls
