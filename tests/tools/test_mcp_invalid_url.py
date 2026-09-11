"""Tests for the MCP remote-URL validator.

Ported from anomalyco/opencode#25019 (``fix: handle invalid mcp urls``).

Previously, a typo in ``config.yaml`` (missing scheme, wrong scheme, empty
string, dict where a URL was expected) caused the MCP server startup code
to enter httpx's URL-parsing path and crash inside the transport layer.
The reconnect-backoff loop would then retry
``_MAX_INITIAL_CONNECT_RETRIES`` times with doubling backoff — a minute or
more of pointless retries plus a confusing opaque error message — before
eventually giving up.

The fix validates the URL once, up front, and fails fast with a specific
error message identifying the offending server.
"""

from __future__ import annotations

import pytest

from tools.mcp_tool_errors import InvalidMcpUrlError, _validate_remote_mcp_url


class TestValidUrlsAccepted:
    """Every valid http(s) URL must pass through untouched (stripped of whitespace)."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:3000/mcp",
            "https://example.com/mcp",
            "https://context7.liam.com/mcp",
            "http://127.0.0.1:8080",
            "https://api.example.com:443/v1/mcp?session=abc",
            "http://[::1]:9000/mcp",  # IPv6
            "https://host.example.com",  # no port, no path
        ],
    )
    def test_accepts_valid_http_url(self, url):
        assert _validate_remote_mcp_url("test", url) == url

    def test_strips_surrounding_whitespace(self):
        assert (
            _validate_remote_mcp_url("test", "  https://example.com/mcp  ")
            == "https://example.com/mcp"
        )


class TestInvalidUrlsRejected:
    """Every broken shape must raise ``InvalidMcpUrlError`` with a clear message."""

    def test_none_rejected(self):
        with pytest.raises(InvalidMcpUrlError, match="context7.*expected a string"):
            _validate_remote_mcp_url("context7", None)

    def test_dict_rejected(self):
        with pytest.raises(InvalidMcpUrlError, match="expected a string, got dict"):
            _validate_remote_mcp_url("ctx", {"url": "nested"})

    def test_int_rejected(self):
        with pytest.raises(InvalidMcpUrlError, match="expected a string, got int"):
            _validate_remote_mcp_url("ctx", 8080)


    def test_error_mentions_server_name(self):
        # So users can find the bad entry when there are multiple configured.
        with pytest.raises(InvalidMcpUrlError, match="my-weird-server"):
            _validate_remote_mcp_url("my-weird-server", "not a url at all")


class TestErrorIsValueError:
    """InvalidMcpUrlError must be a ValueError for broad downstream catch blocks."""

    def test_is_value_error(self):
        try:
            _validate_remote_mcp_url("ctx", "garbage")
        except ValueError:
            pass  # expected
        else:
            pytest.fail("expected ValueError")
