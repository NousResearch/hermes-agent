"""Tests for _normalize_headers — MCP headers config normalization (#53915).

When MCP server ``headers`` in config.yaml is written as a single-quoted JSON
string (common for Cloudflare/GitHub MCP servers), ``dict(str)`` crashes with
``ValueError: dictionary update sequence element #0 has length 1; 2 is required``.

_normalize_headers handles dict, JSON string, None, and invalid inputs.
"""

import json
import pytest

from tools.mcp_tool import _normalize_headers


class TestNormalizeHeadersDict:
    """Proper dict input (YAML mapping) — the fast path."""

    def test_empty_dict(self):
        assert _normalize_headers({}) == {}

    def test_simple_dict(self):
        headers = {"Authorization": "Bearer token123"}
        assert _normalize_headers(headers) == headers

    def test_multiple_headers(self):
        headers = {
            "Authorization": "Bearer token123",
            "X-Custom-Header": "value",
            "Content-Type": "application/json",
        }
        assert _normalize_headers(headers) == headers

    def test_returns_copy_not_same_object(self):
        """_normalize_headers must return a new dict, not the input reference."""
        original = {"Authorization": "Bearer token"}
        result = _normalize_headers(original)
        assert result == original
        assert result is not original


class TestNormalizeHeadersJsonString:
    """JSON string input — the bug from #53915."""

    def test_valid_json_string(self):
        headers_str = '{"Authorization": "Bearer token123"}'
        assert _normalize_headers(headers_str) == {"Authorization": "Bearer token123"}

    def test_json_string_multiple_headers(self):
        headers_str = '{"Authorization": "Bearer token", "X-Custom": "value"}'
        assert _normalize_headers(headers_str) == {
            "Authorization": "Bearer token",
            "X-Custom": "value",
        }

    def test_json_string_with_cfut_token(self):
        """The exact use case from the bug report — Cloudflare MCP server."""
        headers_str = '{"Authorization": "Bearer cfut_xxx"}'
        assert _normalize_headers(headers_str) == {"Authorization": "Bearer cfut_xxx"}

    def test_json_string_empty_object(self):
        assert _normalize_headers("{}") == {}

    def test_invalid_json_string(self):
        """Invalid JSON string should not crash — return empty dict with warning."""
        assert _normalize_headers("not json at all") == {}

    def test_json_string_not_a_dict(self):
        """JSON string that parses to a list/int should return empty dict."""
        assert _normalize_headers("[1, 2, 3]") == {}
        assert _normalize_headers("42") == {}


class TestNormalizeHeadersNone:
    """None / absent input."""

    def test_none(self):
        assert _normalize_headers(None) == {}

    def test_empty_string(self):
        assert _normalize_headers("") == {}


class TestNormalizeHeadersRegression:
    """The exact crash from #53915 must not happen."""

    def test_dict_on_json_string_does_not_crash(self):
        """Before the fix, dict('{"Authorization": ...}') would crash with
        ValueError: dictionary update sequence element #0 has length 1."""
        # This is what the old code did: dict(config.get("headers") or {})
        headers_str = '{"Authorization": "Bearer cfut_xxx"}'
        # Old code: dict(headers_str) → ValueError
        # New code: _normalize_headers(headers_str) → parsed dict
        result = _normalize_headers(headers_str)
        assert result == {"Authorization": "Bearer cfut_xxx"}
