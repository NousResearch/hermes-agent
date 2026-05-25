"""Tests for tools.fal_common — _normalize_fal_queue_url_format and _extract_http_status."""

from __future__ import annotations

import pytest

from tools.fal_common import _extract_http_status, _normalize_fal_queue_url_format


# ============================================================================
# _normalize_fal_queue_url_format
# ============================================================================
class TestNormalizeFalQueueUrlFormat:
    def test_basic_url(self):
        result = _normalize_fal_queue_url_format("https://fal.example.com")
        assert result == "https://fal.example.com/"

    def test_strips_trailing_slash(self):
        result = _normalize_fal_queue_url_format("https://fal.example.com/")
        assert result == "https://fal.example.com/"

    def test_strips_multiple_slashes(self):
        result = _normalize_fal_queue_url_format("https://fal.example.com///")
        assert result == "https://fal.example.com/"

    def test_strips_whitespace(self):
        result = _normalize_fal_queue_url_format("  https://fal.example.com  ")
        assert result == "https://fal.example.com/"

    def test_no_scheme_url(self):
        result = _normalize_fal_queue_url_format("api.fal.example.com")
        assert result == "api.fal.example.com/"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="required"):
            _normalize_fal_queue_url_format("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="required"):
            _normalize_fal_queue_url_format("   ")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="required"):
            _normalize_fal_queue_url_format(None)  # type: ignore[arg-type]

    def test_url_with_path(self):
        result = _normalize_fal_queue_url_format("https://fal.example.com/v1/queue")
        assert result == "https://fal.example.com/v1/queue/"

    def test_integer_input(self):
        """str(42) → '42' → valid origin."""
        result = _normalize_fal_queue_url_format(42)  # type: ignore[arg-type]
        assert result == "42/"


# ============================================================================
# _extract_http_status
# ============================================================================
class TestExtractHttpStatus:
    def test_httpx_http_status_error(self):
        # Simulate httpx.HTTPStatusError with response.status_code
        response = type("Response", (), {"status_code": 429})()
        exc = type("HTTPStatusError", (Exception,), {"response": response})("rate limited")
        assert _extract_http_status(exc) == 429

    def test_direct_status_code_attribute(self):
        exc = type("FalError", (Exception,), {"status_code": 503})("service unavailable")
        assert _extract_http_status(exc) == 503

    def test_response_attribute_without_status_code(self):
        response = type("Response", (), {})()
        exc = type("Error", (Exception,), {"response": response})("msg")
        assert _extract_http_status(exc) is None

    def test_no_response_no_status_code(self):
        exc = ValueError("plain error")
        assert _extract_http_status(exc) is None

    def test_plain_exception(self):
        exc = RuntimeError("something broke")
        assert _extract_http_status(exc) is None

    def test_response_status_not_int(self):
        response = type("Response", (), {"status_code": "200"})()
        exc = type("Error", (Exception,), {"response": response})("msg")
        assert _extract_http_status(exc) is None

    def test_status_code_is_zero(self):
        # 0 is a valid int status code
        exc = type("Error", (Exception,), {"status_code": 0})("msg")
        assert _extract_http_status(exc) == 0

    def test_response_takes_priority_over_direct(self):
        response = type("Response", (), {"status_code": 401})()
        exc = type("Error", (Exception,), {"response": response, "status_code": 500})("msg")
        assert _extract_http_status(exc) == 401

    def test_direct_status_code_when_no_response(self):
        exc = type("Error", (Exception,), {"status_code": 404})("msg")
        assert _extract_http_status(exc) == 404

    def test_status_code_is_boolean(self):
        """bool(True) → isinstance(True, int) is True in Python."""
        exc = type("Error", (Exception,), {"status_code": True})("msg")
        # True is int subclass → caught
        assert _extract_http_status(exc) == 1
