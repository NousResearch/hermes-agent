"""Tests for tools/fal_common.py — shared FAL.ai SDK plumbing.

Covers: import_fal_client, _normalize_fal_queue_url_format,
_extract_http_status.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from tools.fal_common import (
    _extract_http_status,
    _normalize_fal_queue_url_format,
    import_fal_client,
)

# ---------------------------------------------------------------------------
# import_fal_client
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_fal_client(monkeypatch):
    module = types.ModuleType("fal_client")
    monkeypatch.setitem(sys.modules, "fal_client", module)
    return module

class TestImportFalClient:

    def test_pm_ensure_import_error_is_swallowed(self, monkeypatch, fake_fal_client):
        """If pm.ensure_import raises ImportError, the plain import decides."""
        monkeypatch.setattr(
            "pm.ensure_import",
            MagicMock(side_effect=ImportError("not installed")),
        )

        assert import_fal_client() is fake_fal_client

    def test_pm_ensure_other_exception_raises_import_error(self, monkeypatch):
        """A non-ImportError from pm (an install hint) surfaces as ImportError."""
        monkeypatch.setattr(
            "pm.ensure_import",
            MagicMock(side_effect=RuntimeError("install hint")),
        )
        with pytest.raises(ImportError, match="install hint"):
            import_fal_client()

    def test_pm_missing_is_swallowed(self, fake_fal_client):
        """If pm itself can't be imported, the plain import decides."""
        import builtins

        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "pm":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = import_fal_client()
            assert result is fake_fal_client

# ---------------------------------------------------------------------------
# _normalize_fal_queue_url_format
# ---------------------------------------------------------------------------

class TestNormalizeFalQueueUrlFormat:
    def test_adds_trailing_slash(self):
        assert (
            _normalize_fal_queue_url_format("https://queue.example.com")
            == "https://queue.example.com/"
        )

    def test_strips_trailing_slashes_then_adds_one(self):
        assert (
            _normalize_fal_queue_url_format("https://queue.example.com///")
            == "https://queue.example.com/"
        )

    def test_strips_whitespace(self):
        assert (
            _normalize_fal_queue_url_format("  https://queue.example.com  ")
            == "https://queue.example.com/"
        )

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="FAL queue origin is required"):
            _normalize_fal_queue_url_format("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="FAL queue origin is required"):
            _normalize_fal_queue_url_format("   ")

# ---------------------------------------------------------------------------
# _extract_http_status
# ---------------------------------------------------------------------------

class TestExtractHttpStatus:
    def test_returns_status_from_response_attribute(self):
        """httpx.HTTPStatusError exposes .response.status_code."""
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = 404
        assert _extract_http_status(exc) == 404

    def test_returns_status_from_exc_status_code(self):
        """fal_client wrappers expose .status_code directly."""
        exc = MagicMock()
        exc.response = None
        exc.status_code = 500
        assert _extract_http_status(exc) == 500

    def test_returns_none_when_no_response_and_no_status_code(self):
        exc = Exception("plain error")
        assert _extract_http_status(exc) is None

    def test_returns_none_when_response_status_code_not_int(self):
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = "not-int"
        del exc.status_code
        assert _extract_http_status(exc) is None

    def test_response_status_takes_precedence_over_exc_status(self):
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = 200
        exc.status_code = 500
        assert _extract_http_status(exc) == 200

    def test_falls_back_to_exc_status_when_response_status_not_int(self):
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = "bad"
        exc.status_code = 503
        assert _extract_http_status(exc) == 503

