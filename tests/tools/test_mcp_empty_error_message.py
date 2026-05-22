"""Regression tests for MCP error messages when str(exc) is empty.

Issue #19417: ClosedResourceError (and similar exceptions raised without a
message argument) produced ``MCP call failed: ClosedResourceError: `` with
nothing after the colon, making debugging impossible.

Fix: ``_exc_str()`` falls back to ``repr(exc)`` when ``str(exc)`` is empty.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.mcp_tool import _exc_str, _sanitize_error


# ---------------------------------------------------------------------------
# _exc_str unit tests
# ---------------------------------------------------------------------------


class _EmptyMessageError(Exception):
    """Exception whose __str__ returns empty string (like anyio.ClosedResourceError)."""

    def __str__(self):
        return ""


class _NormalError(Exception):
    pass


def test_exc_str_returns_str_when_nonempty():
    exc = _NormalError("something broke")
    assert _exc_str(exc) == "something broke"


def test_exc_str_falls_back_to_repr_when_str_empty():
    exc = _EmptyMessageError()
    result = _exc_str(exc)
    assert result != ""
    assert "_EmptyMessageError" in result


def test_exc_str_falls_back_to_repr_for_whitespace_only():
    """str(exc) that is only whitespace should also trigger the repr fallback."""
    exc = Exception("   ")
    result = _exc_str(exc)
    # After strip(), the text is empty, so repr is used
    assert result.strip() != ""


def test_exc_str_handles_closedresource_like_exception():
    """Simulate anyio.ClosedResourceError which has no message."""
    # Replicate the real anyio.ClosedResourceError behavior
    exc = type("ClosedResourceError", (Exception,), {"__str__": lambda self: ""})()
    result = _exc_str(exc)
    assert "ClosedResourceError" in result
    assert result != ""


# ---------------------------------------------------------------------------
# Integration: error message format in _sanitize_error
# ---------------------------------------------------------------------------


def test_error_message_not_empty_when_exc_has_no_message():
    """The formatted error string should always contain the exception class name."""
    exc = _EmptyMessageError()
    error_msg = _sanitize_error(
        f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
    )
    assert "ClosedResourceError" not in error_msg or "_EmptyMessageError" in error_msg
    # The key invariant: the message must not end with ": "
    assert not error_msg.endswith(": ")
    # And it must contain the exception type name
    assert "_EmptyMessageError" in error_msg


def test_error_message_preserves_normal_exception_text():
    """Normal exceptions should still show their message text."""
    exc = _NormalError("connection refused")
    error_msg = _sanitize_error(
        f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
    )
    assert "connection refused" in error_msg
    assert "_NormalError" in error_msg
