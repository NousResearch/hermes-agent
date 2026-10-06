"""Regression tests for MCP error messages when str(exc) is empty.

Issue #19417: ClosedResourceError (and similar exceptions raised without a
message argument) produced ``MCP call failed: ClosedResourceError: `` with
nothing after the colon, making debugging impossible.

Fix: ``_exc_str()`` falls back to ``repr(exc)`` when ``str(exc)`` is empty.
"""

from tools.mcp_tool_common import _exc_str


class _EmptyMessageError(Exception):
    """Exception whose __str__ returns empty string (like anyio.ClosedResourceError)."""

    def __str__(self):
        return ""


def test_exc_str_falls_back_to_repr_when_message_is_empty():
    """The #19417 shape: an empty-message exception must still yield diagnostics."""
    text = _exc_str(_EmptyMessageError())
    assert text.strip(), "empty exception message must not produce an empty diagnostic"
    assert "_EmptyMessageError" in text


def test_exc_str_falls_back_for_whitespace_only_message():
    text = _exc_str(RuntimeError("   "))
    assert "RuntimeError" in text


def test_exc_str_keeps_real_message():
    assert _exc_str(ValueError("connection refused")) == "connection refused"
