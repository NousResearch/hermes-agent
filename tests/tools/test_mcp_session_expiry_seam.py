"""Compatibility and behavior seams for the extracted MCP session classifier."""

import pytest

from tools import mcp_session_expiry, mcp_tool


_MOVED_NAMES = (
    "_SESSION_EXPIRED_MARKERS",
    "_EXC_TRAVERSAL_MAX_NODES",
    "_is_session_expired_error",
)


@pytest.mark.parametrize("name", _MOVED_NAMES)
def test_mcp_tool_reexports_are_identity_preserving(name):
    """Legacy imports resolve to the canonical extracted objects."""
    assert getattr(mcp_tool, name) is getattr(mcp_session_expiry, name)


def test_session_classifier_matches_markers_case_insensitively():
    for marker in mcp_session_expiry._SESSION_EXPIRED_MARKERS:
        assert mcp_tool._is_session_expired_error(RuntimeError(marker.upper())) is True
    assert mcp_tool._is_session_expired_error(RuntimeError("permission denied")) is False


def test_session_classifier_traverses_causes_and_exception_groups():
    outer = RuntimeError("SDK wrapper")
    outer.__cause__ = RuntimeError("Invalid or expired session")
    assert mcp_tool._is_session_expired_error(outer) is True

    group = RuntimeError("group root")
    group.exceptions = (RuntimeError("unrelated"), RuntimeError("session not found"))
    assert mcp_tool._is_session_expired_error(group) is True


def test_session_classifier_interrupted_error_overrides_transport_signal():
    outer = RuntimeError("transport session expired")
    outer.__cause__ = InterruptedError("user cancelled")
    assert mcp_tool._is_session_expired_error(outer) is False


def test_session_classifier_handles_message_less_anyio_transport_errors():
    from anyio import BrokenResourceError, ClosedResourceError, EndOfStream

    for exc in (BrokenResourceError(), ClosedResourceError(), EndOfStream()):
        assert mcp_tool._is_session_expired_error(exc) is True


def test_session_classifier_terminates_on_cyclic_exception_graph():
    first = RuntimeError("outer")
    second = RuntimeError("session expired")
    first.exceptions = (second, first)
    second.__context__ = first
    assert mcp_tool._is_session_expired_error(first) is True
