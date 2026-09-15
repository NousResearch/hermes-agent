"""Regression: a cyclic ``__cause__``/``__context__`` chain must not turn a connect
failure into a RecursionError. Hit live by ``hermes mcp add`` against an OAuth-only
remote server: the probe failed, error formatting recursed on the self-referential
chain, and the CLI crashed with ``maximum recursion depth exceeded`` instead of
reporting the auth failure.
"""
from tools.mcp_tool_errors import _format_connect_error


def _cyclic_chain() -> BaseException:
    inner = RuntimeError("oauth: authorization required")
    outer = ConnectionError("failed to connect to https://example.invalid/mcp")
    outer.__context__ = inner
    inner.__context__ = outer
    return outer


def test_format_connect_error_survives_cyclic_chain():
    """The real messages survive; the walk terminates instead of recursing."""
    message = _format_connect_error(_cyclic_chain())
    assert "authorization required" in message or "failed to connect" in message
    assert "RecursionError" not in message


def test_format_connect_error_finds_missing_executable_through_a_cycle():
    """A FileNotFoundError is still reported as the missing executable, cycle or not."""
    missing = FileNotFoundError(2, "No such file or directory", "npx")
    outer = ConnectionError("failed to connect")
    outer.__context__ = missing
    missing.__context__ = outer
    assert "missing executable 'npx'" in _format_connect_error(outer)
