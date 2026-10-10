"""Invariants for ``tools/mcp_tool_errors._format_connect_error`` on malformed exception chains.

``__cause__``/``__context__`` can form a cycle (the same OAuth error re-raised on the SSE fallback,
a raised-and-caught pair) and stdio failures can nest deeper than the recursion limit; either used
to turn ``hermes mcp test`` into a RecursionError that hid the real connect error (#111952, #111997).
"""
import sys

from tools.mcp_tool_errors import _format_connect_error


def test_format_connect_error_reports_real_messages_on_cyclic_chain():
    """A two-node ``__cause__``/``__context__`` cycle renders every distinct message, once, in chain order."""
    first = RuntimeError("first failure")
    second = RuntimeError("second failure")
    first.__cause__ = second
    second.__context__ = first

    assert _format_connect_error(first) == "first failure; second failure"


def test_format_connect_error_finds_missing_executable_through_deep_cyclic_chain():
    """A missing stdio binary wrapped deeper than the recursion limit, with the chain looping back to the top,
    is still reported as the missing executable rather than as a RecursionError."""
    missing = FileNotFoundError(2, "No such file or directory", "/opt/homebrew/bin/removed-mcp-server")
    current = missing
    for _ in range(sys.getrecursionlimit() + 10):
        wrapper = RuntimeError("stdio startup failed")
        wrapper.__cause__ = current
        current = wrapper
    missing.__context__ = current

    assert _format_connect_error(current) == "missing executable '/opt/homebrew/bin/removed-mcp-server'"


def test_describe_http_failure_redacts_credentials_from_recorded_request_url():
    from types import SimpleNamespace

    from tools.mcp_tool_errors import _describe_http_failure

    error = RuntimeError("Server returned an error response")
    error.error = SimpleNamespace(code=-32603)
    rejection = {
        "status": 504,
        "method": "POST",
        "url": "https://mcp.example.com/mcp?token=SECRETVALUE&mode=tools",
        "body": "upstream timeout",
    }

    result = _describe_http_failure(error, rejection)

    assert "SECRETVALUE" not in result
    assert "https://mcp.example.com/mcp?token=***&mode=tools" in result


def test_describe_http_failure_forces_url_redaction_when_secret_redaction_is_disabled(monkeypatch):
    from agent import redact
    from tools.mcp_tool_errors import _describe_http_failure
    from types import SimpleNamespace

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    error = RuntimeError("401 Unauthorized for https://user:pass@mcp.example.com/mcp?token=SECRETVALUE")
    error.response = SimpleNamespace(status_code=401)

    result = _describe_http_failure(error, {})

    assert "SECRETVALUE" not in result and "user:pass" not in result
    assert "https://***:***@mcp.example.com/mcp?token=***" in result
