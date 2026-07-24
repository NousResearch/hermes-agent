"""Regression: the MCP circuit breaker must not treat application-level tool
errors as reachability failures.

A tool call that returns an ``isError`` result (for example a codebase-memory
"function not found") reaches ``_make_tool_handler`` as a *delivered response* —
``_call_once()`` returned rather than raising, so the server is reachable.
Counting such payloads toward the "unreachable" breaker let 3 bad symbol
lookups trip a false ``server unreachable`` 60s cooldown on a perfectly healthy
server. Only genuine transport failures (exceptions / not-connected paths) may
open the breaker.

Both tests mock ``_run_on_mcp_loop`` — the seam ``_call_once()`` uses to run the
tool — so they exercise exactly the bump-vs-reset decision without a live MCP
session or event loop.
"""

import json

import pytest


class _FakeConnectedServer:
    """Minimal stand-in: a non-None ``session`` makes the handler treat the
    server as connected and proceed to ``_call_once()``."""

    session = object()


@pytest.mark.no_isolate
def test_app_level_tool_error_does_not_open_breaker(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    name = "breaker-app-error-probe"
    mcp_tool._reset_server_error(name)
    monkeypatch.setitem(mcp_tool._servers, name, _FakeConnectedServer())

    # The tool ran and returned an application-level error payload — this is
    # what the handler produces from a ``result.isError`` response.
    monkeypatch.setattr(
        mcp_tool,
        "_run_on_mcp_loop",
        lambda *a, **k: json.dumps({"error": "function not found"}),
    )

    handler = mcp_tool._make_tool_handler(name, "trace_path", 5.0)

    for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD):
        out = json.loads(handler({}))
        assert out["error"] == "function not found"  # app error surfaced to caller

    # Delivered responses — even error ones — prove reachability: breaker stays shut.
    assert mcp_tool._server_error_counts.get(name, 0) == 0
    assert name not in mcp_tool._server_breaker_opened_at


@pytest.mark.no_isolate
def test_transport_failure_still_opens_breaker(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    name = "breaker-transport-probe"
    mcp_tool._reset_server_error(name)
    monkeypatch.setitem(mcp_tool._servers, name, _FakeConnectedServer())

    # A genuine transport failure raises out of the call.
    def _boom(*a, **k):
        raise RuntimeError("transport is dead")

    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _boom)

    handler = mcp_tool._make_tool_handler(name, "trace_path", 5.0)

    for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD):
        json.loads(handler({}))  # each returns an {"error": ...} transport-failure payload

    # Real outages must still trip the breaker (unchanged behaviour).
    assert (
        mcp_tool._server_error_counts.get(name, 0)
        >= mcp_tool._CIRCUIT_BREAKER_THRESHOLD
    )
    assert name in mcp_tool._server_breaker_opened_at
