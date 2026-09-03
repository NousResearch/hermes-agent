"""Tests for bounded readiness waits on MCP read-path handlers.

When a call lands while an MCP server is still completing its initial
connection or a reconnect, ``server.session`` is briefly ``None``. The
``tools/call`` handler (``_make_tool_handler``) already waits briefly for a
fresh session instead of failing immediately (see #26892). The four
protocol-level read handlers -- ``list_resources``, ``read_resource``,
``list_prompts``, ``get_prompt`` -- did not share that wait: they returned
"not connected" the instant ``session`` was ``None``, even though a
reconnect might complete a fraction of a second later. This file locks in
the fix: all read-path handlers wait up to a bounded timeout for a live
session before giving up, and still fail (rather than hang) when the
session never comes back.
"""
import json
import threading
import time
from unittest.mock import MagicMock

import pytest


def _install_connecting_server(name: str):
    """Register a real MCPServerTask with no session yet (mid-connect)."""
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask

    mcp_tool._ensure_mcp_loop()
    server = MCPServerTask(name)
    server.session = None
    mcp_tool._servers[name] = server
    return server


def _make_ready_result(session_method: str):
    """Return a MagicMock session whose *session_method* succeeds cleanly."""
    session = MagicMock()

    if session_method == "list_resources":
        result = MagicMock(resources=[], nextCursor=None)
    elif session_method == "read_resource":
        result = MagicMock(contents=[])
    elif session_method == "list_prompts":
        result = MagicMock(prompts=[], nextCursor=None)
    else:  # get_prompt
        result = MagicMock(messages=[], description=None)

    async def _call(*a, **kw):
        return result

    setattr(session, session_method, _call)
    return session


@pytest.mark.parametrize(
    "handler_factory, handler_kwargs, session_method, call_args",
    [
        ("_make_list_resources_handler", {"tool_timeout": 2.0}, "list_resources", {}),
        ("_make_read_resource_handler", {"tool_timeout": 2.0}, "read_resource", {"uri": "file://x"}),
        ("_make_list_prompts_handler", {"tool_timeout": 2.0}, "list_prompts", {}),
        ("_make_get_prompt_handler", {"tool_timeout": 2.0}, "get_prompt", {"name": "p1"}),
    ],
)
def test_read_handler_waits_for_session_instead_of_failing_immediately(
    monkeypatch, tmp_path, handler_factory, handler_kwargs, session_method, call_args,
):
    """A call arriving mid-connect must wait for the session, not fail
    on the spot -- a reconnect that completes 150ms later should still
    let the call through."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    server_name = f"connecting-{session_method}"
    server = _install_connecting_server(server_name)

    def _finish_connecting():
        time.sleep(0.15)
        server.session = _make_ready_result(session_method)
        server._ready.set()

    threading.Thread(target=_finish_connecting, daemon=True).start()

    try:
        factory = getattr(mcp_tool, handler_factory)
        handler = factory(server_name, **handler_kwargs)
        out = handler(call_args)
        parsed = json.loads(out)
        assert "error" not in parsed, (
            f"{handler_factory}: expected the handler to wait for the "
            f"in-flight (re)connect and succeed, got {parsed}"
        )
    finally:
        mcp_tool._servers.pop(server_name, None)


@pytest.mark.parametrize(
    "handler_factory, handler_kwargs, call_args",
    [
        ("_make_list_resources_handler", {"tool_timeout": 0.6}, {}),
        ("_make_read_resource_handler", {"tool_timeout": 0.6}, {"uri": "file://x"}),
        ("_make_list_prompts_handler", {"tool_timeout": 0.6}, {}),
        ("_make_get_prompt_handler", {"tool_timeout": 0.6}, {"name": "p1"}),
    ],
)
def test_read_handler_bounded_wait_times_out_cleanly(
    monkeypatch, tmp_path, handler_factory, handler_kwargs, call_args,
):
    """If the session never comes back, the handler must still return an
    error (not hang forever) -- and it must have actually waited some of
    the bounded window rather than failing on the very first check."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    server_name = f"never-ready-{handler_factory}"
    _install_connecting_server(server_name)

    try:
        factory = getattr(mcp_tool, handler_factory)
        handler = factory(server_name, **handler_kwargs)
        started = time.monotonic()
        out = handler(call_args)
        elapsed = time.monotonic() - started
        parsed = json.loads(out)
        assert "error" in parsed
        assert elapsed >= 0.2, (
            f"{handler_factory}: expected a bounded wait of a real fraction "
            f"of the timeout budget before giving up, only waited {elapsed:.3f}s"
        )
    finally:
        mcp_tool._servers.pop(server_name, None)
