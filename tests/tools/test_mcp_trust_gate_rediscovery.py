"""The ``trust: untrusted`` gate decides against the metadata of the session that runs the call (#144380).

The gate classifies from the cached ``readOnlyHint`` before any transport work. A reconnect (the recycled-stdio
wake inside acquire, or a recoverer's reconnect before its retry) re-runs ``MCPServerTask._discover_tools`` on a
server whose tools are already registered, and the new listing can report the tool as write-capable. Such a call
must not reach the server without consent, and a refused call must not keep its spend charge.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool, mcp_tool_costs
from tools import mcp_tool_discovery as discovery
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_registration as registration
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.mcp_tool import MCPServerTask

UNTRUSTED = {"trust": "untrusted", "command": "srv-bin"}
COSTED = {**UNTRUSTED, "tool_costs": {"sync": 40}, "spend_limit": 100}
HTTP_UNTRUSTED = {"trust": "untrusted", "url": "https://mcp.example.test/mcp"}


def _run_on_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _session(read_only: bool):
    """A session whose tools/list reports ``sync`` with *read_only* and whose call succeeds."""
    tool = SimpleNamespace(name="sync", annotations={"readOnlyHint": read_only}, inputSchema={}, description="")
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[tool], nextCursor=None))
    session.call_tool = AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], isError=False, structuredContent=None))
    return session


def _reconnect(task, session):
    """What a reconnect does once the fresh transport is up: publish the session, run the real discovery."""
    task.session = session
    _run_on_loop(task._discover_tools)


def _server(config=UNTRUSTED):
    """A connected server whose ``sync`` tool was registered (and recorded read-only) by its first connect."""
    task = MCPServerTask("srv")
    task._config = dict(config)
    task._registered_tool_names = ["mcp_srv_sync"]
    task.session = _session(read_only=True)
    task._tools = [SimpleNamespace(name="sync", annotations={"readOnlyHint": True})]
    registration._record_tool_trust_metadata("srv", task._config, task._tools)
    return task


@pytest.fixture(autouse=True)
def isolated_state():
    token = set_current_session_key("chat-a")
    with patch.dict(mcp_tool._servers, {}, clear=True), \
         patch.dict(mcp_tool._lazy_server_configs, {}, clear=True), \
         patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=_run_on_loop), \
         patch.object(handlers, "_mcp_loop_running", return_value=True), \
         patch.dict(mcp_tool._server_error_counts, {}, clear=True), \
         patch.dict(mcp_tool._server_trust_levels, {}, clear=True), \
         patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True), \
         patch.dict(mcp_tool._server_cost_policies, {}, clear=True), \
         patch.dict(mcp_tool_costs._spent, {}, clear=True), \
         patch.dict(mcp_tool_costs._ask_locks, {}, clear=True), \
         patch.object(mcp_tool_costs, "_unscoped_moved_to", None):
        yield
    reset_current_session_key(token)


def _install(task):
    mcp_tool._servers["srv"] = task
    return task


def _recycled_wake_rediscovers(task, session):
    """A recycled stdio server: the call's acquire wakes it, and the respawned session is rediscovered."""
    task.session = None
    task._recycled_reason = "idle"

    def wake(server_name, server):
        _reconnect(server, session)
        return True
    return patch.object(discovery, "_request_lazy_reconnect", side_effect=wake)


def _call(answer="accept"):
    asked = []
    with patch("tools.approval_prompt.request_elicitation_consent",
               side_effect=lambda msg, desc, **kw: asked.append(msg) or answer):
        raw = handlers._make_tool_handler("srv", "sync", 30.0)({"target": "x"})
    return json.loads(raw), asked


def test_reconnect_discovery_of_registered_server_refreshes_read_only_hint():
    task = _install(_server())
    _reconnect(task, _session(read_only=False))
    assert handlers._tool_is_read_only("srv", "sync") is False
    assert task._registered_tool_names == ["mcp_srv_sync"]


def test_recycled_wake_that_turns_the_tool_write_capable_asks_and_deny_blocks():
    task = _install(_server())
    new = _session(read_only=False)
    with _recycled_wake_rediscovers(task, new):
        result, asked = _call(answer="decline")
    assert len(asked) == 1 and "UNTRUSTED server 'srv'" in asked[0]
    assert "NOT run" in result["error"]
    new.call_tool.assert_not_awaited()


def test_recycled_wake_that_turns_the_tool_write_capable_runs_once_approved():
    task = _install(_server())
    new = _session(read_only=False)
    with _recycled_wake_rediscovers(task, new):
        result, asked = _call(answer="accept")
    assert result == {"result": "ok"} and len(asked) == 1
    new.call_tool.assert_awaited_once()


def test_recycled_wake_with_unchanged_hint_never_asks():
    task = _install(_server())
    new = _session(read_only=True)
    with _recycled_wake_rediscovers(task, new):
        result, asked = _call()
    assert result == {"result": "ok"} and asked == []


def test_denied_recheck_refunds_the_gate_booked_charge():
    task = _install(_server(COSTED))
    new = _session(read_only=False)
    with _recycled_wake_rediscovers(task, new):
        result, _ = _call(answer="decline")
    assert "error" in result
    assert sum(mcp_tool_costs._spent.values()) == 0
    new.call_tool.assert_not_awaited()


def _reconnect_before_retry(task, session):
    def reconnect(server_name, srv, **kwargs):
        _reconnect(srv, session)
        return True
    return patch.object(handlers._loop, "_signal_reconnect_and_wait", side_effect=reconnect)


def test_session_expired_reconnect_that_turns_the_tool_write_capable_does_not_replay_it():
    task = _install(_server(HTTP_UNTRUSTED))
    old = task.session
    old.call_tool.side_effect = RuntimeError("Session terminated")
    new = _session(read_only=False)
    with _reconnect_before_retry(task, new):
        result, asked = _call()
    assert "NOT run without approval" in result["error"] and asked == []
    old.call_tool.assert_awaited_once()
    new.call_tool.assert_not_awaited()


def test_session_expired_reconnect_with_unchanged_hint_still_replays():
    task = _install(_server(HTTP_UNTRUSTED))
    task.session.call_tool.side_effect = RuntimeError("Session terminated")
    new = _session(read_only=True)
    with _reconnect_before_retry(task, new):
        result, asked = _call()
    assert result == {"result": "ok"} and asked == []
    new.call_tool.assert_awaited_once()


def test_refused_pre_dispatch_stdio_retry_refunds_and_is_no_breaker_strike():
    """The child was dead before dispatch, so nothing ran: the refused retry keeps no charge and no strike."""
    task = _install(_server(COSTED))
    old = task.session
    new = _session(read_only=False)
    with patch.object(task, "_stdio_children_dead", side_effect=[True, False, False]), \
         _reconnect_before_retry(task, new):
        result, _ = _call()
    assert "NOT run without approval" in result["error"]
    old.call_tool.assert_not_called()
    new.call_tool.assert_not_called()
    assert sum(mcp_tool_costs._spent.values()) == 0
    assert sum(mcp_tool._server_error_counts.values()) == 0
