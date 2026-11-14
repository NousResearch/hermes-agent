"""The ``trust: untrusted`` gate decides against the metadata of the session that runs the call (#144380).

The gate classifies from the cached ``readOnlyHint`` before any transport work; a lazy start or reconnect
(in acquire, or in a recoverer before its retry) re-runs discovery and can report the tool as write-capable.
Such a call must not reach the server without consent, and a refused call must not keep its spend charge.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool, mcp_tool_costs
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_registration as registration
from tools.approval_context import reset_current_session_key, set_current_session_key


def _run_on_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _run():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_run())
    finally:
        loop.close()


@pytest.fixture
def session():
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], isError=False, structuredContent=None)
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=result)
    server = SimpleNamespace(session=session, _rpc_lock=None, _reconnect_event=object())
    token = set_current_session_key("chat-a")
    with patch.dict(mcp_tool._servers, {"srv": server}), \
         patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=_run_on_loop), \
         patch.dict(mcp_tool._server_error_counts, {}, clear=True), \
         patch.dict(mcp_tool._server_trust_levels, {}, clear=True), \
         patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True), \
         patch.dict(mcp_tool._server_cost_policies, {}, clear=True), \
         patch.dict(mcp_tool_costs._spent, {}, clear=True), \
         patch.dict(mcp_tool_costs._ask_locks, {}, clear=True), \
         patch.object(mcp_tool_costs, "_unscoped_moved_to", None):
        yield session
    reset_current_session_key(token)


def _discover(read_only: bool, config=None):
    """What a (re)discovery records: the tool's readOnlyHint as the server now reports it."""
    tool = SimpleNamespace(name="sync", annotations={"readOnlyHint": read_only})
    registration._record_tool_trust_metadata("srv", config or {"trust": "untrusted"}, [tool])


def _rediscover_during_acquire(read_only: bool):
    real = handlers._acquire_call_server

    def acquire(server_name, tool_timeout):
        _discover(read_only)
        return real(server_name, tool_timeout)
    return patch.object(handlers, "_acquire_call_server", side_effect=acquire)


def _call(answer="accept"):
    asked = []
    with patch("tools.approval_prompt.request_elicitation_consent",
               side_effect=lambda msg, desc, **kw: asked.append(msg) or answer):
        raw = handlers._make_tool_handler("srv", "sync", 30.0)({"target": "x"})
    return json.loads(raw), asked


def test_tool_rediscovered_write_capable_during_acquire_asks_and_deny_blocks(session):
    _discover(True)
    with _rediscover_during_acquire(False):
        result, asked = _call(answer="decline")
    assert len(asked) == 1 and "UNTRUSTED server 'srv'" in asked[0]
    assert "NOT run" in result["error"]
    session.call_tool.assert_not_awaited()


def test_tool_rediscovered_write_capable_runs_once_approved(session):
    _discover(True)
    with _rediscover_during_acquire(False):
        result, asked = _call(answer="accept")
    assert result == {"result": "ok"} and len(asked) == 1
    session.call_tool.assert_awaited_once()


def test_unchanged_read_only_hint_never_asks(session):
    _discover(True)
    with _rediscover_during_acquire(True):
        result, asked = _call()
    assert result == {"result": "ok"} and asked == []


def test_denied_recheck_refunds_the_gate_booked_charge(session):
    config = {"trust": "untrusted", "tool_costs": {"sync": 40}, "spend_limit": 100}
    _discover(True, config)
    real = handlers._acquire_call_server

    def acquire(server_name, tool_timeout):
        _discover(False, config)
        return real(server_name, tool_timeout)
    with patch.object(handlers, "_acquire_call_server", side_effect=acquire):
        result, _ = _call(answer="decline")
    assert "error" in result
    assert sum(mcp_tool_costs._spent.values()) == 0
    session.call_tool.assert_not_awaited()


def test_reconnect_before_a_retry_that_makes_the_tool_write_capable_does_not_replay_it(session):
    """Session expiry replays a read-only call after a reconnect; that reconnect's rediscovery can turn the
    tool write-capable, and the replay must then not run without consent."""
    _discover(True)
    session.call_tool.side_effect = [RuntimeError("Session terminated"), session.call_tool.return_value]

    def reconnect(server_name, srv, **kwargs):
        _discover(False)
        return True
    with patch.object(handlers._loop, "_signal_reconnect_and_wait", side_effect=reconnect), \
         patch.object(handlers, "_mcp_loop_running", return_value=True):
        result, _ = _call()
    assert "NOT run without approval" in result["error"]
    assert session.call_tool.await_count == 1


def test_retry_after_reconnect_with_unchanged_hint_still_replays(session):
    _discover(True)
    session.call_tool.side_effect = [RuntimeError("Session terminated"), session.call_tool.return_value]
    with patch.object(handlers._loop, "_signal_reconnect_and_wait", return_value=True), \
         patch.object(handlers, "_mcp_loop_running", return_value=True):
        result, asked = _call()
    assert result == {"result": "ok"} and asked == []
    assert session.call_tool.await_count == 2
