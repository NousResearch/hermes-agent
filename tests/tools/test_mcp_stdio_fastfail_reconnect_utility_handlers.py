"""Regression tests: the #81995 stdio-dead-child fast-fail/respawn-retry
protection (already covered for ``tools/call`` in
tests/tools/test_mcp_stdio_fastfail_reconnect.py) never reached the four
"generated utility" RPC handlers registered in the same MCP tool namespace:
``resources/list``, ``resources/read``, ``prompts/list``, ``prompts/get``.

A gateway restart kills every MCP stdio subprocess. Before this fix, a call
to any of these four RPC types made right after such a restart rode out the
full ``tool_timeout`` instead of failing fast and auto-recovering, exactly
reproducing the pre-#81995 bug for these four RPC types while ``tools/call``
had already been fixed.

Mirrors the fixture shape of test_mcp_stdio_fastfail_reconnect.py (stub
server + reconnect-event adapter), parameterized over the four handler
factories and their session RPC method names.
"""

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp")

from tools import mcp_tool  # noqa: E402


def _install_stub_server(name: str, session_method_name: str, rpc_impl,
                          *, children_dead, on_reconnect=None):
    """Fake MCP server with real-bool stdio liveness and a countable
    reconnect event (mirrors test_mcp_stdio_fastfail_reconnect.py)."""
    server = MagicMock()
    server.name = name
    session = MagicMock()
    setattr(session, session_method_name, rpc_impl)
    server.session = session

    ready_flag = threading.Event()
    ready_flag.set()

    class _ReconnectAdapter:
        def __init__(self):
            self.set_calls = 0

        def set(self):
            self.set_calls += 1
            if on_reconnect is not None:
                on_reconnect(server)

    server._reconnect_event = _ReconnectAdapter()
    server._ready = ready_flag
    server._is_recycled_stdio.return_value = False
    server._stdio_children_dead = children_dead

    mcp_tool._servers[name] = server
    mcp_tool._server_error_counts.pop(name, None)
    return server


def _cleanup(name: str) -> None:
    mcp_tool._servers.pop(name, None)
    mcp_tool._server_error_counts.pop(name, None)


def _resources_page(items):
    result = MagicMock()
    result.resources = items
    result.next_cursor = None
    return result


def _prompts_page(items):
    result = MagicMock()
    result.prompts = items
    result.next_cursor = None
    return result


def _read_resource_result(text="ok"):
    result = MagicMock()
    block = MagicMock()
    block.text = text
    block.blob = None
    result.contents = [block]
    return result


def _get_prompt_result(text="ok"):
    result = MagicMock()
    msg = MagicMock()
    msg.role = "user"
    msg.content = text
    result.messages = [msg]
    result.description = None
    return result


# Each case: (handler_factory_name, session_method_name, args, success_impl,
#             hanging_impl, assert_success)
_CASES = [
    (
        "_make_list_resources_handler",
        "list_resources",
        {},
        lambda *a, **kw: _resources_page([]),
        lambda parsed: parsed.get("resources") == [],
    ),
    (
        "_make_read_resource_handler",
        "read_resource",
        {"uri": "file:///x"},
        lambda *a, **kw: _read_resource_result("ok"),
        lambda parsed: parsed.get("result") == "ok",
    ),
    (
        "_make_list_prompts_handler",
        "list_prompts",
        {},
        lambda *a, **kw: _prompts_page([]),
        lambda parsed: parsed.get("prompts") == [],
    ),
    (
        "_make_get_prompt_handler",
        "get_prompt",
        {"name": "p1"},
        lambda *a, **kw: _get_prompt_result("ok"),
        lambda parsed: parsed.get("messages") == [{"role": "user", "content": "ok"}],
    ),
]


@pytest.mark.parametrize(
    "factory_name,method_name,args,success_impl,assert_success", _CASES,
)
def test_precall_dead_children_respawn_and_retry(
    monkeypatch, tmp_path, factory_name, method_name, args, success_impl,
    assert_success,
):
    """Dead-at-call-time subprocess: respawn, retry once, clean result —
    no error reaches the model, for every utility RPC type."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handler_factory = getattr(mcp_tool, factory_name)

    called = {"n": 0}
    alive = {"v": False}

    async def _rpc(*a, **kw):
        called["n"] += 1
        return success_impl(*a, **kw)

    def _respawn(server):
        alive["v"] = True
        new_session = MagicMock()
        setattr(new_session, method_name, _rpc)
        server.session = new_session
        server._ready.set()

    name = f"srv-dead-{method_name}"
    server = _install_stub_server(
        name, method_name, _rpc,
        children_dead=lambda: not alive["v"],
        on_reconnect=_respawn,
    )
    mcp_tool._ensure_mcp_loop()
    try:
        handler = handler_factory(name, 10.0)
        parsed = json.loads(handler(dict(args)))
        assert "error" not in parsed, parsed
        assert assert_success(parsed), parsed
        assert server._reconnect_event.set_calls == 1
        assert called["n"] == 1, "exactly one RPC — the retry after respawn"
        assert mcp_tool._server_error_counts.get(name, 0) == 0
    finally:
        _cleanup(name)


@pytest.mark.parametrize(
    "factory_name,method_name,args,success_impl,assert_success", _CASES,
)
def test_midcall_child_exit_respawn_and_retry(
    monkeypatch, tmp_path, factory_name, method_name, args, success_impl,
    assert_success,
):
    """Subprocess dies while the RPC is in flight → respawn and retry once,
    for every utility RPC type."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handler_factory = getattr(mcp_tool, factory_name)

    alive = {"v": True}

    async def _hanging_rpc(*a, **kw):
        await asyncio.sleep(30)

    async def _good_rpc(*a, **kw):
        return success_impl(*a, **kw)

    async def _watch_children():
        while alive["v"]:
            await asyncio.sleep(0.05)

    def _respawn(server):
        alive["v"] = True
        new_session = MagicMock()
        setattr(new_session, method_name, _good_rpc)
        server.session = new_session
        server._ready.set()

    name = f"srv-midcall-{method_name}"
    server = _install_stub_server(
        name, method_name, _hanging_rpc,
        children_dead=lambda: not alive["v"],
        on_reconnect=_respawn,
    )
    server._watch_stdio_children = _watch_children
    mcp_tool._ensure_mcp_loop()
    try:
        handler = handler_factory(name, 10.0)
        alive["v"] = False
        parsed = json.loads(handler(dict(args)))
        assert "error" not in parsed, parsed
        assert assert_success(parsed), parsed
        assert server._reconnect_event.set_calls == 1
    finally:
        _cleanup(name)
