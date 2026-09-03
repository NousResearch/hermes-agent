"""Request-family and lifecycle tests for MCP generation admission."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tools import mcp_tool
from tools.mcp_tool import MCPServerTask, _track_inflight_rpc


class _RealHandlerSession:
    def __init__(self, server: MCPServerTask) -> None:
        self.server = server
        self.calls: list[str] = []

    def _record(self, method: str) -> None:
        self.calls.append(method)
        assert asyncio.current_task() in self.server._inflight_tasks
        assert self.server._admitting_generation == self.server._rpc_generation

    async def call_tool(self, name, arguments):
        self._record("call_tool")
        return SimpleNamespace(content=[], is_error=False)

    async def list_resources(self, cursor=None):
        self._record("list_resources")
        return SimpleNamespace(resources=[], nextCursor=None)

    async def read_resource(self, uri):
        self._record("read_resource")
        return SimpleNamespace(contents=[])

    async def list_prompts(self, cursor=None):
        self._record("list_prompts")
        return SimpleNamespace(prompts=[], nextCursor=None)

    async def get_prompt(self, name, arguments):
        self._record("get_prompt")
        return SimpleNamespace(messages=[])


_REQUEST_FAMILIES = [
    ("_make_tool_handler", ("probe", 5.0), {}, "call_tool"),
    ("_make_list_resources_handler", (5.0,), {}, "list_resources"),
    (
        "_make_read_resource_handler",
        (5.0,),
        {"uri": "test://resource"},
        "read_resource",
    ),
    ("_make_list_prompts_handler", (5.0,), {}, "list_prompts"),
    ("_make_get_prompt_handler", (5.0,), {"name": "probe"}, "get_prompt"),
]


@pytest.mark.parametrize(
    ("builder_name", "builder_args", "request_args", "session_method"),
    _REQUEST_FAMILIES,
)
def test_real_handlers_track_open_generation_and_refuse_closed_generation(
    monkeypatch, builder_name, builder_args, request_args, session_method
):
    """Every public request family uses the same admission/tracking boundary."""
    server = MCPServerTask("handlers")
    session = _RealHandlerSession(server)
    server._publish_session(session)

    monkeypatch.setattr(
        mcp_tool, "_get_connected_server_for_call", lambda _name: server
    )
    monkeypatch.setattr(
        mcp_tool,
        "_run_on_mcp_loop",
        lambda coroutine_factory, timeout: asyncio.run(coroutine_factory()),
    )
    monkeypatch.setattr(MCPServerTask, "_watch_stdio_children", lambda self: None)

    handler = getattr(mcp_tool, builder_name)("handlers", *builder_args)
    admitted = json.loads(handler(request_args))
    assert "error" not in admitted
    assert session.calls == [session_method]
    assert not server._inflight_tasks

    server._close_rpc_admission()
    refused = json.loads(handler(request_args))
    assert "retry the request on the rebuilt session" in refused["error"]
    assert session.calls == [session_method]
    assert not server._inflight_tasks


@pytest.mark.asyncio
async def test_stale_generation_never_admits_even_with_published_session():
    server = MCPServerTask("stale")
    published = object()
    server._publish_session(published)
    generation = server._rpc_generation

    server._admitting_generation = generation - 1
    with pytest.raises(RuntimeError, match="retry the request"):
        async with _track_inflight_rpc(server, server.name, "resources/list"):
            pytest.fail("stale generation entered the RPC body")

    assert server.session is published
    assert server._rpc_generation == generation
    assert not server._inflight_tasks


@pytest.mark.asyncio
async def test_close_is_idempotent_and_does_not_change_session_generation():
    server = MCPServerTask("invariants")
    published = object()
    server._publish_session(published)
    generation = server._rpc_generation

    server._close_rpc_admission()
    server._close_rpc_admission()

    assert server.session is published
    assert server._rpc_generation == generation
    assert server._admitting_generation is None


class _AsyncStreamsContext:
    async def __aenter__(self):
        return object(), object()

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _BlockedSessionContext:
    def __init__(self, session) -> None:
        self.session = session
        self.exit_started = asyncio.Event()
        self.release_exit = asyncio.Event()

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, traceback):
        self.exit_started.set()
        await self.release_exit.wait()
        return False


@pytest.mark.asyncio
async def test_admission_closes_before_blocked_client_session_exit(monkeypatch):
    """A late RPC is refused while ClientSession.__aexit__ is still blocked."""
    server = MCPServerTask("blocked-exit")
    published = SimpleNamespace()
    session_context = _BlockedSessionContext(published)

    monkeypatch.setattr(mcp_tool, "sse_client", lambda **kwargs: _AsyncStreamsContext())
    monkeypatch.setattr(
        mcp_tool, "ClientSession", lambda *args, **kwargs: session_context
    )

    async def _negotiate(self, session, timeout):
        return SimpleNamespace()

    async def _discover(self):
        return None

    async def _wait(self):
        return "shutdown"

    monkeypatch.setattr(MCPServerTask, "_negotiate_session", _negotiate)
    monkeypatch.setattr(MCPServerTask, "_discover_tools", _discover)
    monkeypatch.setattr(MCPServerTask, "_wait_for_lifecycle_event", _wait)

    run_task = asyncio.create_task(
        server._run_http({
            "url": "https://example.invalid/mcp",
            "transport": "sse",
            "connect_timeout": 1.0,
        })
    )
    await asyncio.wait_for(session_context.exit_started.wait(), timeout=2.0)

    assert server.session is published
    assert server._rpc_generation == 1
    assert server._admitting_generation is None
    with pytest.raises(RuntimeError, match="retry the request"):
        async with _track_inflight_rpc(server, server.name, "prompts/get"):
            pytest.fail("late RPC entered while ClientSession.__aexit__ was blocked")

    session_context.release_exit.set()
    assert await asyncio.wait_for(run_task, timeout=2.0) == "shutdown"
