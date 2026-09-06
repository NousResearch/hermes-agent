"""Tests for the MCP tool-listing guard (#101669).

The mcp 2.x SDK validates a ``tools/list`` page as a whole, so one tool the
negotiated wire schema rejects fails the entire response and Hermes parks
the server: the reporter lost 311 good tools to one. The guard wraps the
session's stream pair between the transport and ``ClientSession``: the
write proxy remembers each outbound request's method by id, and the read
proxy validates every tool of the response to ``tools/list`` — and only
that response — against the SDK's per-version ``Tool`` model, so only
genuinely invalid tools are dropped, by name, and the rest of the catalog
loads.

Accepting the reporter's *boolean property subschema* is the SDK model's
job and is owned by a separate change (upstream python-sdk#3354; #102900 for
the pinned release) — this guard neither accepts nor rewrites schemas. These
tests therefore use a tool every era rejects (``inputSchema.type !=
"object"``) as the invalid case, and treat the boolean tool as "kept once
the model in use accepts it, isolated by name until then".

The end-to-end tests drive a *real* ``ClientSession`` over in-memory streams
against a fake server; no subprocess, no network.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from tools.mcp_listing_guard import (
    GuardedWriteStream,
    ToolListingGuard,
    drop_invalid_tools,
    guard_session_streams,
)

# The protocol era most servers still negotiate.
LEGACY_VERSION = "2025-11-25"
GUARD_LOGGER = "tools.mcp_listing_guard"

GOOD = {"name": "good", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}
# Genuinely invalid for every era's tools/list wire schema: root must be an object.
BROKEN = {"name": "broken", "inputSchema": {"type": "array"}}
# The reporter's case: a boolean property subschema (legal JSON Schema).
BOOL_PROP = {"name": "posthogmcp_endpoint_run", "inputSchema": {"type": "object", "properties": {"refresh": True}}}


def _sdk_accepts_boolean_property_schemas() -> bool:
    """Whether the installed SDK model (with any Hermes widening) takes the
    reporter's shape — the guard's verdict must track the model, not
    hard-code either answer."""
    from mcp_types.methods import validate_server_result
    from pydantic import ValidationError

    try:
        validate_server_result("tools/list", LEGACY_VERSION, {"tools": [copy.deepcopy(BOOL_PROP)]})
    except ValidationError:
        return False
    return True


# ---------------------------------------------------------------------------
# drop_invalid_tools — needs the SDK generation with per-version validation
# ---------------------------------------------------------------------------

class TestDropInvalidTools:
    @pytest.fixture(autouse=True)
    def _needs_versioned_sdk(self):
        pytest.importorskip("mcp_types.methods")

    def test_valid_page_is_left_alone(self):
        result = {"tools": [copy.deepcopy(GOOD), {"name": "b", "inputSchema": {"type": "object"}}]}
        before = copy.deepcopy(result)

        assert drop_invalid_tools(result, LEGACY_VERSION) == []
        assert result == before

    def test_only_the_offending_tool_is_dropped_and_named(self):
        result = {"tools": [
            copy.deepcopy(GOOD), copy.deepcopy(BROKEN),
            {"name": "also_good", "inputSchema": {"type": "object", "properties": {}}},
        ]}
        dropped = drop_invalid_tools(result, LEGACY_VERSION)

        assert [name for name, _ in dropped] == ["broken"]
        assert "inputSchema" in dropped[0][1]
        assert [t["name"] for t in result["tools"]] == ["good", "also_good"]

    def test_verdict_tracks_the_sdk_model_for_boolean_property_schemas(self):
        """Once the model accepts booleans (upstream #3354 / the Hermes
        widening) the tool stays; until then it is isolated, not the server."""
        result = {"tools": [copy.deepcopy(GOOD), copy.deepcopy(BOOL_PROP)]}
        dropped = drop_invalid_tools(result, LEGACY_VERSION)
        names = [t["name"] for t in result["tools"]]

        if _sdk_accepts_boolean_property_schemas():
            assert dropped == [] and names == ["good", "posthogmcp_endpoint_run"]
        else:
            assert [n for n, _ in dropped] == ["posthogmcp_endpoint_run"]
            assert "properties.refresh" in dropped[0][1]
            assert names == ["good"]

    def test_page_level_errors_neither_hide_nor_condemn_tools(self):
        """A broken page-level field (the SDK clamps a negative ttl itself,
        and rejects a non-string cursor on its own) must not stop an invalid
        tool from being dropped, nor take a valid tool with it."""
        result = {
            "tools": [{"name": "fine", "inputSchema": {"type": "object"}}, copy.deepcopy(BROKEN)],
            "nextCursor": 12345,
            "ttlMs": -5,
        }
        dropped = drop_invalid_tools(result, LEGACY_VERSION)

        assert [name for name, _ in dropped] == ["broken"]
        assert [t["name"] for t in result["tools"]] == ["fine"]
        assert result["nextCursor"] == 12345
        assert result["ttlMs"] == -5

    def test_unknown_version_defers_to_the_sdk(self):
        result = {"tools": [copy.deepcopy(BROKEN)]}
        assert drop_invalid_tools(result, "9999-01-01") == []
        assert len(result["tools"]) == 1

    def test_non_listing_input_is_untouched(self):
        assert drop_invalid_tools({"tools": "nope"}, LEGACY_VERSION) == []
        assert drop_invalid_tools({"resources": []}, LEGACY_VERSION) == []


# ---------------------------------------------------------------------------
# The guarded stream pair
# ---------------------------------------------------------------------------

class _ScriptedStream:
    """Minimal ReadStream stand-in yielding a fixed sequence of items."""

    last_context = "sender-context"

    def __init__(self, items):
        self._items = list(items)
        self.entered = 0
        self.closed = False

    async def receive(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.receive()

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *exc):
        return False

    async def aclose(self):
        self.closed = True


class _RecordingWrite:
    """Minimal WriteStream stand-in that records what was sent."""

    def __init__(self):
        self.sent = []
        self.entered = 0
        self.closed = False

    async def send(self, item):
        self.sent.append(item)

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *exc):
        return False

    async def aclose(self):
        self.closed = True


def _request(request_id, method):
    return SimpleNamespace(message=SimpleNamespace(id=request_id, method=method, params=None), metadata=None)


def _notification(method):
    return SimpleNamespace(message=SimpleNamespace(method=method, params=None), metadata=None)


def _response(request_id, result):
    return SimpleNamespace(message=SimpleNamespace(id=request_id, result=result), metadata=None)


def _error(request_id):
    return SimpleNamespace(message=SimpleNamespace(id=request_id, error={"code": -1}), metadata=None)


def _listing(request_id, *tools):
    return _response(request_id, {"tools": [copy.deepcopy(t) for t in tools]})


def _guarded(inbound, *, outbound=(), version=LEGACY_VERSION, on_drop=None):
    """A guarded pair whose write side has already seen ``outbound`` requests."""
    inner_write = _RecordingWrite()
    read, write = guard_session_streams(
        _ScriptedStream(inbound), inner_write, server_name="srv",
        version_getter=lambda: version, on_drop=on_drop,
    )

    async def prime():
        for item in outbound:
            await write.send(item)

    asyncio.run(prime())
    return read, write, inner_write


async def _drain(guard):
    # The SDK's dispatcher consumes the stream with ``async with`` +
    # ``async for``, so exercise that path rather than ``receive()``.
    async with guard:
        return [item async for item in guard]


def _names(item):
    return [t["name"] for t in item.message.result["tools"]]


class TestGuardedStreamPair:
    def test_only_the_response_to_tools_list_is_isolated(self):
        pytest.importorskip("mcp_types.methods")
        listing = _listing(1, GOOD, BROKEN)
        custom = _response(2, {"tools": [copy.deepcopy(BROKEN)], "note": "extension payload"})
        read, _, _ = _guarded([listing, custom], outbound=[_request(1, "tools/list"), _request(2, "x/custom")])

        items = asyncio.run(_drain(read))

        assert items == [listing, custom]
        assert _names(listing) == ["good"]
        assert custom.message.result == {"tools": [BROKEN], "note": "extension payload"}

    def test_a_tools_shaped_result_without_a_known_request_is_untouched(self):
        pytest.importorskip("mcp_types.methods")
        orphan = _listing(99, GOOD, BROKEN)
        read, _, _ = _guarded([orphan], outbound=[_request(1, "tools/list")])

        asyncio.run(_drain(read))
        assert _names(orphan) == ["good", "broken"]

    def test_request_ids_are_retired_by_their_answer(self):
        """An id answered by an error (or a response) must not linger and
        later claim an unrelated result that reuses the id."""
        pytest.importorskip("mcp_types.methods")
        errored = _error(1)
        reused = _listing(1, GOOD, BROKEN)  # same id, but no tools/list request behind it now
        read, _, _ = _guarded([errored, reused], outbound=[_request(1, "tools/list")])

        items = asyncio.run(_drain(read))
        assert items[0] is errored
        assert _names(reused) == ["good", "broken"]

    def test_server_originated_requests_and_notifications_pass_through(self):
        inbound = [
            _request(7, "sampling/createMessage"),  # server → client request
            _notification("notifications/tools/list_changed"),
            ConnectionResetError("boom"),
        ]
        read, _, _ = _guarded(inbound, outbound=[_request(7, "tools/list")])

        items = asyncio.run(_drain(read))
        assert items == inbound

    def test_write_proxy_forwards_every_send(self):
        read, write, inner_write = _guarded([], outbound=[])
        outbound = [_request(1, "ping"), _notification("notifications/initialized")]

        async def send_all():
            for item in outbound:
                await write.send(item)

        asyncio.run(send_all())
        assert inner_write.sent == outbound
        assert isinstance(write, GuardedWriteStream)

    def test_mcp_1x_root_wrapped_messages_are_handled_too(self):
        pytest.importorskip("mcp_types.methods")
        request = SimpleNamespace(message=SimpleNamespace(root=SimpleNamespace(id=1, method="tools/list")), metadata=None)
        inner = SimpleNamespace(id=1, result={"tools": [copy.deepcopy(GOOD), copy.deepcopy(BROKEN)]})
        listing = SimpleNamespace(message=SimpleNamespace(root=inner), metadata=None)
        read, _, _ = _guarded([listing], outbound=[request])

        asyncio.run(read.receive())
        assert [t["name"] for t in inner.result["tools"]] == ["good"]

    def test_delegates_protocol_extras_to_the_wrapped_streams(self):
        inner_read = _ScriptedStream([])
        inner_write = _RecordingWrite()
        read, write = guard_session_streams(inner_read, inner_write, server_name="srv")

        async def lifecycle():
            async with read, write:
                pass
            await read.aclose()
            await write.aclose()

        asyncio.run(lifecycle())
        assert isinstance(read, ToolListingGuard)
        assert read.last_context == "sender-context"
        assert (inner_read.entered, inner_read.closed) == (1, True)
        assert (inner_write.entered, inner_write.closed) == (1, True)
        with pytest.raises(AttributeError):
            read._not_an_attribute  # private names are never forwarded

    def test_guard_failure_never_breaks_the_stream(self):
        """A shape the isolation code chokes on must be forwarded, not raised."""
        class _Explosive(dict):
            def get(self, *_a, **_k):
                raise RuntimeError("malformed")

        item = _response(1, _Explosive(tools=[]))
        read, _, _ = _guarded([item], outbound=[_request(1, "tools/list")])

        assert asyncio.run(read.receive()) is item

    def test_drops_invalid_tools_using_the_negotiated_version(self, caplog):
        pytest.importorskip("mcp_types.methods")
        listing = _listing(1, GOOD, BROKEN)
        seen = []
        read, _, _ = _guarded([listing], outbound=[_request(1, "tools/list")], on_drop=seen.append)

        with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
            asyncio.run(_drain(read))

        assert _names(listing) == ["good"]
        assert [name for name, _ in seen[0]] == ["broken"]
        assert any("'broken'" in rec.getMessage() and "srv" in rec.getMessage()
                   for rec in caplog.records)

    def test_falls_back_to_the_version_the_handshake_returned(self):
        """Before the session exposes a negotiated version, the handshake
        response that passed through this same stream is the next best
        source — the SDK adopts exactly that value."""
        pytest.importorskip("mcp_types.methods")
        handshake = _response(1, {"protocolVersion": LEGACY_VERSION, "capabilities": {"tools": {}}})
        listing = _listing(2, GOOD, BROKEN)
        read, _, _ = _guarded([handshake, listing], version=None,
                              outbound=[_request(1, "initialize"), _request(2, "tools/list")])

        asyncio.run(_drain(read))
        assert _names(listing) == ["good"]

    def test_a_protocol_version_in_a_non_handshake_result_is_not_adopted(self):
        pytest.importorskip("mcp_types.methods")
        decoy = _response(1, {"protocolVersion": LEGACY_VERSION})
        listing = _listing(2, GOOD, BROKEN)
        read, _, _ = _guarded([decoy, listing], version=None,
                              outbound=[_request(1, "x/custom"), _request(2, "tools/list")])

        asyncio.run(_drain(read))
        assert _names(listing) == ["good", "broken"]

    def test_repeated_verdicts_are_reported_once(self, caplog):
        """A keepalive that re-lists tools every few seconds must not repeat
        the same WARNING into agent.log on every tick."""
        pytest.importorskip("mcp_types.methods")
        pages = [_listing(i, GOOD, BROKEN) for i in range(3)]
        read, _, _ = _guarded(pages, outbound=[_request(i, "tools/list") for i in range(3)])

        with caplog.at_level(logging.DEBUG, logger=GUARD_LOGGER):
            asyncio.run(_drain(read))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "'broken'" in warnings[0].getMessage()
        for page in pages:  # every page was still isolated, not just the first
            assert _names(page) == ["good"]


# ---------------------------------------------------------------------------
# MCPServerTask wiring
# ---------------------------------------------------------------------------

class TestServerTaskWiring:
    def test_guard_reads_the_live_session_version_and_records_drops(self):
        pytest.importorskip("mcp_types.methods")
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("wired")
        listing = _listing(1, GOOD, BROKEN)
        read, write = server._guard_session_streams(_ScriptedStream([listing]), _RecordingWrite())
        assert isinstance(read, ToolListingGuard) and isinstance(write, GuardedWriteStream)

        # The session is negotiated after the streams are wrapped; the guard
        # must pick the version up from whatever session is live at read time.
        server.session = SimpleNamespace(protocol_version=LEGACY_VERSION)

        async def drive():
            await write.send(_request(1, "tools/list"))
            return await _drain(read)

        asyncio.run(drive())
        assert _names(listing) == ["good"]
        assert list(server._dropped_tools) == ["broken"]
        assert "inputSchema" in server._dropped_tools["broken"]

    def test_status_lists_dropped_tools_for_a_connected_server(self, monkeypatch):
        import tools.mcp_tool as mcp_tool
        from tools import mcp_tool_config as _mcp_config
        from tools import mcp_tool_discovery as _mcp_discovery

        server = mcp_tool.MCPServerTask("dropper")
        server.session = SimpleNamespace(protocol_version=LEGACY_VERSION)
        server._record_dropped_tools([("broken", "inputSchema.type: Input should be 'object'")])
        clean = mcp_tool.MCPServerTask("clean")
        clean.session = SimpleNamespace(protocol_version=LEGACY_VERSION)

        monkeypatch.setattr(
            _mcp_config, "_load_mcp_config",
            lambda: {"dropper": {"command": "x"}, "clean": {"command": "y"}},
        )
        with mcp_tool._lock:
            saved = dict(mcp_tool._servers)
            mcp_tool._servers.clear()
            mcp_tool._servers.update({"dropper": server, "clean": clean})
        try:
            statuses = {e["name"]: e for e in _mcp_discovery.get_mcp_status()}
        finally:
            with mcp_tool._lock:
                mcp_tool._servers.clear()
                mcp_tool._servers.update(saved)

        assert statuses["dropper"]["status"] == "connected"
        assert statuses["dropper"]["dropped_tools"] == ["broken"]
        assert "dropped_tools" not in statuses["clean"]

    def test_rediscovery_forgets_stale_drops(self):
        """A tool the server fixed must not haunt status after a refresh."""
        from unittest.mock import MagicMock

        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("refreshing")
        server._record_dropped_tools([("broken", "stale reason")])

        async def fake_list(cursor=None):
            return SimpleNamespace(tools=[])

        server.session = MagicMock()
        server.session.list_tools = fake_list
        asyncio.run(server._discover_tools())

        assert server._dropped_tools == {}

    @staticmethod
    def _server_without_tools_capability(name):
        """A task whose captured handshake omits ``tools`` — ``tools/list`` is never called."""
        from unittest.mock import AsyncMock, MagicMock

        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask(name)
        server._record_dropped_tools([("broken", "verdict from the previous catalog")])
        server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(tools=None))
        server.session = MagicMock()
        server.session.list_tools = AsyncMock(side_effect=AssertionError("tools/list must not be called"))
        return server

    def test_rediscovery_without_tools_capability_forgets_stale_drops(self):
        """A server that reconnects as prompt-/resource-only fetches no catalog;
        the previous catalog's verdicts must not survive into status."""
        server = self._server_without_tools_capability("prompts-only")

        asyncio.run(server._discover_tools())

        assert server._dropped_tools == {}
        server.session.list_tools.assert_not_called()

    def test_refresh_without_tools_capability_forgets_stale_drops(self):
        """Same for a ``tools/list_changed`` refresh that early-returns on the capability gate."""
        server = self._server_without_tools_capability("prompts-only")

        asyncio.run(server._refresh_tools())

        assert server._dropped_tools == {}
        server.session.list_tools.assert_not_called()


# ---------------------------------------------------------------------------
# End to end: a real ClientSession over memory streams
# ---------------------------------------------------------------------------

CATALOG = [GOOD, BOOL_PROP, BROKEN]
# A custom/extension method whose result merely *looks like* a tool catalog. The pinned SDK
# forwards custom results without surface validation, so the guard must leave it alone.
CUSTOM_METHOD = "x/tools_like"
CUSTOM_RESULT = {"tools": [copy.deepcopy(BROKEN)], "note": "extension payload"}


def _unwrap(message):
    # mcp 1.x wraps JSON-RPC messages in a root model; 2.x does not.
    return getattr(message, "root", message)


def _wrap(types_mod, response):
    cls = getattr(types_mod, "JSONRPCMessage", None)
    if isinstance(cls, type) and "root" in getattr(cls, "model_fields", {}):
        return cls(root=response)
    return response


async def _fake_server(recv, send, catalog):
    """Answer ``initialize``, ``tools/list`` and the custom method; ignore notifications."""
    import mcp.types as types
    from mcp.shared.message import SessionMessage

    async with recv, send:
        async for item in recv:
            msg = _unwrap(item.message)
            request_id = getattr(msg, "id", None)
            if request_id is None:
                continue
            method = getattr(msg, "method", None)
            if method == "initialize":
                result = {
                    "protocolVersion": LEGACY_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0"},
                }
            elif method == "tools/list":
                result = {"tools": copy.deepcopy(catalog)}
            elif method == CUSTOM_METHOD:
                result = copy.deepcopy(CUSTOM_RESULT)
            else:
                result = {}
            response = types.JSONRPCResponse(jsonrpc="2.0", id=request_id, result=result)
            await send.send(SessionMessage(message=_wrap(types, response)))


async def _session_through(catalog, *, guarded, use):
    """Run ``use(session)`` on a real ClientSession, optionally behind the guard."""
    import anyio
    from mcp.client.session import ClientSession

    client_send, server_recv = anyio.create_memory_object_stream(64)
    server_send, client_recv = anyio.create_memory_object_stream(64)
    server = asyncio.create_task(_fake_server(server_recv, server_send, catalog))
    live = {}
    read, write = client_recv, client_send
    if guarded:
        read, write = guard_session_streams(
            client_recv, client_send, server_name="fake",
            version_getter=lambda: getattr(live.get("session"), "protocol_version", None),
        )
    try:
        async with ClientSession(read, write) as session:
            live["session"] = session
            await asyncio.wait_for(session.initialize(), timeout=5)
            return await asyncio.wait_for(use(session), timeout=5)
    finally:
        server.cancel()


async def _list_tool_names(session):
    return [tool.name for tool in (await session.list_tools()).tools]


async def _call_custom_method(session):
    import mcp.types as types
    from pydantic import TypeAdapter

    return await session.send_request(
        types.Request(method=CUSTOM_METHOD, params=None), TypeAdapter(dict[str, Any]),
    )


def _leaf_exceptions(exc):
    nested = getattr(exc, "exceptions", None)
    if nested is None:
        return [exc]
    return [leaf for sub in nested for leaf in _leaf_exceptions(sub)]


class TestEndToEndWithRealClientSession:
    @pytest.fixture(autouse=True)
    def _needs_sdk(self):
        pytest.importorskip("mcp")
        pytest.importorskip("mcp_types.methods")

    def test_premise_sdk_rejects_the_whole_page_without_the_guard(self):
        """Pins the failure mode from #101669: one rejected tool, whole page gone."""
        from pydantic import ValidationError

        with pytest.raises(BaseException) as excinfo:
            asyncio.run(_session_through([GOOD, BROKEN], guarded=False, use=_list_tool_names))

        leaves = _leaf_exceptions(excinfo.value)
        assert any(isinstance(leaf, ValidationError) for leaf in leaves), leaves
        assert any("inputSchema" in str(leaf) for leaf in leaves)

    def test_guarded_session_loads_the_catalog_minus_the_invalid_tool(self, caplog):
        with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
            names = asyncio.run(_session_through(CATALOG, guarded=True, use=_list_tool_names))

        expected = ["good"]
        if _sdk_accepts_boolean_property_schemas():
            expected.append("posthogmcp_endpoint_run")
        assert names == expected
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("'broken'" in m for m in messages)
        assert not any("'good'" in m for m in messages)

    def test_custom_method_result_carrying_a_tools_list_is_untouched(self, caplog):
        """The pinned SDK forwards custom-method results as-is; so must the guard."""
        with caplog.at_level(logging.DEBUG, logger=GUARD_LOGGER):
            unguarded = asyncio.run(_session_through(CATALOG, guarded=False, use=_call_custom_method))
            guarded = asyncio.run(_session_through(CATALOG, guarded=True, use=_call_custom_method))

        assert guarded == unguarded == CUSTOM_RESULT
        assert not caplog.records
