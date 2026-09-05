"""Tests for the MCP tool-listing guard (#101669).

The mcp 2.x SDK validates a ``tools/list`` page as a whole, so one tool the
negotiated wire schema rejects fails the entire response and Hermes parks
the server: the reporter lost 311 good tools to one. The guard sits between
the transport and ``ClientSession`` and validates each tool on its own
against the SDK's per-version ``Tool`` model, so only genuinely invalid
tools are dropped — by name — and the rest of the catalog loads.

Accepting the reporter's *boolean property subschema* is the SDK model's
job (upstream widened it; Hermes carries the same widening for the pinned
release). These tests therefore use a tool every era rejects
(``inputSchema.type != "object"``) as the invalid case, and treat the
boolean tool as "kept once the model accepts it, isolated by name until
then".

The end-to-end tests drive a *real* ``ClientSession`` over in-memory streams
against a fake server; no subprocess, no network.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace

import pytest

from tools.mcp_listing_guard import ToolListingGuard, drop_invalid_tools

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
# ToolListingGuard as a read stream
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


def _response(result):
    return SimpleNamespace(message=SimpleNamespace(result=result), metadata=None)


def _listing(*tools):
    return _response({"tools": [copy.deepcopy(t) for t in tools]})


async def _drain(guard):
    # The SDK's dispatcher consumes the stream with ``async with`` +
    # ``async for``, so exercise that path rather than ``receive()``.
    async with guard:
        return [item async for item in guard]


class TestToolListingGuardStream:
    def test_non_listing_items_pass_through_by_identity(self):
        init = _response({"protocolVersion": LEGACY_VERSION, "capabilities": {}})
        transport_error = ConnectionResetError("boom")
        request = SimpleNamespace(message=SimpleNamespace(method="ping", id=1), metadata=None)
        guard = ToolListingGuard(
            _ScriptedStream([init, transport_error, request]), server_name="srv",
        )

        items = asyncio.run(_drain(guard))
        assert items[0] is init
        assert items[1] is transport_error
        assert items[2] is request

    def test_listing_is_isolated_in_place_before_the_consumer_sees_it(self):
        pytest.importorskip("mcp_types.methods")
        listing = _listing(GOOD, BROKEN)
        guard = ToolListingGuard(
            _ScriptedStream([listing]), server_name="srv", version_getter=lambda: LEGACY_VERSION,
        )

        (item,) = asyncio.run(_drain(guard))
        assert item is listing
        assert [t["name"] for t in item.message.result["tools"]] == ["good"]

    def test_mcp_1x_root_wrapped_messages_are_handled_too(self):
        pytest.importorskip("mcp_types.methods")
        inner = SimpleNamespace(result={"tools": [copy.deepcopy(GOOD), copy.deepcopy(BROKEN)]})
        item = SimpleNamespace(message=SimpleNamespace(root=inner), metadata=None)
        guard = ToolListingGuard(
            _ScriptedStream([item]), server_name="srv", version_getter=lambda: LEGACY_VERSION,
        )

        asyncio.run(guard.receive())
        assert [t["name"] for t in inner.result["tools"]] == ["good"]

    def test_delegates_protocol_extras_to_the_wrapped_stream(self):
        inner = _ScriptedStream([])
        guard = ToolListingGuard(inner, server_name="srv")

        async def lifecycle():
            async with guard:
                pass
            await guard.aclose()

        asyncio.run(lifecycle())
        assert guard.last_context == "sender-context"
        assert inner.entered == 1
        assert inner.closed is True
        with pytest.raises(AttributeError):
            guard._not_an_attribute  # private names are never forwarded

    def test_guard_failure_never_breaks_the_stream(self):
        """A shape the isolation code chokes on must be forwarded, not raised."""
        class _Explosive(dict):
            def get(self, *_a, **_k):
                raise RuntimeError("malformed")

        item = _response(_Explosive(tools=[]))
        guard = ToolListingGuard(_ScriptedStream([item]), server_name="srv")

        assert asyncio.run(guard.receive()) is item

    def test_drops_invalid_tools_using_the_negotiated_version(self, caplog):
        pytest.importorskip("mcp_types.methods")
        listing = _listing(GOOD, BROKEN)
        seen = []
        guard = ToolListingGuard(
            _ScriptedStream([listing]), server_name="srv",
            version_getter=lambda: LEGACY_VERSION, on_drop=seen.append,
        )

        with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
            asyncio.run(_drain(guard))

        assert [t["name"] for t in listing.message.result["tools"]] == ["good"]
        assert [name for name, _ in seen[0]] == ["broken"]
        assert any("'broken'" in rec.getMessage() and "srv" in rec.getMessage()
                   for rec in caplog.records)

    def test_falls_back_to_the_version_the_server_announced(self):
        """Before the session exposes a negotiated version, the handshake
        response that passed through this same stream is the next best
        source — the SDK adopts exactly that value."""
        pytest.importorskip("mcp_types.methods")
        handshake = _response({"protocolVersion": LEGACY_VERSION, "capabilities": {"tools": {}}})
        listing = _listing(GOOD, BROKEN)
        guard = ToolListingGuard(
            _ScriptedStream([handshake, listing]), server_name="srv",
            version_getter=lambda: None,
        )

        asyncio.run(_drain(guard))
        assert [t["name"] for t in listing.message.result["tools"]] == ["good"]

    def test_without_any_version_the_page_is_untouched(self):
        listing = _listing(GOOD, BROKEN)
        guard = ToolListingGuard(
            _ScriptedStream([listing]), server_name="srv", version_getter=lambda: None,
        )

        asyncio.run(_drain(guard))
        assert [t["name"] for t in listing.message.result["tools"]] == ["good", "broken"]

    def test_repeated_verdicts_are_reported_once(self, caplog):
        """A keepalive that re-lists tools every few seconds must not repeat
        the same WARNING into agent.log on every tick."""
        pytest.importorskip("mcp_types.methods")
        pages = [_listing(GOOD, BROKEN) for _ in range(3)]
        guard = ToolListingGuard(
            _ScriptedStream(pages), server_name="srv", version_getter=lambda: LEGACY_VERSION,
        )

        with caplog.at_level(logging.DEBUG, logger=GUARD_LOGGER):
            asyncio.run(_drain(guard))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "'broken'" in warnings[0].getMessage()
        # Every page was still isolated, not just the first.
        for page in pages:
            assert [t["name"] for t in page.message.result["tools"]] == ["good"]


# ---------------------------------------------------------------------------
# MCPServerTask wiring
# ---------------------------------------------------------------------------

class TestServerTaskWiring:
    def test_guard_reads_the_live_session_version_and_records_drops(self):
        pytest.importorskip("mcp_types.methods")
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("wired")
        listing = _listing(GOOD, BROKEN)
        guard = server._guard_read_stream(_ScriptedStream([listing]))
        assert isinstance(guard, ToolListingGuard)

        # The session is negotiated after the stream is wrapped; the guard
        # must pick the version up from whatever session is live at read time.
        server.session = SimpleNamespace(protocol_version=LEGACY_VERSION)
        asyncio.run(_drain(guard))

        assert [t["name"] for t in listing.message.result["tools"]] == ["good"]
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


# ---------------------------------------------------------------------------
# End to end: a real ClientSession over memory streams
# ---------------------------------------------------------------------------

CATALOG = [GOOD, BOOL_PROP, BROKEN]


def _unwrap(message):
    # mcp 1.x wraps JSON-RPC messages in a root model; 2.x does not.
    return getattr(message, "root", message)


def _wrap(types_mod, response):
    cls = getattr(types_mod, "JSONRPCMessage", None)
    if isinstance(cls, type) and "root" in getattr(cls, "model_fields", {}):
        return cls(root=response)
    return response


async def _fake_server(recv, send, catalog):
    """Answer ``initialize`` and ``tools/list``; ignore notifications."""
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
            else:
                result = {}
            response = types.JSONRPCResponse(jsonrpc="2.0", id=request_id, result=result)
            await send.send(SessionMessage(message=_wrap(types, response)))


async def _list_tools_through(catalog, *, guarded):
    import anyio
    from mcp.client.session import ClientSession

    client_send, server_recv = anyio.create_memory_object_stream(64)
    server_send, client_recv = anyio.create_memory_object_stream(64)
    server = asyncio.create_task(_fake_server(server_recv, server_send, catalog))
    live = {}
    read = client_recv
    if guarded:
        read = ToolListingGuard(
            client_recv, server_name="fake",
            version_getter=lambda: getattr(live.get("session"), "protocol_version", None),
        )
    try:
        async with ClientSession(read, client_send) as session:
            live["session"] = session
            await asyncio.wait_for(session.initialize(), timeout=5)
            result = await asyncio.wait_for(session.list_tools(), timeout=5)
            return [tool.name for tool in result.tools]
    finally:
        server.cancel()


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
            asyncio.run(_list_tools_through([GOOD, BROKEN], guarded=False))

        leaves = _leaf_exceptions(excinfo.value)
        assert any(isinstance(leaf, ValidationError) for leaf in leaves), leaves
        assert any("inputSchema" in str(leaf) for leaf in leaves)

    def test_guarded_session_loads_the_catalog_minus_the_invalid_tool(self, caplog):
        with caplog.at_level(logging.WARNING, logger=GUARD_LOGGER):
            names = asyncio.run(_list_tools_through(CATALOG, guarded=True))

        expected = ["good"]
        if _sdk_accepts_boolean_property_schemas():
            expected.append("posthogmcp_endpoint_run")
        assert names == expected
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("'broken'" in m for m in messages)
        assert not any("'good'" in m for m in messages)
