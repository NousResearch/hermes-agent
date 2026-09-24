"""Empty ``params._meta`` stripping for MCP HTTP transports.

Exercises ``_make_empty_meta_stripping_transport`` with a real httpx
AsyncClient over a MockTransport: information-free ``params._meta`` (``{}`` or
``null``) is removed from outbound JSON-RPC request bodies before send, while
populated metadata, non-request items, batches, and non-JSON bodies pass
through unchanged. ``_wrap_mcp_transport`` composes the repair with the
response-body cap.
"""

import json

import httpx
import pytest

from tools.mcp_tool_errors import (
    _MCP_HTTP_MAX_BODY_BYTES,
    _make_empty_meta_stripping_transport,
    _wrap_mcp_transport,
)


def _client_for(handler, wrap=_make_empty_meta_stripping_transport):
    inner = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=wrap(httpx, inner))


def _recorder(seen):
    async def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})
    return handler


async def _post(client, payload, content_type="application/json"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return await client.post("http://mcp.test/rpc", content=body,
                             headers={"content-type": content_type})


@pytest.mark.asyncio
async def test_empty_meta_object_stripped():
    seen = []
    async with _client_for(_recorder(seen)) as client:
        await _post(client, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": "x", "_meta": {}}})
    assert seen == [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "x"}}]


@pytest.mark.asyncio
async def test_null_meta_stripped():
    seen = []
    async with _client_for(_recorder(seen)) as client:
        await _post(client, {"jsonrpc": "2.0", "id": 1, "method": "tools/list",
                             "params": {"_meta": None}})
    assert seen == [{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}]


@pytest.mark.asyncio
async def test_populated_meta_preserved():
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"_meta": {"progressToken": "t1"}}}
    seen = []
    async with _client_for(_recorder(seen)) as client:
        await _post(client, payload)
    assert seen == [payload]


@pytest.mark.asyncio
async def test_request_without_meta_untouched():
    payload = {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {"a": 1}}
    seen = []
    async with _client_for(_recorder(seen)) as client:
        await _post(client, payload)
    assert seen == [payload]


@pytest.mark.asyncio
async def test_result_shaped_item_untouched():
    # No "method" key -> not a request item; empty _meta must survive.
    payload = {"jsonrpc": "2.0", "id": 3, "result": {"_meta": {}}}
    seen = []
    async with _client_for(_recorder(seen)) as client:
        await _post(client, payload)
    assert seen == [payload]


@pytest.mark.asyncio
async def test_batch_items_repaired_per_item():
    seen = []
    async with _client_for(_recorder(seen)) as client:
        await _post(client, [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {"_meta": {"traceparent": "x"}}},
        ])
    assert seen == [[
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {"_meta": {"traceparent": "x"}}},
    ]]


@pytest.mark.asyncio
async def test_non_json_content_type_untouched():
    raw = b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"_meta":{}}}'
    seen = []

    async def handler(request):
        seen.append(request.content)
        return httpx.Response(200, json={"ok": True})

    async with _client_for(handler) as client:
        await _post(client, raw, content_type="text/plain")
    assert seen == [raw]


@pytest.mark.asyncio
async def test_unparseable_body_untouched():
    raw = b"not json {"
    seen = []

    async def handler(request):
        seen.append(request.content)
        return httpx.Response(200, json={"ok": True})

    async with _client_for(handler) as client:
        await _post(client, raw)
    assert seen == [raw]


@pytest.mark.asyncio
async def test_wrap_composes_strip_and_body_cap():
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=b"z" * (_MCP_HTTP_MAX_BODY_BYTES + 1),
                              headers={"content-length": str(_MCP_HTTP_MAX_BODY_BYTES + 1)})

    async with _client_for(handler, wrap=_wrap_mcp_transport) as client:
        with pytest.raises(httpx.ReadError, match=r"bytes cap"):
            await _post(client, {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"_meta": {}}})
    assert seen == [{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}]
