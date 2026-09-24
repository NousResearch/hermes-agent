"""Outbound ``params._meta`` stripping for MCP HTTP transports.

Exercises ``_make_empty_meta_stripping_transport`` (via the composed
``_wrap_mcp_transport``) with a real httpx AsyncClient over a MockTransport:
JSON-RPC requests carrying an empty or null ``_meta`` have the key removed
before the body hits the wire, while populated metadata, metadata-free
requests, result-shaped bodies, batches and non-JSON payloads pass through
byte-identical. The composed wrapper still applies the response wire-body cap.
"""

import json

import httpx
import pytest

from tools.mcp_tool_errors import (
    _make_empty_meta_stripping_transport,
    _make_mcp_body_cap_transport,
    _wrap_mcp_transport,
)


def _client_for(handler, wrapper=_make_empty_meta_stripping_transport):
    inner = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=wrapper(httpx, inner))


def _post(client, payload, content_type="application/json"):
    body = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    return client.post("http://mcp.test/rpc", content=body,
                       headers={"Content-Type": content_type})


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_meta", [{}, None])
async def test_empty_or_null_meta_is_stripped_from_requests(empty_meta):
    seen = {}

    async def handler(request):
        seen["body"] = request.content
        seen["content_length"] = request.headers.get("content-length")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    async with _client_for(handler) as client:
        resp = await _post(client, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                    "params": {"_meta": empty_meta, "capabilities": {}}})
        assert resp.status_code == 200
    sent = json.loads(seen["body"])
    assert "_meta" not in sent["params"]  # the information-free form never reaches the wire
    assert sent["params"]["capabilities"] == {}
    assert int(seen["content_length"]) == len(seen["body"])  # framing recomputed for the new body


@pytest.mark.asyncio
async def test_populated_meta_passes_through_unchanged():
    seen = {}
    payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": "run_ads_report",
                          "_meta": {"progressToken": "7", "traceparent": "00-abc-def-01"}}}

    async def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 2, "result": {}})

    async with _client_for(handler) as client:
        await _post(client, payload)
    assert json.loads(seen["body"]) == payload  # populated metadata is forwarded verbatim


@pytest.mark.asyncio
async def test_request_without_meta_key_is_untouched():
    seen = {}
    payload = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {"cursor": None}}

    async def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 3, "result": {}})

    async with _client_for(handler) as client:
        await _post(client, payload)
    assert json.loads(seen["body"]) == payload


@pytest.mark.asyncio
async def test_json_rpc_result_shape_is_untouched():
    seen = {}
    payload = {"jsonrpc": "2.0", "id": 4, "result": {"_meta": {}}}  # a response echo, not a request

    async def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 4, "result": {}})

    async with _client_for(handler) as client:
        await _post(client, payload)
    assert json.loads(seen["body"]) == payload  # no "method": repair only targets requests


@pytest.mark.asyncio
async def test_batch_repairs_each_request_item():
    seen = {}
    payload = [{"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {"_meta": {}}},
               {"jsonrpc": "2.0", "id": 6, "method": "tools/list", "params": {}},
               {"jsonrpc": "2.0", "id": 7, "result": {"_meta": None}}]

    async def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, json=[])

    async with _client_for(handler) as client:
        await _post(client, payload)
    sent = json.loads(seen["body"])
    assert "_meta" not in sent[0]["params"]  # empty form stripped per item
    assert sent[1] == payload[1]             # metadata-free item unchanged
    assert sent[2] == payload[2]             # result-shaped item unchanged


@pytest.mark.asyncio
async def test_non_json_content_type_is_untouched():
    seen = {}
    body = b"not json at \xd5 all"

    async def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, content=b"ok")

    async with _client_for(handler) as client:
        await _post(client, body, content_type="application/octet-stream")
    assert seen["body"] == body


@pytest.mark.asyncio
async def test_unparseable_json_body_still_sends():
    seen = {}

    async def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 8, "result": {}})

    async with _client_for(handler) as client:
        resp = await _post(client, b'{"jsonrpc": "2.0", "id": 8, "method":')  # truncated JSON
        assert resp.status_code == 200  # best-effort repair never breaks the send path
    assert seen["body"] == b'{"jsonrpc": "2.0", "id": 8, "method":'


@pytest.mark.asyncio
async def test_wrap_mcp_transport_strips_on_the_request_path():
    seen = {}

    async def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 10, "result": {}})

    async with _client_for(handler, wrapper=_wrap_mcp_transport) as client:
        resp = await _post(client, {"jsonrpc": "2.0", "id": 10, "method": "initialize",
                                    "params": {"_meta": {}, "capabilities": {}}})
        assert resp.status_code == 200
    assert "_meta" not in json.loads(seen["body"])["params"]


@pytest.mark.asyncio
async def test_composed_wrapper_strips_requests_and_caps_responses():
    # ``_wrap_mcp_transport`` = body cap over the meta strip (cap outer, strip inner). Recreate that
    # exact composition with a small cap: the request direction loses its empty ``_meta`` while the
    # response direction still trips the wire-body cap.
    seen = {}

    async def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, content=b"x" * 64)

    inner = httpx.MockTransport(handler)
    capped = _make_mcp_body_cap_transport(
        httpx, _make_empty_meta_stripping_transport(httpx, inner), limit=32)
    async with httpx.AsyncClient(transport=capped) as client:
        with pytest.raises(httpx.ReadError, match=r"bytes cap"):
            await _post(client, {"jsonrpc": "2.0", "id": 9, "method": "ping", "params": {"_meta": {}}})
    assert "_meta" not in json.loads(seen["body"])["params"]
