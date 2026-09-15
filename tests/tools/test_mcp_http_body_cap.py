"""Wire-body cap for MCP HTTP/SSE transports.

Exercises ``_make_mcp_body_cap_transport`` with a real httpx AsyncClient over
a MockTransport: oversized finite bodies and oversized SSE events fail with a
ReadError naming the byte cap; bodies/events under the cap pass; long-lived
SSE connections reset accounting at event boundaries so cumulative keepalive
traffic is unlimited.
"""

import httpx
import httpx2
import pytest

from tools.mcp_tool_errors import _MCP_HTTP_MAX_BODY_BYTES, _make_mcp_body_cap_transport

LIMIT = 1024  # small cap for tests


def _client_for(handler, limit=LIMIT):
    inner = httpx.MockTransport(handler)
    capped = _make_mcp_body_cap_transport(httpx, inner, limit=limit)
    return httpx.AsyncClient(transport=capped)


@pytest.mark.asyncio
async def test_small_json_body_passes():
    async def handler(request):
        return httpx.Response(200, json={"ok": True})
    async with _client_for(handler) as client:
        resp = await client.get("http://mcp.test/rpc")
        assert resp.json() == {"ok": True}


@pytest.mark.asyncio
async def test_oversized_body_rejected_via_content_length():
    body = b"x" * (LIMIT + 1)
    async def handler(request):
        return httpx.Response(200, content=body)
    async with _client_for(handler) as client:
        with pytest.raises(httpx.ReadError, match=r"Content-Length"):
            await client.get("http://mcp.test/rpc")


@pytest.mark.asyncio
async def test_oversized_streamed_body_rejected_without_content_length():
    # A streaming body with no Content-Length must still trip the cap.
    async def gen():
        for _ in range(8):
            yield b"y" * (LIMIT // 4)

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            async for c in gen():
                yield c

    async def handler(request):
        return httpx.Response(200, stream=_Stream())
    async with _client_for(handler) as client:
        with pytest.raises(httpx.ReadError, match=r"HTTP response exceeds"):
            await client.get("http://mcp.test/rpc")


@pytest.mark.asyncio
async def test_sse_event_over_cap_rejected():
    async def gen():
        yield b"data: " + b"z" * (LIMIT + 64)  # one giant unterminated event

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            async for c in gen():
                yield c

    async def handler(request):
        return httpx.Response(
            200, stream=_Stream(),
            headers={"content-type": "text/event-stream"},
        )
    async with _client_for(handler) as client:
        with pytest.raises(httpx.ReadError, match=r"SSE event exceeds"):
            async with client.stream("GET", "http://mcp.test/sse") as resp:
                async for _ in resp.aiter_bytes():
                    pass


@pytest.mark.asyncio
async def test_sse_cumulative_keepalives_unlimited():
    # Many small completed events whose TOTAL far exceeds the cap must all
    # pass: accounting resets at every completed event boundary.
    async def gen():
        for i in range(64):
            yield b": keepalive %d\n\n" % i + b"data: {\"n\": %d}\n\n" % i

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            async for c in gen():
                yield c

    async def handler(request):
        return httpx.Response(
            200, stream=_Stream(),
            headers={"content-type": "text/event-stream"},
        )
    total = 0
    async with _client_for(handler, limit=64) as client:
        async with client.stream("GET", "http://mcp.test/sse") as resp:
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
    assert total > 64  # cumulative traffic exceeded the per-event cap


@pytest.mark.asyncio
async def test_sse_event_split_across_chunks_counts_prefix():
    # An event streamed in pieces (no boundary) accumulates until it
    # crosses the cap.
    async def gen():
        for _ in range(6):
            yield b"data: " + b"q" * (LIMIT // 4)

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            async for c in gen():
                yield c

    async def handler(request):
        return httpx.Response(
            200, stream=_Stream(),
            headers={"content-type": "text/event-stream"},
        )
    async with _client_for(handler) as client:
        with pytest.raises(httpx.ReadError, match=r"SSE event exceeds"):
            async with client.stream("GET", "http://mcp.test/sse") as resp:
                async for _ in resp.aiter_bytes():
                    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("httpx_mod", [httpx, httpx2], ids=["httpx", "httpx2"])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize("chunk_size", [None, 1, 11], ids=["coalesced", "bytes", "split"])
@pytest.mark.parametrize("declared_length", [False, True], ids=["streamed", "content-length"])
async def test_sse_budget_is_independent_of_wire_chunks(httpx_mod, newline, chunk_size, declared_length):
    # Each event is at or below 64 bytes, but arbitrary network chunking and the stream's
    # total length must not change the per-event verdict. Include empty lines and
    # multi-line events, with every SSE line-ending form accepted by the protocol.
    event = b"data: one" + newline + b"data: " + b"x" * (64 - 15 - 3 * len(newline)) + newline + newline
    assert len(event) == 64
    mixed_event = b"data: one\rdata: two\ndata: three\r\n\r\n"
    payload = (newline + b": keepalive" + newline + newline + event + mixed_event) * 64
    chunks = [payload] if chunk_size is None else [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)]

    class Stream(httpx_mod.AsyncByteStream):
        async def __aiter__(self):
            for chunk in chunks:
                yield chunk

    async def handler(request):
        headers = {"content-type": "Text/Event-Stream; charset=utf-8"}
        if declared_length:
            headers["content-length"] = str(len(payload))
        return httpx_mod.Response(200, stream=Stream(), headers=headers)

    transport = _make_mcp_body_cap_transport(httpx_mod, httpx_mod.MockTransport(handler), limit=64)
    async with httpx_mod.AsyncClient(transport=transport) as client:
        response = await client.get("http://mcp.test/sse")
        assert response.content == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("httpx_mod", [httpx, httpx2], ids=["httpx", "httpx2"])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize("chunk_size", [None, 1, 11], ids=["coalesced", "bytes", "split"])
async def test_sse_budget_rejects_one_oversized_event_after_small_events(httpx_mod, newline, chunk_size):
    payload = (b": ok" + newline + newline) * 32
    payload += b"data: " + b"x" * 65 + newline + newline
    chunks = [payload] if chunk_size is None else [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)]

    class Stream(httpx_mod.AsyncByteStream):
        async def __aiter__(self):
            for chunk in chunks:
                yield chunk

    async def handler(request):
        return httpx_mod.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"})

    transport = _make_mcp_body_cap_transport(httpx_mod, httpx_mod.MockTransport(handler), limit=64)
    async with httpx_mod.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx_mod.ReadError, match="SSE event exceeds 64 bytes"):
            await client.get("http://mcp.test/sse")


@pytest.mark.asyncio
@pytest.mark.parametrize("httpx_mod", [httpx, httpx2], ids=["httpx", "httpx2"])
async def test_non_sse_content_type_does_not_reset_body_budget(httpx_mod):
    class Stream(httpx_mod.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(32):
                yield b"small\n\n"

    async def handler(request):
        return httpx_mod.Response(
            200, stream=Stream(),
            headers={"content-type": "application/x-text/event-stream"},
        )

    transport = _make_mcp_body_cap_transport(httpx_mod, httpx_mod.MockTransport(handler), limit=64)
    async with httpx_mod.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx_mod.ReadError, match="HTTP response exceeds 64 bytes"):
            await client.get("http://mcp.test/rpc")
