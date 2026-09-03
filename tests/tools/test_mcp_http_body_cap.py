"""Wire-body cap for MCP HTTP/SSE transports (port of openclaw/openclaw#123194).

Exercises ``_make_mcp_body_cap_transport`` with a real httpx AsyncClient over
a MockTransport: oversized finite bodies and oversized SSE events fail with a
ReadError naming the byte cap; bodies/events under the cap pass; long-lived
SSE connections reset accounting at event boundaries so cumulative keepalive
traffic is unlimited.
"""

import httpx
import pytest

from tools.mcp_tool import _make_mcp_body_cap_transport, _MCP_HTTP_MAX_BODY_BYTES

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


def test_default_cap_is_10mib():
    assert _MCP_HTTP_MAX_BODY_BYTES == 10 * 1024 * 1024
