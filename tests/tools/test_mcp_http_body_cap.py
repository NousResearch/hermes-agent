"""Wire-body cap for MCP HTTP/SSE transports.

Exercises ``_make_mcp_body_cap_transport`` with a real httpx AsyncClient over
a MockTransport: oversized finite bodies and oversized SSE events fail with a
ReadError naming the byte cap; bodies/events under the cap pass; long-lived
SSE connections reset accounting at event boundaries so cumulative keepalive
traffic is unlimited.
"""

import httpx
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


def _sse_client(chunks, limit=64):
    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for c in chunks:
                yield c

    async def handler(request):
        return httpx.Response(
            200, stream=_Stream(),
            headers={"content-type": "text/event-stream"},
        )
    return _client_for(handler, limit=limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("sep,cut", [
    (b"\n\n", 1),            # ...\n | \n...
    (b"\r\n\r\n", 1),        # ...\r | \n\r\n...
    (b"\r\n\r\n", 2),        # ...\r\n | \r\n...
    (b"\r\n\r\n", 3),        # ...\r\n\r | \n...
    # The spec also allows CR terminators and mixed styles; every pairing is a boundary.
    (b"\r\r", 1),            # ...\r | \r...
    (b"\n\r", 1),            # ...\n | \r...
    (b"\n\r\n", 1),          # ...\n | \r\n...
    (b"\n\r\n", 2),          # ...\n\r | \n...
    (b"\r\n\n", 1),          # ...\r\n | \n...
    (b"\r\n\n", 2),          # ...\r\n\n | ...
    (b"\r\n\r", 1),          # ...\r\n | \r...
    (b"\r\n\r", 2),          # ...\r\n\r | ...
    (b"\r\r\n", 1),          # ...\r | \r\n...
    (b"\r\r\n", 2),          # ...\r\r | \n...
])
async def test_sse_boundary_split_across_chunks_still_resets(sep, cut):
    # A completed event boundary straddling two chunks must still reset the
    # per-event counter; otherwise the finished event's bytes are charged to
    # the next event and the cap trips early.
    event1 = b"data: " + b"a" * 40 + sep[:cut]
    event2 = sep[cut:] + b"data: " + b"b" * 40 + sep
    chunks = [event1, event2]
    async with _sse_client(chunks) as client:
        async with client.stream("GET", "http://mcp.test/sse") as resp:
            async for _ in resp.aiter_bytes():
                pass


@pytest.mark.asyncio
async def test_sse_crlf_inside_event_is_not_a_boundary():
    # A lone CRLF ends a line, not the event. If \r\n could split the count, a
    # multi-line event over the cap would evade it.
    chunks = [b"data: " + b"a" * 40 + b"\r\ndata: " + b"b" * 40 + b"\r\n\r\n"]
    async with _sse_client(chunks) as client:
        with pytest.raises(httpx.ReadError, match=r"SSE event exceeds"):
            async with client.stream("GET", "http://mcp.test/sse") as resp:
                async for _ in resp.aiter_bytes():
                    pass


@pytest.mark.asyncio
async def test_sse_crlf_split_across_chunks_inside_event_is_not_a_boundary():
    # Same check with the \r and \n straddling the chunk seam: the carry must not
    # turn a mid-event line ending into a boundary.
    chunks = [b"data: " + b"a" * 40 + b"\r", b"\ndata: " + b"b" * 40 + b"\r\n\r\n"]
    async with _sse_client(chunks) as client:
        with pytest.raises(httpx.ReadError, match=r"SSE event exceeds"):
            async with client.stream("GET", "http://mcp.test/sse") as resp:
                async for _ in resp.aiter_bytes():
                    pass


@pytest.mark.asyncio
async def test_sse_split_boundary_over_cap_event_still_rejected():
    # The carry must not weaken the cap: an event genuinely over the limit is
    # still rejected even when its preceding boundary straddled chunks.
    chunks = [b"data: ok\n", b"\ndata: " + b"z" * 80 + b"\n\n"]
    async with _sse_client(chunks) as client:
        with pytest.raises(httpx.ReadError, match=r"SSE event exceeds"):
            async with client.stream("GET", "http://mcp.test/sse") as resp:
                async for _ in resp.aiter_bytes():
                    pass


@pytest.mark.asyncio
async def test_sse_boundary_split_three_ways():
    # A \r\n\r\n boundary dribbled across three chunks still resets the count.
    chunks = [b"data: " + b"a" * 40 + b"\r", b"\n\r", b"\ndata: " + b"b" * 40 + b"\r\n\r\n"]
    async with _sse_client(chunks) as client:
        async with client.stream("GET", "http://mcp.test/sse") as resp:
            async for _ in resp.aiter_bytes():
                pass
