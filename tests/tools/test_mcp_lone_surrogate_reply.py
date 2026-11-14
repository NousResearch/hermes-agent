"""A server reply carrying a lone UTF-16 surrogate escape (a description cut inside an emoji)
must still answer its request. pydantic's JSON parser rejects it, the SDK drops the message,
and before the fix the request waited for its full timeout (discovery stalled, calls hung)."""

import asyncio
import sys
import textwrap

import pytest

pytest.importorskip("mcp")

_SERVER = textwrap.dedent(r'''
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        mid, meth = m.get("id"), m.get("method")
        if mid is None:
            continue
        if meth == "initialize":
            out = json.dumps({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": m["params"]["protocolVersion"], "capabilities": {"tools": {}},
                "serverInfo": {"name": "s", "version": "1"}}})
        elif meth == "tools/list":
            out = '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"t","description":"cut \\ud83d","inputSchema":{"type":"object"}}]}}' % json.dumps(mid)
        elif meth == "tools/call":
            out = '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"hi \\ud83d"}]}}' % json.dumps(mid)
        else:
            out = json.dumps({"jsonrpc": "2.0", "id": mid, "result": {}})
        sys.stdout.write(out + "\n")
        sys.stdout.flush()
''')


def test_lone_surrogate_reply_answers_instead_of_timing_out(tmp_path):
    from tools import mcp_tool
    assert mcp_tool._ensure_mcp_sdk()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = tmp_path / "server.py"
    server.write_text(_SERVER, encoding="utf-8")

    async def run():
        params = StdioServerParameters(command=sys.executable, args=[str(server)])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), 10)
                tools = await asyncio.wait_for(session.list_tools(), 5)
                result = await asyncio.wait_for(session.call_tool("t", {}), 5)
                return tools.tools[0].description, result.content[0].text

    description, text = asyncio.run(run())
    assert (description, text) == ("cut \ufffd", "hi \ufffd")


def test_reply_invalid_on_both_parsers_still_raises():
    from tools import mcp_tool
    assert mcp_tool._ensure_mcp_sdk()
    import mcp_types

    with pytest.raises(ValueError):
        mcp_types.jsonrpc_message_adapter.validate_json('{"jsonrpc":"2.0","id":1,"result":')
