"""``get_prompt`` through a real MCP SDK client and server: argument values the model sends as
ints, bools and nulls still render the prompt (prompt arguments are strings on the wire)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_get_prompt_non_string_arguments_reach_a_real_server():
    """Prompt arguments are strings on the wire; the model sends ints, bools and nulls.
    Through a real SDK client and server, those still render the prompt."""
    from mcp.client import Client
    from mcp.server import MCPServer
    from tools.mcp_tool_handlers import _make_get_prompt_handler

    sdk = MCPServer("prompts")

    @sdk.prompt()
    def report(days: str, verbose: str, style: str = "plain") -> str:
        return f"days={days} verbose={verbose} style={style}"

    loop = asyncio.get_running_loop()
    async with Client(sdk) as client:
        server = SimpleNamespace(session=client.session, _rpc_lock=asyncio.Lock())
        with patch("tools.mcp_tool_discovery._get_connected_server_for_call", return_value=server), \
             patch("tools.mcp_tool_loop._run_on_mcp_loop",
                   side_effect=lambda call, timeout=30: asyncio.run_coroutine_threadsafe(call(), loop).result(timeout)):
            handler = _make_get_prompt_handler("prompts", 30)
            result = json.loads(await asyncio.to_thread(
                handler, {"name": "report", "arguments": {"days": 7, "verbose": True, "style": None}}))
    assert result == {"messages": [{"role": "user", "content": "days=7 verbose=true style=plain"}]}
