import asyncio
import importlib
import io
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


class Dumpable:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, **_kwargs):
        return self.payload


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def bridge(monkeypatch):
    package = ModuleType("arxiv_mcp_server")
    server = ModuleType("arxiv_mcp_server.server")

    async def list_tools():
        return [Dumpable({"name": "search_papers"})]

    async def list_prompts():
        return [Dumpable({"name": "summarize_paper"})]

    async def call_tool(name, arguments):
        return [Dumpable({"type": "text", "text": f"{name}:{arguments.get('query', '')}"})]

    async def get_prompt(name, arguments):
        return {
            "description": f"prompt:{name}",
            "messages": [{"role": "user", "content": arguments or {}}],
        }

    server.call_tool = call_tool
    server.get_prompt = get_prompt
    server.list_prompts = list_prompts
    server.list_tools = list_tools
    server.settings = SimpleNamespace(APP_NAME="fake-arxiv", APP_VERSION="9.9.9")

    monkeypatch.setitem(sys.modules, "arxiv_mcp_server", package)
    monkeypatch.setitem(sys.modules, "arxiv_mcp_server.server", server)
    monkeypatch.delitem(sys.modules, "tools.arxiv_mcp_stdio_bridge", raising=False)

    module = importlib.import_module("tools.arxiv_mcp_stdio_bridge")
    yield module
    sys.modules.pop("tools.arxiv_mcp_stdio_bridge", None)


def test_initialize_and_runtime_lists_return_json_rpc_payloads(bridge):
    init = _run(
        bridge._handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        )
    )
    tools = _run(bridge._handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
    prompts = _run(
        bridge._handle_request({"jsonrpc": "2.0", "id": 3, "method": "prompts/list"})
    )

    assert init["result"]["protocolVersion"] == "2024-11-05"
    assert init["result"]["capabilities"]["tools"] == {"listChanged": False}
    assert init["result"]["capabilities"]["prompts"] == {"listChanged": False}
    assert init["result"]["serverInfo"]["name"] == "fake-arxiv"
    assert tools["result"]["tools"] == [{"name": "search_papers"}]
    assert prompts["result"]["prompts"] == [{"name": "summarize_paper"}]


def test_tools_call_and_prompts_get_delegate_to_package_handlers(bridge):
    tool = _run(
        bridge._handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": "mcp"}},
            }
        )
    )
    prompt = _run(
        bridge._handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "prompts/get",
                "params": {"name": "summary", "arguments": {"paper": "1234.5678"}},
            }
        )
    )

    assert tool["result"]["isError"] is False
    assert tool["result"]["content"] == [{"type": "text", "text": "search:mcp"}]
    assert prompt["result"]["description"] == "prompt:summary"
    assert prompt["result"]["messages"][0]["content"] == {"paper": "1234.5678"}


def test_resources_list_is_explicitly_unsupported(bridge):
    response = _run(
        bridge._handle_request({"jsonrpc": "2.0", "id": 6, "method": "resources/list"})
    )

    assert response["error"]["code"] == -32601
    assert response["error"]["message"] == "Method not found: resources/list"


def test_initialized_notification_is_silent(bridge):
    assert (
        _run(
            bridge._handle_request(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
        )
        is None
    )


def test_main_redacts_secret_shaped_exception_messages(bridge, monkeypatch, capsys):
    secret_fixture = "sk-" + "testsecret" + "1234567890"

    async def boom(_payload):
        raise RuntimeError("upstream token: " + secret_fixture)

    monkeypatch.setattr(bridge, "_handle_request", boom)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO('{"jsonrpc":"2.0","id":7,"method":"tools/list"}\n'),
    )

    assert bridge.main() == 0

    output = capsys.readouterr().out
    response = json.loads(output)
    assert response["error"]["code"] == -32000
    assert secret_fixture not in output
    message = response["error"]["message"]
    assert "token" in message
    assert "[REDACTED]" in message
