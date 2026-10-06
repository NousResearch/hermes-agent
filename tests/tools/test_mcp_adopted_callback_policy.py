"""An adopted remote definition's sampling/elicitation policy governs the rebuilt SDK session.

Security-audit follow-up for #74809 (runtime-confirmed lead): when ``_refresh_remote_config``
adopted a changed endpoint whose definition set ``sampling.enabled: false``, the rebuilt
``ClientSession`` still received the handler built for the replaced definition, advertised
``sampling`` in its second ``initialize`` and served the new peer's ``sampling/createMessage``
through Hermes' LLM. Real SDK session over in-memory streams; only the LLM call is a counter.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

from tools import mcp_tool, mcp_tool_config


@pytest.mark.parametrize("callback", ["sampling", "elicitation"])
def test_adopted_definition_disables_callbacks_on_the_rebuilt_session(
        callback, tmp_path, monkeypatch):
    import anyio
    import mcp.types as types
    from mcp.shared.message import SessionMessage
    from pydantic import TypeAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert mcp_tool._ensure_mcp_sdk()
    rpc = TypeAdapter(types.JSONRPCMessage)
    raw = {"url": "https://example.invalid/a", "protocol": "legacy", "skip_preflight": True,
           "connect_timeout": 3, "sampling": {"enabled": True}, "elicitation": {"enabled": True}}

    def save():
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"private": raw}}))

    save()
    llm_calls, advertised, answers = [], [], {}
    answered = asyncio.Event()

    def counting_llm(**kwargs):
        llm_calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="dummy-answer", tool_calls=None), finish_reason="stop")],
            model="dummy-model", usage=SimpleNamespace(total_tokens=1))

    monkeypatch.setattr("agent.auxiliary_client.call_llm", counting_llm)

    @asynccontextmanager
    async def transport(self, url, headers, *args):
        to_client, client_read = anyio.create_memory_object_stream(8)
        client_write, from_client = anyio.create_memory_object_stream(8)

        async def send(data):
            await to_client.send(SessionMessage(rpc.validate_python(data)))

        async def peer():
            async for message in from_client:
                data = message.message.model_dump(by_alias=True, exclude_none=True)
                method = data.get("method")
                if method == "initialize":
                    advertised.append((url, sorted(data["params"]["capabilities"])))
                    await send({"jsonrpc": "2.0", "id": data["id"], "result": {
                        "protocolVersion": data["params"]["protocolVersion"],
                        "capabilities": {"tools": {}}, "serverInfo": {"name": "dummy", "version": "1"}}})
                elif method == "tools/list":
                    await send({"jsonrpc": "2.0", "id": data["id"], "result": {"tools": []}})
                    if url.endswith("/b") and callback == "sampling":
                        await send({"jsonrpc": "2.0", "id": "sampling-fixture", "method": "sampling/createMessage",
                                    "params": {"messages": [{"role": "user", "content": {
                                        "type": "text", "text": "Dummy request"}}], "maxTokens": 8}})
                    elif url.endswith("/b"):
                        answered.set()
                elif data.get("id") == "sampling-fixture":
                    answers[url] = data
                    answered.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(peer)
            try:
                yield client_read, client_write
            finally:
                tg.cancel_scope.cancel()

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_streamable_http_transport", transport)

    async def scenario():
        task = mcp_tool.MCPServerTask("private")
        try:
            await task.start(mcp_tool_config._load_mcp_config()["private"])
            first = task.session
            raw["url"] = "https://example.invalid/b"
            raw[callback] = {"enabled": False}
            save()
            task._reconnect_event.set()
            await asyncio.wait_for(answered.wait(), 10)
            assert task.session is not first and task._config["url"].endswith("/b")
        finally:
            await task.shutdown()

    asyncio.run(scenario())

    assert [url for url, _ in advertised] == ["https://example.invalid/a", "https://example.invalid/b"]
    assert callback in advertised[0][1]  # A's definition enables it
    assert callback not in advertised[1][1]  # B's definition disables it on the rebuilt session
    other = "elicitation" if callback == "sampling" else "sampling"
    assert other in advertised[1][1]  # the unchanged callback keeps serving
    if callback == "sampling":
        assert llm_calls == []
        assert "error" in answers["https://example.invalid/b"]
