"""``connect_owner`` reads a ``session.resume`` answer over 1 MiB (#145215)."""

import asyncio
import json

from tui_gateway import owner_forward


def test_connect_owner_reads_a_resume_snapshot_larger_than_one_mib(monkeypatch, tmp_path):
    from websockets.asyncio.server import serve

    from hermes_cli import gateway_client

    history = [{"role": "user", "content": "x" * 1024} for _ in range(2048)]  # about 2 MiB

    async def scenario():
        async def handler(ws):
            async for raw in ws:
                request = json.loads(raw)
                result = {"session_id": "s", "messages": history}
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}))

        async with serve(handler, "127.0.0.1", 0, subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL],
                         max_size=None) as listener:
            port = listener.sockets[0].getsockname()[1]
            monkeypatch.setattr(gateway_client, "_session_ticket", lambda _home, _endpoint, **_kw: "ticket")
            monkeypatch.setattr(gateway_client, "gateway_ws_target", lambda _endpoint, _ticket: (
                f"ws://127.0.0.1:{port}", [gateway_client.GATEWAY_WS_PROTOCOL]))
            async with owner_forward.connect_owner(tmp_path, object()) as client:
                return await client.rpc("session.resume", session_id="s")

    attached = asyncio.run(scenario())
    assert len(attached["messages"]) == len(history)
