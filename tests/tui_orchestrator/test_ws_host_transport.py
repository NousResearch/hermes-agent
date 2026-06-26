"""Prove ws_host stands up a REAL WebSocket server, gates on the credential,
and routes accepted sockets into handle_ws. We stub handle_ws (upstream,
already production-tested for web/desktop) to isolate OUR host + auth layer.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time

import uvicorn
import websockets


def main() -> int:
    CRED = "test-credential-xyz"
    routed = {"count": 0, "peers": []}

    # Stub the upstream session handler so we test OUR host + gate, not dispatch.
    import tui_gateway.ws as ws_mod

    async def fake_handle_ws(ws):
        routed["count"] += 1
        await ws.accept()
        await ws.send_text("routed-ok")
        await ws.close()

    ws_mod.handle_ws = fake_handle_ws

    import tui_gateway.ws_host as ws_host
    app = ws_host.build_app(CRED)

    config = uvicorn.Config(app, host="127.0.0.1", port=8917, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # wait for startup
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "uvicorn host did not start"

    async def run_clients():
        # 1) WRONG credential → must be rejected (close 4401), handle_ws NOT called.
        rejected = False
        try:
            async with websockets.connect("ws://127.0.0.1:8917/api/ws?internal=WRONG") as c:
                await asyncio.wait_for(c.recv(), timeout=2)
        except Exception:
            rejected = True
        assert rejected, "wrong credential was NOT rejected"
        assert routed["count"] == 0, "handle_ws ran despite bad credential!"
        print("PASS: wrong credential rejected, handle_ws not invoked")

        # 2) CORRECT credential → routed into handle_ws, gets the reply.
        async with websockets.connect(f"ws://127.0.0.1:8917/api/ws?internal={CRED}") as c:
            msg = await asyncio.wait_for(c.recv(), timeout=2)
            assert msg == "routed-ok", f"unexpected reply: {msg!r}"
        assert routed["count"] == 1, f"handle_ws not invoked once: {routed['count']}"
        print("PASS: correct credential routed into handle_ws, reply received")

        # 3) RECONNECT with same multi-use credential → routed again (the
        #    'renderer dies and reconnects' case; credential is multi-use).
        async with websockets.connect(f"ws://127.0.0.1:8917/api/ws?internal={CRED}") as c:
            msg = await asyncio.wait_for(c.recv(), timeout=2)
            assert msg == "routed-ok"
        assert routed["count"] == 2, "multi-use credential failed on reconnect"
        print("PASS: same credential reused on reconnect (renderer re-attach works)")

    asyncio.run(run_clients())
    server.should_exit = True
    t.join(timeout=5)
    print("\nALL WS_HOST TRANSPORT/AUTH TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
