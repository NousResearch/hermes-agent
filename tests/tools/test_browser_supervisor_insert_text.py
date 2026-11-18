"""``CDPSupervisor.insert_text`` types only into the page session the caller validated.

A real ``CDPSupervisor`` talks to a local WebSocket server that speaks just enough CDP: one http
page target, and a fresh session id per ``Target.attachToTarget`` (as Chrome hands out). The test
records which session every ``Input.insertText`` frame was sent on.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from tools.browser_supervisor import _SupervisorRegistry


class _FakeCDP:
    def __init__(self) -> None:
        self.inserts: list[tuple[str, str | None]] = []  # (text, sessionId)
        self.attaches = 0
        self.url = ""
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._serve, args=(ready,), daemon=True).start()
        assert ready.wait(5)

    def _serve(self, ready: threading.Event) -> None:
        from websockets.asyncio.server import serve

        async def main():
            async with serve(self._handle, "127.0.0.1", 0) as server:
                self.url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}/devtools/browser/x"
                ready.set()
                await asyncio.Event().wait()

        self.loop.run_until_complete(main())

    async def _handle(self, ws) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            method = msg["method"]
            result: dict = {}
            if method == "Target.getTargets":
                result = {"targetInfos": [{"targetId": "T1", "type": "page", "url": "https://example.com/login"}]}
            elif method == "Target.attachToTarget":
                self.attaches += 1
                result = {"sessionId": f"S{self.attaches}"}
            elif method == "Input.insertText":
                self.inserts.append((msg["params"]["text"], msg.get("sessionId")))
            await ws.send(json.dumps({"id": msg["id"], "result": result}))


@pytest.fixture
def supervisor():
    cdp = _FakeCDP()
    reg = _SupervisorRegistry()
    sup = reg.get_or_start("t", cdp.url, start_timeout=5)
    try:
        yield sup, cdp
    finally:
        reg.stop_all()


def test_insert_goes_to_the_pinned_session(supervisor):
    sup, cdp = supervisor
    pinned = sup.current_page_session()
    assert sup.insert_text("abc", session_id=pinned) == {"ok": True}
    assert cdp.inserts == [("abc", pinned)]


def test_insert_refuses_once_the_supervisor_moved_to_another_session(supervisor):
    """Between the caller's focus check and its keystroke, another tool can retarget the supervisor
    (focus_page attaches a new session). The keystroke must not follow it to the new page."""
    sup, cdp = supervisor
    pinned = sup.current_page_session()
    assert sup.focus_page("https://example.com")["ok"] is True
    assert sup.current_page_session() != pinned

    out = sup.insert_text("s3cret", session_id=pinned)
    assert out["ok"] is False
    assert cdp.inserts == []
