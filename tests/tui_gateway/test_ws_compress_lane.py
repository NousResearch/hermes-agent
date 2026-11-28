"""Compress lanes keep other sessions responsive, ordered and bounded.

The real ``handle_ws`` with an event-gated authority stand-in.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tests.waiting import async_wait_until
from tui_gateway import ws as ws_mod


class Socket:
    def __init__(self):
        self.inbound = asyncio.Queue()
        self.replies = {}
        self.order = []
        self.arrived = asyncio.Event()
        self.reader_cancelled = asyncio.Event()
        self.received = 0
        self.fail_ids = set()

    async def accept(self):
        pass

    async def send_text(self, line):
        frame = json.loads(line)
        if frame.get("id") in self.fail_ids:
            raise RuntimeError("peer gone")
        if "id" in frame:
            self.replies[frame["id"]] = frame
            self.order.append(frame["id"])
            self.arrived.set()

    async def receive_text(self):
        try:
            frame = await self.inbound.get()
        except asyncio.CancelledError:
            self.reader_cancelled.set()
            raise
        if frame is None:
            raise ws_mod._WebSocketDisconnect()
        self.received += 1
        return json.dumps(frame)

    async def close(self):
        pass

    def send(self, request_id, method="prompt.submit", session_id="A", **params):
        if session_id is not None:
            params["session_id"] = session_id
        self.inbound.put_nowait(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )

    async def reply(self, request_id):
        async def arrived():
            while request_id not in self.replies:
                self.arrived.clear()
                await self.arrived.wait()
            return self.replies[request_id]

        return await asyncio.wait_for(arrived(), 5)


class Authority:
    def __init__(self):
        self.actor = SimpleNamespace(profile_id="test-profile")
        self.gates = {}
        self.started = {}
        self.log = []
        self.replyless = set()

    def awaits_profile_start(self, req, held):
        return None

    async def dispatch(self, req):
        request_id = req["id"]
        self.log.append(f"start:{request_id}")
        self.started.setdefault(request_id, asyncio.Event()).set()
        try:
            if request_id in self.gates:
                await self.gates[request_id].wait()
        except asyncio.CancelledError:
            self.log.append(f"cancelled:{request_id}")
            raise
        self.log.append(f"done:{request_id}")
        if request_id in self.replyless:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}

    async def close(self):
        self.log.append("closed")

    def block(self, request_id):
        self.gates[request_id] = asyncio.Event()
        self.started[request_id] = asyncio.Event()
        return self.gates[request_id]

    async def wait_started(self, request_id):
        await asyncio.wait_for(self.started[request_id].wait(), 5)


@pytest_asyncio.fixture
async def rig(monkeypatch):
    sock, authority = Socket(), Authority()
    monkeypatch.setattr(ws_mod, "_WS_DISPATCH_QUEUE_MAX", 4)
    monkeypatch.setattr(ws_mod, "_authority_connection", lambda *args: authority)
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    try:
        yield sock, authority, task
    finally:
        for gate in authority.gates.values():
            gate.set()
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)


@pytest.mark.asyncio
async def test_compress_does_not_block_other_sessions_or_sessionless_frames(rig):
    sock, authority, _ = rig
    release = authority.block("compress")
    sock.send("compress", "session.mutate", operation="compress")
    await authority.wait_started("compress")
    sock.send("other", session_id="B")
    sock.send("sessionless", "session.create", session_id=None)
    sock.send("read", "session.transcript")
    await sock.reply("other")
    await sock.reply("sessionless")
    await sock.reply("read")
    assert "compress" not in sock.replies
    release.set()
    await sock.reply("compress")


@pytest.mark.asyncio
async def test_same_session_frames_and_second_compress_keep_fifo(rig):
    sock, authority, _ = rig
    release = authority.block("compress")
    sock.send("compress", "session.mutate", operation="compress")
    await authority.wait_started("compress")
    sock.send("after", "session.interrupt")
    sock.send("again", "session.mutate", operation="compress")
    sock.send("barrier", session_id="B")
    await sock.reply("barrier")
    assert authority.log == ["start:compress", "start:barrier", "done:barrier"]
    release.set()
    await sock.reply("again")
    assert sock.order == ["barrier", "compress", "after", "again"]


@pytest.mark.asyncio
async def test_compress_cap_parks_only_the_third_session(rig):
    sock, authority, _ = rig
    release = authority.block("compress-a")
    authority.block("compress-c")
    authority.block("compress-d")
    sock.send("compress-a", "session.mutate", operation="compress")
    await authority.wait_started("compress-a")
    sock.send("compress-c", "session.mutate", session_id="C", operation="compress")
    await authority.wait_started("compress-c")
    sock.send("compress-d", "session.mutate", session_id="D", operation="compress")
    sock.send("other", session_id="B")
    await sock.reply("other")
    assert "start:compress-d" not in authority.log
    release.set()
    await authority.wait_started("compress-d")
    assert "done:compress-a" in authority.log
    assert "done:compress-c" not in authority.log


@pytest.mark.asyncio
async def test_server_shutdown_cancels_lane_and_drops_queued_frames(rig):
    sock, authority, task = rig
    authority.block("compress")
    sock.send("compress", "session.mutate", operation="compress")
    await authority.wait_started("compress")
    sock.send("queued")
    sock.send("barrier", session_id="B")
    await sock.reply("barrier")
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
    assert "cancelled:compress" in authority.log
    assert "start:queued" not in authority.log
    assert authority.log[-1] == "closed"


@pytest.mark.asyncio
async def test_compress_bounds_reader_and_answers_every_frame_in_order(rig):
    sock, authority, _ = rig
    release = authority.block("compress")
    sock.send("compress", "session.mutate", operation="compress")
    await authority.wait_started("compress")
    sock.send("barrier", session_id="B")
    await sock.reply("barrier")
    before = sock.received
    queued = [f"queued-{n}" for n in range(32)]
    for request_id in queued:
        sock.send(request_id)
    previous = before

    def reader_parked():
        nonlocal previous
        current = sock.received
        stopped = current > before and current == previous
        previous = current
        return stopped

    await async_wait_until(reader_parked, timeout=5, what="reader backpressure")
    assert sock.received - before < len(queued)
    release.set()
    await sock.reply(queued[-1])
    assert sock.order == ["barrier", "compress", *queued]


@pytest.mark.asyncio
async def test_failed_lane_reply_stops_followers_without_cancelling_running_work(rig):
    sock, authority, task = rig
    release = authority.block("compress")
    sibling = authority.block("sibling")
    control = authority.block("control")
    # A deferred handler sends no inline reply, so only the send_failed check stops its lane.
    authority.replyless.add("sibling")
    sock.fail_ids.add("compress")
    sock.send("compress", "session.mutate", operation="compress")
    await authority.wait_started("compress")
    sock.send("sibling", "session.mutate", session_id="C", operation="compress")
    await authority.wait_started("sibling")
    sock.send("queued", "session.mutate", session_id="C", operation="compress")
    sock.send("control", session_id="B")
    await authority.wait_started("control")
    release.set()
    await asyncio.wait_for(sock.reader_cancelled.wait(), 5)
    assert not any(entry.startswith("cancelled:") for entry in authority.log)
    assert "closed" not in authority.log
    sibling.set()
    control.set()
    await asyncio.wait_for(task, 5)
    assert "done:sibling" in authority.log and "done:control" in authority.log
    assert "start:queued" not in authority.log
    assert authority.log[-1] == "closed"
