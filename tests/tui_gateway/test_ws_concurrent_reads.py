"""The WS concurrent-read lane (#134889 GW-18 review): waiting context reads stay bounded, a server
shutdown or client hang-up never waits for reads that have not started, and a reply the read lane
cannot send never cancels the control handler the serial lane is running.

The real ``handle_ws`` dispatcher; only ``_dispatch_request`` is a gated async stand-in.
"""
import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway import ws as ws_mod


class Socket:
    def __init__(self):
        self.inbound = asyncio.Queue()
        self.received = 0
        self.sent = []
        self.fail_ids = set()

    async def accept(self):
        pass

    async def send_text(self, line):
        frame = json.loads(line)
        if frame.get("id") in self.fail_ids:
            raise RuntimeError("peer gone")
        self.sent.append(frame)

    async def receive_text(self):
        item = await self.inbound.get()
        if item is None:
            raise ws_mod._WebSocketDisconnect()
        self.received += 1
        return json.dumps(item)

    async def close(self):
        pass


@pytest.fixture
def rig(monkeypatch):
    gates, log = {}, []

    async def dispatch(authority, req, method, transport):
        log.append(f"start:{req['id']}")
        try:
            gate = gates.get(method)
            if gate is not None:
                await gate.wait()
        except asyncio.CancelledError:
            log.append(f"cancelled:{req['id']}")
            raise
        log.append(f"done:{req['id']}")
        return {"jsonrpc": "2.0", "id": req["id"], "result": {}}

    monkeypatch.setattr(ws_mod, "_dispatch_request", dispatch)
    monkeypatch.setattr(ws_mod, "_start_backend_liveness", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "_ensure_lease_watcher", lambda: None)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (log.append("teardown"), (0, 0))[1])
    return Socket(), gates, log


def rpc(req_id, method="session.context_breakdown"):
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": {}}


async def settle(predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize('read_method', ['session.context_breakdown', 'session.control.read'])
async def test_waiting_context_reads_keep_reader_backpressure(rig, read_method):
    sock, gates, log = rig
    gates[read_method] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    burst = ws_mod._WS_DISPATCH_QUEUE_MAX * 4
    for n in range(burst):
        sock.inbound.put_nowait(rpc(n, read_method))
    try:
        # A bounded reader parks with frames unread; give it every chance to drain them all.
        assert not await settle(lambda: sock.inbound.empty(), timeout=2.0)
        bound = ws_mod._WS_DISPATCH_QUEUE_MAX + ws_mod._WS_CONCURRENT_READ_LIMIT + ws_mod._WS_WAITING_READ_LIMIT + 2
        assert sock.received <= bound, (sock.received, bound)
        assert sum(entry.startswith("start:") for entry in log) == ws_mod._WS_CONCURRENT_READ_LIMIT
    finally:
        release.set()
    assert await settle(lambda: sock.inbound.empty())
    sock.inbound.put_nowait(None)
    await asyncio.wait_for(task, 30)
    assert len(sock.sent) == burst + 1  # gateway.ready + one reply per read


@pytest.mark.asyncio
@pytest.mark.parametrize('read_method', ['session.context_breakdown', 'session.control.read'])
async def test_server_shutdown_does_not_wait_for_a_running_context_read(rig, read_method):
    sock, gates, log = rig
    gates[read_method] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    sock.inbound.put_nowait(rpc(1, read_method))
    try:
        assert await settle(lambda: "start:1" in log)
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done, "socket shutdown still waits for the slow context read"
        assert "cancelled:1" in log
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_hang_up_drops_reads_that_never_started(rig):
    sock, gates, log = rig
    gates["session.context_breakdown"] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    for n in range(6):
        sock.inbound.put_nowait(rpc(n))
    assert await settle(lambda: sock.received == 6 and sum(e.startswith("start:") for e in log) == 2)
    sock.inbound.put_nowait(None)
    assert not await settle(lambda: "teardown" in log, timeout=2.0), "teardown ran under a started read"
    release.set()
    await asyncio.wait_for(task, 10)
    assert sorted(e for e in log if e.startswith("start:")) == ["start:0", "start:1"]
    assert log[-1] == "teardown"


@pytest.mark.asyncio
async def test_a_failed_read_reply_does_not_cancel_the_running_control(rig):
    """The serial lane's handler (a submit, an interrupt, a mutate) finishes; then the link ends."""
    sock, gates, log = rig
    gates["prompt.submit"] = release = asyncio.Event()
    gates["session.context_breakdown"] = read_gate = asyncio.Event()
    sock.fail_ids.add("read")
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    sock.inbound.put_nowait(rpc("read"))
    assert await settle(lambda: "start:read" in log)
    sock.inbound.put_nowait(rpc("submit", "prompt.submit"))
    assert await settle(lambda: "start:submit" in log)
    read_gate.set()  # the read's reply now fails while the submit is mid-await
    assert await settle(lambda: "done:read" in log)
    assert not await settle(lambda: task.done(), timeout=2.0), "the link ended under the running control"
    release.set()
    await asyncio.wait_for(task, 10)
    assert "cancelled:submit" not in log and "done:submit" in log
    assert log[-1] == "teardown"


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['prompt.submit', 'session.control'])
async def test_control_read_does_not_queue_behind_a_blocked_mutation(rig, mutation):
    """A control snapshot must not inherit a slow mutating request's latency.

    The desktop's own request timeout reports whichever request was waiting when it fired, so a
    ``session.control.read`` parked behind ``prompt.submit`` surfaces as "Session controls
    unavailable: request timed out after 30s: session.control.read" — naming the victim while the
    real blocker is never logged anywhere.
    """
    sock, gates, log = rig
    gates[mutation] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))

    sock.inbound.put_nowait(rpc("submit", mutation))

    assert await settle(lambda: "start:submit" in log)

    sock.inbound.put_nowait(rpc("read", "session.control.read"))
    sock.inbound.put_nowait(rpc("later-mutation", "session.control"))

    try:
        read_finished = await settle(lambda: "done:read" in log, timeout=2.0)
        submit_still_blocked = "done:submit" not in log
        later_mutation_still_waiting = "start:later-mutation" not in log
        read_reply_sent = any(frame.get("id") == "read" for frame in sock.sent)
    finally:
        release.set()

    assert await settle(lambda: "done:submit" in log)
    sock.inbound.put_nowait(None)
    await asyncio.wait_for(task, 10)
    assert read_finished and read_reply_sent, log
    assert submit_still_blocked and later_mutation_still_waiting, log
    assert log.index("done:submit") < log.index("start:later-mutation"), log


@pytest.mark.asyncio
async def test_slow_dispatch_warning_names_the_blocking_method(rig, monkeypatch, caplog):
    """The blocker is named in the log, so a client-side timeout is never untraceable."""
    sock, gates, log = rig
    monkeypatch.setattr(ws_mod, "_WS_SLOW_DISPATCH_LOG_MS", 1)
    clock = [0.0]
    monkeypatch.setattr(ws_mod, "time", SimpleNamespace(
        perf_counter=lambda: clock[0], monotonic=ws_mod.time.monotonic, time=ws_mod.time.time))
    gates["prompt.submit"] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))

    with caplog.at_level(logging.WARNING, logger="tui_gateway.ws"):
        sock.inbound.put_nowait(rpc("submit", "prompt.submit"))
        assert await settle(lambda: "start:submit" in log)
        clock[0] = 0.02
        release.set()
        assert await settle(lambda: "done:submit" in log)
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(task, 10)

    warnings = [record.getMessage() for record in caplog.records if "ws dispatch slow" in record.getMessage()]
    assert warnings, [record.getMessage() for record in caplog.records]
    assert "prompt.submit" in warnings[0]


@pytest.fixture
def starting_profile(monkeypatch, tmp_path):
    """The authority's pending-start contract; the transport must wait without spending a read slot."""
    started, observed = asyncio.Event(), []

    class Authority:
        actor = SimpleNamespace(profile_id=str(tmp_path))

        def awaits_profile_start(self, req, held=()):
            if req.get('params', {}).get('profile') == 'beta' and not started.is_set():
                observed.append(req['id'])
                return 'beta', started
            return None

        async def close(self):
            pass

    monkeypatch.setattr(ws_mod, '_authority_connection', lambda *_args: Authority())
    return started, observed


@pytest.mark.asyncio
async def test_reads_wait_for_profile_start_without_occupying_running_slots(rig, starting_profile):
    sock, _, log = rig
    started, observed = starting_profile
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    for rid in ('beta-one', 'beta-two'):
        sock.inbound.put_nowait({**rpc(rid, 'session.control.read'), 'params': {'profile': 'beta'}})
    try:
        assert await settle(lambda: len(observed) == 2)
        sock.inbound.put_nowait(rpc('alpha', 'session.control.read'))
        assert await settle(lambda: 'done:alpha' in log), log
        assert 'start:beta-one' not in log and 'start:beta-two' not in log
    finally:
        started.set()
    assert await settle(lambda: 'done:beta-one' in log and 'done:beta-two' in log)
    sock.inbound.put_nowait(None)
    await asyncio.wait_for(task, 10)


@pytest.mark.asyncio
async def test_disconnect_drops_reads_waiting_for_profile_start(rig, starting_profile):
    sock, _, log = rig
    started, observed = starting_profile
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    sock.inbound.put_nowait({**rpc('beta', 'session.control.read'), 'params': {'profile': 'beta'}})
    assert await settle(lambda: bool(observed))
    sock.inbound.put_nowait(None)
    try:
        await asyncio.wait_for(task, 10)
        assert 'start:beta' not in log
    finally:
        started.set()


@pytest.mark.asyncio
async def test_cancelled_dispatch_does_not_report_a_completed_slow_dispatch(rig, monkeypatch, caplog):
    sock, gates, log = rig
    monkeypatch.setattr(ws_mod, '_WS_SLOW_DISPATCH_LOG_MS', 0)
    gates['session.context_breakdown'] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    with caplog.at_level(logging.WARNING, logger='tui_gateway.ws'):
        sock.inbound.put_nowait(rpc('cancelled'))
        try:
            assert await settle(lambda: 'start:cancelled' in log)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            release.set()
    assert not any('ws dispatch slow' in r.getMessage() for r in caplog.records)
