"""A sleeping group member must not occupy the shared socket's control dispatcher."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from tests.gateway.test_session_group_peers import (
    _seat, call, gateway as gateway_fixture, linked_peer as linked_peer_fixture,
)
from tests.gateway.test_sibling_attach_while_profile_starting import _Socket, _on_disconnect_path
from tests.tui_gateway.test_ws_concurrent_reads import rig as dispatch_rig
from tests.waiting import async_wait_until
from tui_gateway import server, ws as ws_mod
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError

gateway = gateway_fixture
linked_peer = linked_peer_fixture
rig = dispatch_rig


def rpc(request_id, method, **params):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


@pytest.fixture
def socket_setup(monkeypatch):
    monkeypatch.setattr(ws_mod, "_start_backend_liveness", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "_ensure_lease_watcher", lambda: None)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["disband", "stop"])
async def test_disband_waiting_for_peer_keeps_group_reads_responsive(
        gateway, linked_peer, socket_setup, monkeypatch, operation):
    _, url, catalog, room = linked_peer
    assert (await _seat(gateway, room, url, catalog))[1]["registered"]
    runtime = gateway.service.runtime
    monkeypatch.setattr(runtime, "status", lambda: {**type(runtime).status(runtime), "running": True})
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def sleeping_member(*args, **kwargs):
        attempts.append(args)
        entered.set()
        assert release.wait(15)
        raise PeerRunsHTTPError("member asleep", retryable=True, ambiguous=True)

    client = gateway.service.peer_clients["linked", "reviewer"]
    if operation == "stop":
        assert (await call(gateway.owner, "groups.send", room_id="linked", event_id="ask",
                           payload={"text": "@reviewer review", "thread_id": "thread"}))["accepted"]
        turn, = driver.list_tasks(gateway.db.db_path, room_id="linked", status="queued")
        lease = runtime._ensure_lease(gateway.service.bindings()[0])
        driver.start_task(gateway.db.db_path, turn["identity"], lease, expected_cancel_generation=0,
                          clock=runtime.clock)
    monkeypatch.setattr(client, "_request", sleeping_member)
    sock = _Socket(gateway.authority)
    task = asyncio.create_task(ws_mod.handle_ws(sock, auth_identity={"user_id": "owner"}))
    sock.inbound.put_nowait(rpc("slow", f"groups.{operation}", room_id="linked"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        for method in ("groups.state", "groups.log", "groups.list", "groups.capabilities"):
            params = {"room_id": "linked"} if method in {"groups.state", "groups.log"} else {}
            sock.inbound.put_nowait(rpc(method, method, **params))
        for method in ("groups.state", "groups.log", "groups.list", "groups.capabilities"):
            assert "result" in await sock.reply(method, timeout=5)
        assert "slow" not in sock.replies
    finally:
        release.set()
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(task, 10)
    if operation == "disband":
        assert len(attempts) == 1
        assert sock.replies["slow"]["error"]["data"]["reason"] == "peer_unreachable"
    else:
        assert len(attempts) == 2  # an uncertain admission is replayed once, inline in Stop
        assert sock.replies["slow"]["result"]["cancelled"] == 1
        assert driver.get_task(gateway.db.db_path, turn["identity"])["status"] == "stopping"
    assert hosted_rooms.room_state(gateway.db.db_path, room_id="linked").get("disbanded_at") is None


@pytest.fixture
def canonical_rig(rig, monkeypatch):
    async def close():
        rig[2].append("authority closed")

    connection = SimpleNamespace(
        actor=SimpleNamespace(profile_id="default"),
        awaits_profile_start=lambda req, held: None, close=close)
    connection._sibling_for = lambda profile: (None if profile == "default" else
                                             SimpleNamespace(actor=SimpleNamespace(profile_id=profile)))
    monkeypatch.setattr(ws_mod, "_authority_connection", lambda *args: connection)
    return rig


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_method", ["groups.stop", "groups.disband"])
async def test_slow_group_control_does_not_hold_other_controls(canonical_rig, slow_method):
    sock, gates, log = canonical_rig
    gates[slow_method] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    sock.inbound.put_nowait(rpc("slow", slow_method, room_id="sleeping"))
    try:
        await async_wait_until(lambda: "start:slow" in log)
        # The other Stop must have its own gate, not the sleeping room's stand-in.
        gates.pop(slow_method)
        for method in ("groups.stop", "groups.approve", "session.interrupt", "prompt.submit",
                       "approval.respond", "clarify.lock", "gateway.ping"):
            sock.inbound.put_nowait(rpc(method, method, room_id="other"))
        sock.inbound.put_nowait(rpc("profile", "groups.send", room_id="sleeping", profile="other"))
        await async_wait_until(
            lambda: len([frame for frame in sock.sent if frame.get("id")]) == 8,
            timeout=5, what="unrelated controls answered while the member sleeps")
        assert "done:slow" not in log
    finally:
        release.set()
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(task, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "default"])
async def test_stop_then_send_keeps_its_fence_across_profile_aliases(
        gateway, linked_peer, socket_setup, monkeypatch, profile):
    """Delay Stop BEFORE its durable fence: a Send that overtakes it would be cancelled."""
    _, url, catalog, room = linked_peer
    assert (await _seat(gateway, room, url, catalog))[1]["registered"]
    entered, release = threading.Event(), threading.Event()
    stop = gateway.service.stop_room

    def paused_stop(*args, **kwargs):
        entered.set()
        assert release.wait(15)
        return stop(*args, **kwargs)

    monkeypatch.setattr(gateway.service, "stop_room", paused_stop)
    sock = _Socket(gateway.authority)
    task = asyncio.create_task(ws_mod.handle_ws(sock, auth_identity={"user_id": "owner"}))
    sock.inbound.put_nowait(rpc("stop", "groups.stop", room_id="linked", cancel_id="stop"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        sock.inbound.put_nowait(rpc("send", "groups.send", room_id="linked", event_id="next",
                                   payload={"text": "@reviewer Next", "thread_id": "next"},
                                   **({"profile": profile} if profile else {})))
        sock.inbound.put_nowait(rpc("read", "groups.log", room_id="linked"))
        assert "result" in await sock.reply("read", timeout=5)
        assert "send" not in sock.replies
        release.set()
        assert (await sock.reply("send"))["result"]["accepted"]
        assert sock.order.index("stop") < sock.order.index("send")
        events = hosted_rooms.read_events(gateway.db.db_path, room_id="linked")["events"]
        fence = next(e for e in events if e["kind"] == "room.stop_requested")
        message = next(e for e in events if e["event_id"] == hosted_rooms.user_event_id("next"))
        assert message["seq"] > fence["seq"]
        assert (await call(gateway.owner, "groups.stop", room_id="linked", cancel_id="stop"))["cancelled"] == 0
        assert len(driver.list_tasks(gateway.db.db_path, room_id="linked", status="queued")) == 1
    finally:
        release.set()
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(task, 10)


@pytest.mark.asyncio
async def test_profile_start_preserves_group_order_and_concurrent_reads(canonical_rig):
    sock, gates, log = canonical_rig
    started = asyncio.Event()
    connection = ws_mod._authority_connection(None)

    def starting(req, held):
        if req["params"].get("profile") == "new" and (not started.is_set() or "new" in held):
            return "new", started
        return None

    connection.awaits_profile_start = starting
    gates["groups.disband"] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    for req in (rpc("slow", "groups.disband", room_id=" room ", profile="new"),
                rpc("send", "groups.send", room_id="room", profile="new"),
                rpc("read", "groups.state", room_id="room", profile="new"),
                rpc("other", "groups.stop", room_id="other", profile="new"),
                rpc("serial", "prompt.submit")):
        sock.inbound.put_nowait(req)
    try:
        await async_wait_until(lambda: "done:serial" in log)
        assert log == ["start:serial", "done:serial"]
        started.set()
        await async_wait_until(lambda: all(f"done:{key}" in log for key in ("read", "other")))
        assert "start:slow" in log and "start:send" not in log
        release.set()
        await async_wait_until(lambda: "done:send" in log)
        assert log.index("done:slow") < log.index("start:send")
    finally:
        started.set()
        release.set()
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(task, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("distinct_rooms", [False, True])
async def test_group_control_backpressure_is_bounded(canonical_rig, monkeypatch, distinct_rooms):
    sock, gates, log = canonical_rig
    monkeypatch.setattr(ws_mod, "_WS_DISPATCH_QUEUE_MAX", 3)
    gates["groups.stop"] = release = asyncio.Event()
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    for n in range(40):
        sock.inbound.put_nowait(rpc(n, "groups.stop", room_id=str(n) if distinct_rooms else "room"))
    try:
        await async_wait_until(lambda: sock.received >= 8)
        assert sock.received == 8  # three held, three queued, dispatcher and reader each hold one
        assert not sock.inbound.empty()
        assert sum(e.startswith("start:") for e in log) == (3 if distinct_rooms else 1)
        release.set()
        await async_wait_until(lambda: sum(e.startswith("done:") for e in log) == 40)
    finally:
        release.set()
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(task, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_disconnect_drops_group_controls_that_never_started(canonical_rig, monkeypatch, shutdown):
    sock, gates, log = canonical_rig
    gates["groups.stop"] = release = asyncio.Event()
    disconnected = asyncio.Event()
    receive = sock.receive_text
    sock.on_disconnect = []

    async def receive_text():
        try:
            return await receive()
        except ws_mod._WebSocketDisconnect:
            for step in sock.on_disconnect:
                step()
            raise

    monkeypatch.setattr(sock, "receive_text", receive_text)
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    sock.task = task
    _on_disconnect_path(monkeypatch, sock, disconnected.set)
    sock.inbound.put_nowait(rpc("stop", "groups.stop", room_id="room"))
    sock.inbound.put_nowait(rpc("send", "groups.send", room_id="room"))
    sock.inbound.put_nowait(rpc("read", "groups.log", room_id="room"))
    try:
        await async_wait_until(lambda: "start:stop" in log and "done:read" in log)
        if shutdown:
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=5)
            assert task in done
            assert "cancelled:stop" in log
        else:
            sock.inbound.put_nowait(None)
            await asyncio.wait_for(disconnected.wait(), 5)
            assert "authority closed" not in log
            release.set()
            await asyncio.wait_for(task, 5)
            assert "done:stop" in log
        assert "start:send" not in log
        assert log[-1] == "teardown"
    finally:
        release.set()
        sock.inbound.put_nowait(None)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_method", ["groups.state", "groups.stop"])
async def test_failed_reply_preserves_running_group_and_serial_controls(canonical_rig, failed_method):
    sock, gates, log = canonical_rig
    gates["groups.disband"] = group_release = asyncio.Event()
    gates["prompt.submit"] = serial_release = asyncio.Event()
    gates[failed_method] = read_release = asyncio.Event()
    sock.fail_ids.add("fail")
    task = asyncio.create_task(ws_mod.handle_ws(sock))
    for req in (rpc("group", "groups.disband", room_id="sleeping"),
                rpc("fail", failed_method, room_id="other"), rpc("serial", "prompt.submit")):
        sock.inbound.put_nowait(req)
    try:
        await async_wait_until(lambda: all(f"start:{key}" in log for key in ("group", "fail", "serial")))
        read_release.set()
        await async_wait_until(lambda: "done:fail" in log)
        assert not task.done()
        group_release.set()
        serial_release.set()
        await asyncio.wait_for(task, 5)
        assert all(f"done:{key}" in log for key in ("group", "serial"))
        assert not any(e.startswith("cancelled:") for e in log)
        assert log[-1] == "teardown"
    finally:
        for gate in (read_release, group_release, serial_release):
            gate.set()
        sock.inbound.put_nowait(None)
        await asyncio.gather(task, return_exceptions=True)
