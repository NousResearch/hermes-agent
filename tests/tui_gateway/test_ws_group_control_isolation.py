"""Scheduling a room control must preserve its profile, socket and worker isolation."""

import asyncio

import pytest

from gateway.session_controls import AuthorityConnection
from tests.gateway.test_sibling_attach_while_profile_starting import (
    rig as multiplex_rig_fixture,
)

from tui_gateway import ws as ws_mod

multiplex_rig = multiplex_rig_fixture


def request(sock, rid, method, **params):
    sock.inbound.put_nowait(
        {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
    )


@pytest.mark.asyncio
async def test_queued_control_refuses_a_retargeted_profile(multiplex_rig, monkeypatch):
    sock = await multiplex_rig.connect()
    alias = multiplex_rig.alpha.parent / "selected"
    alias.symlink_to(multiplex_rig.alpha.parent.parent, target_is_directory=True)
    resolve = AuthorityConnection._sibling_for
    resolutions = []

    def retarget_after_lookup(connection, profile):
        target = resolve(connection, profile)
        if profile == alias.name and not resolutions:
            resolutions.append((target or connection).authority.profile_id)
            alias.unlink()
            alias.symlink_to(multiplex_rig.alpha, target_is_directory=True)
        return target

    monkeypatch.setattr(AuthorityConnection, "_sibling_for", retarget_after_lookup)
    request(
        sock,
        "control",
        "groups.stop",
        profile=alias.name,
        room_id="absent",
        cancel_id="stop",
    )
    reply = await sock.reply("control", timeout=10)
    assert resolutions == [str(multiplex_rig.alpha.parent.parent)]
    assert reply["error"]["data"]["reason"] == "profile_mismatch", reply


@pytest.mark.asyncio
async def test_sibling_initialization_error_is_request_scoped(
    multiplex_rig, monkeypatch
):
    from gateway import owner_identity

    sock = await multiplex_rig.connect()
    principal = owner_identity.principal_subject

    failed = False

    def identify(identity):
        nonlocal failed
        if not failed and identity.get("profile_id") == str(multiplex_rig.alpha):
            failed = True
            raise PermissionError("identity temporarily unavailable")
        return principal(identity)

    monkeypatch.setattr(owner_identity, "principal_subject", identify)
    request(
        sock,
        "control",
        "groups.stop",
        profile="alpha",
        room_id="absent",
        cancel_id="stop",
    )
    request(sock, "other", "ping")
    error = asyncio.create_task(sock.reply("control", timeout=10))
    try:
        await asyncio.wait({sock.task, error}, return_when=asyncio.FIRST_COMPLETED)
        assert error.done(), (
            "sibling initialization ended the socket without answering the request"
        )
        assert error.result()["error"]["code"] == -32603
        assert "result" in await sock.reply("other", timeout=10)
        request(sock, "retry", "groups.list", profile="alpha")
        assert "result" in await sock.reply("retry", timeout=10)
    finally:
        error.cancel()
        await asyncio.gather(error, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["session.transcript", "groups.list"])
async def test_starting_profile_reads_leave_the_reader_live(
    multiplex_rig, monkeypatch, method
):
    monkeypatch.setattr(ws_mod, "_WS_CONCURRENT_READ_LIMIT", 1)
    monkeypatch.setattr(ws_mod, "_WS_WAITING_READ_LIMIT", 1)
    sock = await multiplex_rig.connect()
    params = {"session_id": "absent"} if method == "session.transcript" else {}
    for rid in ("first", "overflow", "next"):
        request(sock, rid, method, profile="beta", **params)
    request(sock, "heartbeat", "gateway.ping")
    assert (await sock.reply("heartbeat", timeout=5))["result"]["ok"]
    for rid in ("overflow", "next"):
        reply = await sock.reply(rid, timeout=5)
        assert reply["error"]["data"]["reason"] == "runtime_draining"
    if method == "session.transcript":
        first = await sock.reply("first", timeout=5)
        assert first["error"]["data"]["reason"] == "runtime_draining"
    else:
        assert "first" not in sock.replies
        multiplex_rig.release.set()
        assert (await sock.reply("first"))["result"]["rooms"] == []


@pytest.mark.asyncio
async def test_waiting_group_controls_leave_workers_for_another_socket(
    multiplex_rig, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from gateway import session_group_controls

    origin, observer = await multiplex_rig.connect(), await multiplex_rig.connect()
    request(origin, "ready", "gateway.ping")
    request(observer, "ready", "gateway.ping")
    await origin.reply("ready")
    await observer.reply("ready")

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(pool)
    entered = {room: asyncio.Event() for room in ("first", "second")}
    release = threading.Event()
    group = session_group_controls._group

    def wait_for_peer(authority, actor, method, params, **kwargs):
        if method == "groups.stop":
            loop.call_soon_threadsafe(entered[params["room_id"]].set)
            assert release.wait(15)
        return group(authority, actor, method, params, **kwargs)

    monkeypatch.setattr(session_group_controls, "_group", wait_for_peer)
    try:
        for room in entered:
            request(origin, room, "groups.stop", room_id=room, cancel_id=room)
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())), 10
        )
        request(observer, "read", "groups.list")
        assert "result" in await observer.reply("read", timeout=5)
    finally:
        release.set()
        for room in entered:
            await origin.reply(room)
