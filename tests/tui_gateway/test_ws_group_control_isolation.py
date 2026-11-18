"""Scheduling a room control must preserve its profile, socket and worker isolation."""

import asyncio

import pytest

from gateway.session_controls import AuthorityConnection
from tests.gateway.test_sibling_attach_while_profile_starting import (
    rig as multiplex_rig_fixture,
)

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
