"""Scheduling a room control must preserve its profile, socket and worker isolation."""

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
