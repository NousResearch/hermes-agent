"""Both ``groups.capabilities`` surfaces advertise friendly @mention tags (#89720).

The standalone TUI/Desktop backend and the native gateway serve the same room protocol; a client
gates its alias suggestions on this feature string, so neither surface may omit it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from tui_gateway import methods_groups

FEATURE = "room_mention_aliases_v1"


@pytest.fixture
def home(tmp_path, monkeypatch):
    class DurableRunStore:
        durable = True

    path = tmp_path / ".hermes"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    monkeypatch.setenv("API_SERVER_KEY", "gateway-api-key-1234567890")
    monkeypatch.setattr(srv, "_run_idempotency_store", DurableRunStore(), raising=False)
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def test_standalone_capabilities_advertise_friendly_mentions(home):
    result = srv._methods["groups.capabilities"](1, {})

    assert FEATURE in result["result"]["features"], result


def test_native_capabilities_advertise_friendly_mentions(tmp_path, monkeypatch):
    from gateway import session_group_peers, session_hosted_members
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB

    path = tmp_path / "state"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    monkeypatch.setattr(session_group_peers, "room_link", lambda authority: {"enabled": False, "reason": "test"})
    monkeypatch.setattr(
        session_hosted_members, "member_roster", lambda authority, siblings=True: ({}, {}))
    with SessionDB(path / "state.db") as db:
        authority = SimpleNamespace(profile_id=str(path), instance_id="owner", db=db, events={}, sessions={})

        async def probe():
            connection = AuthorityConnection(
                authority, object(), {"user_id": "reader", "capabilities": ["session:read"]})
            return await connection.dispatch({"id": 1, "method": "groups.capabilities", "params": {}})

        reply = asyncio.run(probe())

    result = reply["result"]
    assert FEATURE in result["features"], result
    assert "seatable_profiles" in result
