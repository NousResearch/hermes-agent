"""A standalone backend sends Group Chat calls to the gateway that owns the profile (WA-19)."""

from __future__ import annotations

import pytest

import tui_gateway.server as srv
from tui_gateway import methods_groups, owner_forward

MEMBERS = [{"member_id": "default", "profile": "default", "handle": "hermes"},
           {"member_id": "ops", "profile": "ops", "handle": "ops"}]


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / ".hermes"
    (path / "profiles" / "ops").mkdir(parents=True)
    (path / "profiles" / "ops" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(path))
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


ENDPOINT = object()


def _discovery(monkeypatch, *states):
    """Gateway discovery answers the given states in turn (the last one repeats)."""
    from hermes_cli import gateway_runtime
    seen = list(states)

    def discover(home, *, timeout=2.0):
        state = seen.pop(0) if len(seen) > 1 else seen[0]
        return gateway_runtime.GatewayDiscovery(state, endpoint=ENDPOINT if state == "ready" else None)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint", discover)


@pytest.fixture
def owner(monkeypatch):
    """A ready gateway serving the home; records what reached it."""
    calls = []
    endpoint = ENDPOINT
    _discovery(monkeypatch, "ready")

    def forward_rpc(home, seen_endpoint, method, params):
        assert seen_endpoint is endpoint
        calls.append((method, dict(params), home))
        if method == "groups.capabilities":
            return {"driver": True, "methods": ["groups.list", "groups.create", "groups.attachment.upload"]}
        if method == "groups.list":
            return {"rooms": [{"room_id": "room-webapp"}], "next_offset": None}
        return {"room": {"room_id": params.get("room_id")}}
    monkeypatch.setattr(owner_forward, "forward_rpc", forward_rpc)
    return calls


def test_room_calls_go_to_the_owning_gateway_and_nothing_lands_locally(home, owner):
    created = srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})
    listed = srv._methods["groups.list"](2, {})

    assert created["result"]["room"]["room_id"] == "room-desk"
    assert listed["result"]["rooms"] == [{"room_id": "room-webapp"}]
    assert [call[0] for call in owner] == ["groups.create", "groups.list"]
    from gateway.hosted_rooms import default_db_path, list_rooms
    assert list_rooms(default_db_path()) == []  # no second store, no local driver for it


def test_capabilities_advertise_only_what_this_wire_carries(home, owner):
    result = srv._methods["groups.capabilities"](1, {})["result"]
    assert result["methods"] == ["groups.list", "groups.create"]


def test_owner_refusal_is_returned_not_run_locally(home, monkeypatch):
    from hermes_cli.gateway_client import GatewayRPCError
    _discovery(monkeypatch, "ready")

    def refuse(*_args):
        raise GatewayRPCError("permission_denied", code=4030, data={"reason": "permission_denied"})
    monkeypatch.setattr(owner_forward, "forward_rpc", refuse)

    reply = srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})
    assert reply["error"]["code"] == 4030 and reply["error"]["data"] == {"reason": "permission_denied"}
    from gateway.hosted_rooms import default_db_path, list_rooms
    assert list_rooms(default_db_path()) == []


def test_without_a_gateway_rooms_stay_standalone(home):
    created = srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})
    assert "error" not in created, created
    assert [r["room_id"] for r in srv._methods["groups.list"](2, {})["result"]["rooms"]] == ["room-desk"]


def test_gateway_less_list_hides_rooms_handed_to_the_gateway(home):
    from gateway.hosted_rooms import default_db_path
    from gateway.hosted_rooms_gateway_import import import_standalone_rooms
    import sqlite3
    srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})
    target = default_db_path().with_name("state.db")
    with sqlite3.connect(target) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
    assert import_standalone_rooms(target) == 1

    assert srv._methods["groups.list"](2, {})["result"]["rooms"] == []
    refused = srv._methods["groups.send"](3, {"room_id": "room-desk", "event_id": "e1", "payload": {"text": "hi"}})
    assert "error" in refused  # the standalone copy is fenced: one driver per room


def test_a_starting_gateway_is_waited_for_and_nothing_lands_locally(home, owner, monkeypatch):
    from tui_gateway import methods_groups_forward
    monkeypatch.setattr(methods_groups_forward, "_READY_POLL_S", 0.0, raising=False)
    _discovery(monkeypatch, "starting", "starting", "ready")

    created = srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})

    assert created["result"]["room"]["room_id"] == "room-desk"
    assert [call[0] for call in owner] == ["groups.create"]
    from gateway.hosted_rooms import default_db_path, list_rooms
    assert list_rooms(default_db_path()) == []  # never a room the gateway's list would not show


def test_a_named_profile_gateway_never_hides_the_standalone_rooms(home, owner, monkeypatch):
    _discovery(monkeypatch, "absent")  # Desktop on its own first
    srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})
    _discovery(monkeypatch, "ready")  # then `hermes -p ops gateway` serves the selected profile

    listed = srv._methods["groups.list"](2, {"profile": "ops"})["result"]["rooms"]
    state = srv._methods["groups.state"](3, {"profile": "ops", "room_id": "room-desk"})

    assert [room["room_id"] for room in listed] == ["room-webapp", "room-desk"]
    assert state["result"]["room"]["name"] == "Desk"  # served by the store that holds it
    assert [call[0] for call in owner] == ["groups.list"]


def test_a_named_profile_reaches_the_rooms_handed_to_the_root_gateway(home, owner, monkeypatch):
    import sqlite3
    from gateway.hosted_rooms import default_db_path
    from gateway.hosted_rooms_gateway_import import import_standalone_rooms
    _discovery(monkeypatch, "absent")
    srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})
    target = default_db_path().with_name("state.db")
    with sqlite3.connect(target) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
    assert import_standalone_rooms(target) == 1  # the default profile's gateway took it over
    _discovery(monkeypatch, "ready")

    srv._methods["groups.state"](2, {"profile": "ops", "room_id": "room-desk"})

    method, params, reached = owner[-1]
    assert (method, reached) == ("groups.state", default_db_path().parent)
    assert "profile" not in params  # the root gateway answers for its own profile


def test_a_handed_room_is_never_renamed_in_the_standalone_store_while_its_gateway_is_down(home, monkeypatch):
    """A room handed to the gateway lives in the gateway's store; the standalone copy is frozen. With
    the gateway down, a rename (or any call on that room) is a retryable refusal, never a local
    write the gateway would never see and the import marker would never copy again."""
    import sqlite3
    from gateway.hosted_rooms import default_db_path, list_rooms
    from gateway.hosted_rooms_gateway_import import import_standalone_rooms
    from tui_gateway.methods_groups_forward import OWNER_LOST_CODE
    _discovery(monkeypatch, "absent")
    srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Before", "members": MEMBERS})
    target = default_db_path().with_name("state.db")
    with sqlite3.connect(target) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
    assert import_standalone_rooms(target) == 1

    reply = srv._methods["groups.rename"](2, {"room_id": "room-desk", "event_id": "rename-1", "name": "After"})

    assert reply.get("error", {}).get("code") == OWNER_LOST_CODE, reply
    [kept] = list_rooms(default_db_path(), include_disbanded=True)
    assert kept["name"] == "Before"
