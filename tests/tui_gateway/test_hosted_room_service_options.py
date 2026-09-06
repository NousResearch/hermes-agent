"""Separate room storage from native profiles without trusting wire-supplied actors."""

from pathlib import Path
from types import ModuleType

import pytest

from gateway import hosted_rooms
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def service(tmp_path, monkeypatch):
    home = tmp_path / "runtime"
    (home / "profiles" / "planner").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    service = HostedRoomService(ModuleType("test_server"), db_path=home / "state.db")
    service.create_room(room_id="options-room", name="Options", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "planner", "profile": "planner", "handle": "planner"}])
    # No worker starts: send exercises real validation, SQLite and policy admission,
    # but nothing submits prompts or creates child sessions.
    return service


@pytest.mark.parametrize("separate_root", [False, True])
def test_profiles_and_turn_locks_share_the_selected_root(tmp_path, monkeypatch, separate_root):
    from tools.bot_relay import acquire_turn_lock, turn_lock_path, TurnBusyError

    home = tmp_path / "room-store"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "native-home" if separate_root else home
    (root / "profiles" / "planner").mkdir(parents=True)
    if separate_root:
        (home / "profiles" / "decoy").mkdir(parents=True)
    kwargs = {"profiles_root": root} if separate_root else {}
    service = HostedRoomService(ModuleType("test_server"), db_path=home / "state.db", **kwargs)

    assert service.root == home
    assert service.local_profiles() == ("default", "planner")
    service.create_room(room_id="profiles-room", name="Profiles", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "planner", "profile": "planner", "handle": "planner"}])
    with service._turn_lock("planner") as lock_path:
        assert lock_path == turn_lock_path(root, "planner")
        # Real kernel lock proves coordination with an independent native caller.
        with pytest.raises(TurnBusyError):
            with acquire_turn_lock(root, "planner", timeout_seconds=0):
                pytest.fail("another caller acquired the same profile turn")
    with acquire_turn_lock(root, "planner", timeout_seconds=0):
        pass


@pytest.mark.parametrize("actor_id", [None, False, 123, "", " \t\n", "x" * 129])
def test_invalid_trusted_actor_does_not_append_or_admit_work(service, actor_id):
    before = hosted_rooms.read_events(service.db_path, room_id="options-room")
    with pytest.raises(ValueError):
        service.send(room_id="options-room", event_id="bad-actor",
                     payload={"text": "Hello", "thread_id": "thread"}, actor_id=actor_id)
    assert hosted_rooms.read_events(service.db_path, room_id="options-room") == before


@pytest.mark.parametrize("actor_id", ["telegram:5953464757", "x" * 128])
def test_trusted_actor_persists_and_participates_in_idempotency(service, actor_id):
    kwargs = dict(room_id="options-room", event_id="trusted-actor",
                  payload={"text": "Hello", "thread_id": "thread"}, actor_id=actor_id)
    first = service.send(**kwargs)
    repeated = service.send(**kwargs)
    assert repeated["idempotent"] is True
    assert repeated["seq"] == first["seq"]
    stored = hosted_rooms.read_events(service.db_path, room_id="options-room")["events"][0]
    assert stored["actor"] == first["actor"] == {"kind": "user", "id": actor_id}
    with pytest.raises(hosted_rooms.EventConflictError):
        service.send(**{**kwargs, "actor_id": "different-user"})


def test_groups_send_cannot_override_desktop_actor(service, monkeypatch):
    import tui_gateway.server as server

    monkeypatch.setattr(server, "get_hosted_room_service", lambda: service)
    result = server._methods["groups.send"](1, {
        "room_id": "options-room", "event_id": "wire-actor",
        "actor_id": "telegram:forged", "actor": {"kind": "user", "id": "forged"},
        "payload": {"text": "Hello", "thread_id": "thread"},
    })
    assert "error" not in result, result
    assert result["result"]["event"]["actor"] == {"kind": "user", "id": "desktop"}
    stored = hosted_rooms.read_events(service.db_path, room_id="options-room")["events"][0]
    assert stored["actor"] == {"kind": "user", "id": "desktop"}
