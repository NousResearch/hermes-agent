"""Early close ordering and continued cleanup through the real Group Chat service."""

import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_link_records as links
from gateway import hosted_rooms as rooms
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def service(tmp_path, monkeypatch):
    import tui_gateway.server as server
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "home")
    (tmp_path / "profiles" / "ops").mkdir(parents=True)
    service = HostedRoomService(SimpleNamespace(), db_path=tmp_path / "state.db")
    service.create_room(room_id="room", name="Workshop", members=[
        {"member_id": "default", "profile": "default", "handle": "writer"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"},
    ])
    monkeypatch.setattr(server, "get_hosted_room_service", lambda: service)
    return service


@pytest.mark.parametrize("failure", ["stop", "revoke", None])
def test_registered_disband_persists_fence_before_stop_even_when_close_fails(service, monkeypatch, failure):
    import tui_gateway.server as server
    phases = []
    stop, revoke = service.stop_room, service.revoke_room_routes

    def stopped(*args, **kwargs):
        phases.append(("stop", links.room_link_retirement_started(service.db_path, room_id="room")))
        if failure == "stop":
            raise RuntimeError("exact work still stopping")
        return stop(*args, **kwargs)

    def revoked(*args, **kwargs):
        phases.append(("revoke", links.room_link_retirement_started(service.db_path, room_id="room")))
        if failure == "revoke":
            raise RuntimeError("peer unavailable")
        return revoke(*args, **kwargs)

    monkeypatch.setattr(service, "stop_room", stopped)
    monkeypatch.setattr(service, "revoke_room_routes", revoked)
    result = server._methods["groups.disband"](1, {"room_id": "room"})
    assert phases[0] == ("stop", True)
    assert links.room_link_retirement_started(service.db_path, room_id="room")
    room = rooms.room_state(service.db_path, room_id="room", include_disbanded=True)
    if failure:
        assert "error" in result
        assert room.get("disbanded_at") is None
        restarted = HostedRoomService(SimpleNamespace(), db_path=service.db_path)
        with pytest.raises((driver.RoomUnavailableError, rooms.HostedRoomError)):
            restarted.send(room_id="room", event_id="late-user", payload={"text": "New work", "thread_id": "thread"})
    else:
        assert "error" not in result
        assert phases == [("stop", True), ("revoke", True)]
        assert room["disbanded_at"] is not None
        assert server._methods["groups.disband"](2, {"room_id": "room"})["result"]["tombstone"]["idempotent"]


def seed_discussion(service):
    return rooms.append_event(
        service.db_path, room_id="room", event_id="seed", kind="message.user",
        actor={"kind": "user", "id": "desktop"}, payload={"text": "@ops inspect", "thread_id": "thread"},
        authority_gateway_id="home", authority_epoch=1)


def test_closing_prepare_publishes_accepted_terminal_without_admitting_more(service):
    seed_discussion(service)
    binding = service.bindings()[0]
    service.prepare_room(binding)
    task = driver.list_tasks(service.db_path, room_id="room")[0]
    held = driver.acquire_lease(service.db_path, room_id="room", gateway_id="home", authority_epoch=1,
                                process_generation="accepted", ttl_seconds=30, clock=time.time)
    attempt = driver.start_task(service.db_path, task["identity"], held, expected_cancel_generation=0, clock=time.time)
    service.begin_room_disband("room")
    driver.settle_task(service.db_path, attempt, settlement_id="accepted-receipt", status="settled",
                       result={"text": "Finished accepted work"}, clock=time.time)
    service.prepare_room(binding)
    assert any(event["kind"] == "message.member" for event in rooms.read_events(service.db_path, room_id="room")["events"])
    assert not driver.list_tasks(service.db_path, room_id="room", status="queued")
    replay = service.send(room_id="room", event_id="seed", payload={"text": "@ops inspect", "thread_id": "thread"})
    assert replay["idempotent"]


def test_close_between_policy_plan_and_admission_is_checked_in_sql(service, monkeypatch):
    from gateway import hosted_room_discussion as discussion
    seed_discussion(service)
    planned = discussion.plan_next_task
    second = HostedRoomService(SimpleNamespace(), db_path=service.db_path)

    def race(*args, **kwargs):
        decision = planned(*args, **kwargs)
        assert decision.status == "task"
        second.begin_room_disband("room")
        return decision

    monkeypatch.setattr(discussion, "plan_next_task", race)
    with pytest.raises(driver.RoomUnavailableError, match="being disbanded"):
        service.prepare_room(service.bindings()[0])
    assert driver.list_tasks(service.db_path, room_id="room") == []


def test_close_rejects_fresh_allow_and_retry_but_permits_deny(service):
    calls = []
    service.rpc = SimpleNamespace(approve=lambda **kwargs: calls.append(kwargs) or {"resolved": 1})
    service._set_pending_action("room", "ops", {
        "request_id": "approval", "task_id": "task", "execution_generation": 1, "session_id": "session"})
    service.begin_room_disband("room")
    kwargs = dict(room_id="room", member_id="ops", task_id="task", execution_generation=1, request_id="approval")
    with pytest.raises(driver.RoomUnavailableError, match="being disbanded"):
        service.approve_room_task(**kwargs, choice="once")
    with pytest.raises(driver.RoomUnavailableError, match="being disbanded"):
        service.retry_room_task("room", task_id="task")
    assert calls == []
    assert service.approve_room_task(**kwargs, choice="deny") == {"resolved": 1}
    assert [call["choice"] for call in calls] == ["deny"]


def test_begin_disband_keeps_foreign_authority_guard(service):
    rooms.create_room(service.db_path, room_id="foreign", name="Foreign", members=[], authority_gateway_id="peer")
    with pytest.raises(rooms.AuthorityConflictError):
        service.begin_room_disband("foreign")
    assert not links.room_link_retirement_started(service.db_path, room_id="foreign")


def test_closing_peer_observation_and_stop_are_not_failed_by_route_health_write(service):
    from gateway import hosted_room_links as stored_links
    from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
    catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
        installation_id="peer", target_profile="ops", persistent_process=True))
    stored = stored_links.make_stored_link(room_id="room", member_id="ops", target_url="http://127.0.0.1:9999",
                                          target_profile="ops", grant="accepted-grant", catalog=catalog,
                                          cancellation_scope_id="cancel", trace_id="trace")
    stored_links.save_room_link(service.db_path, stored)
    stored_links.mark_room_link_status(service.db_path, room_id="room", member_id="ops", status="unavailable")
    service._load_stored_links()
    calls = []
    client = SimpleNamespace(status=lambda **kw: calls.append("status") or {"active": True},
                             stop_receipt=lambda **kw: calls.append("stop") or {"status": "cancelled"})
    tracked = service._tracked_peer_client("room", "ops", client, binding=service.bindings()[0])
    service.begin_room_disband("room")
    assert tracked.status(grant=stored.grant)["active"]
    assert tracked.stop_receipt(grant=stored.grant)["status"] == "cancelled"
    assert calls == ["status", "stop"]
    assert stored_links.load_room_link(service.db_path, room_id="room", member_id="ops").status == "unavailable"
    with pytest.raises(rooms.HostedRoomError):
        stored_links.save_room_link(service.db_path, stored)
