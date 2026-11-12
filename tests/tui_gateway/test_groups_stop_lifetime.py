"""Both room-control transports must stop newer work on a fresh Stop or Disband."""

import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from gateway.session_group_controls import _group
from hermes_agent.hermes_state import SessionDB
from tests.tui_gateway.test_hosted_room_service import _FakeRPC
from tests.tui_gateway.test_hosted_room_stop_replay import _service


@pytest.fixture(params=["canonical", "standalone"])
def controls(request, tmp_path, monkeypatch):
    import hermes_agent.tui_gateway.server as srv
    from hermes_agent.tui_gateway import methods_groups_forward

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with SessionDB(tmp_path / "state.db") as db:
        service = _service(db.db_path)
        service.create_room(room_id="room", name="Stop lifetime", members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"}])
        monkeypatch.setattr(service.runtime, "status", lambda: {"running": True})
        monkeypatch.setattr(srv, "get_hosted_room_service", lambda: service)
        monkeypatch.setattr(methods_groups_forward, "owner_endpoint", lambda _home: None)
        # Ownership is supplied by the authenticated connection in production.
        service.authorize_room = lambda *_args, **_kwargs: None
        authority = SimpleNamespace(db=db, hosted_room_service=service)

        def call(method, **params):
            params = {"room_id": "room", **params}
            if request.param == "canonical":
                return _group(authority, SimpleNamespace(subject="alice"), method, params)
            result = srv._methods[method](1, params)
            if "error" in result:
                raise RuntimeError(result["error"]["message"])
            return result["result"]

        yield service, call


@pytest.mark.parametrize("cancel_id", [None, "explicit-stop"])
def test_successive_stops_use_a_fresh_fence_unless_explicitly_replayed(controls, cancel_id):
    service, call = controls
    params = {} if cancel_id is None else {"cancel_id": cancel_id}
    call("groups.send", event_id="A", payload={"text": "@ops A", "thread_id": "A"})
    assert call("groups.stop", **params) == {"cancelled": 1}
    call("groups.send", event_id="B", payload={"text": "@ops B", "thread_id": "B"})
    assert call("groups.stop", **params) == {"cancelled": 1 if cancel_id is None else 0}
    queued = driver.list_tasks(service.db_path, room_id="room", status="queued")
    assert len(queued) == (0 if cancel_id is None else 1)


def test_retried_disband_waits_for_work_admitted_after_its_first_fence(controls):
    service, call = controls

    class PendingStopRPC(_FakeRPC):
        def info(self, **_kwargs):
            return {"active": True, "task_id": task["identity"].task_id}

        def interrupt(self, **_kwargs):
            return None

    service.runtime.rpc = rpc = PendingStopRPC()
    rpc.sessions[("ops", "Group: room")] = {"session_id": "ops-session"}
    binding, = service.bindings()
    lease = service.runtime._ensure_lease(binding)
    for event_id in ("A", "B"):
        call("groups.send", event_id=event_id, payload={"text": f"@ops {event_id}", "thread_id": event_id})
        task, = driver.list_tasks(service.db_path, room_id="room", status="queued")
        driver.start_task(service.db_path, task["identity"], lease, expected_cancel_generation=0, clock=time.time)
        with pytest.raises(RuntimeError, match="still stopping"):
            call("groups.disband", cancel_id="same-disband")
        assert hosted_rooms.room_state(service.db_path, room_id="room").get("disbanded_at") is None
        task = driver.get_task(service.db_path, task["identity"])
        assert task["status"] == "stopping"
        driver.complete_task_cancel(
            service.db_path, task["identity"], cancel_id=task["cancel_id"],
            expected_cancel_generation=task["cancel_generation"], clock=time.time)
    assert call("groups.disband", cancel_id="same-disband")["tombstone"]
