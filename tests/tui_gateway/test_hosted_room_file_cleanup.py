"""Accepted file turns do not need private bytes or admission preflight to stop."""

import hashlib
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from gateway.platforms import api_server
from tests.tui_gateway.test_groups_methods import home
from tests.tui_gateway.test_hosted_room_idle_grant_renewal import renewal


@pytest.mark.parametrize("retiring", [False, True])
@pytest.mark.parametrize("rotated", [False, True])
def test_accepted_file_reads_and_stop_survive_lost_file_access(renewal, monkeypatch, retiring, rotated):
    r = renewal
    binding = r.service.bindings()[0]
    r.peer.clock = lambda: r.clock[0]
    r.service.runtime.maintain_leased_room = None
    original_request = r.peer._request
    original_preflight = r.service._refresh_peer_attachment_catalog
    preflights, staged, admissions, stops, forbidden, errors = [], [], [], [], [], []
    target = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    target._room_grant_secret = lambda: r.secret

    def preflight(*args):
        preflights.append(True)
        return original_preflight(*args)

    def stage(**kwargs):
        staged.append(kwargs["attachments"])
        return {"complete": True}

    def request(path, *, method="GET", body=None, room_grant=None, **kwargs):
        if path.startswith("/v1/runs"):
            permission = "stop" if path.endswith("/stop") else "dispatch" if method == "POST" else "status"
            assert target._check_run_auth(SimpleNamespace(
                headers={"Authorization": f"HermesRoom {room_grant}"}, path=path, method=method,
            ), permission=permission) is None
            if permission == "dispatch":
                admissions.append(body["hosted_room_dispatch"])
            if permission == "stop":
                stops.append(True)
            return {"run_id": "accepted-file-run", "status": "cancelled" if stops else "running"}
        return original_request(path, method=method, body=body, room_grant=room_grant, **kwargs)

    monkeypatch.setattr(r.peer, "_request", request)
    monkeypatch.setattr(r.peer, "stage_attachments", stage)
    monkeypatch.setattr(r.service, "_refresh_peer_attachment_catalog", preflight)
    data = b"Private release notes for one review."
    uploaded = r.service.put_attachment(room_id=binding.room_id, upload_id="cleanup-file", kind="file",
                                         name="notes.txt", mime="text/plain", data=data)
    manifest = {key: uploaded[key] for key in ("attachment_id", "kind", "name", "size", "mime")}
    r.service.send(room_id=binding.room_id, event_id="file-stop", payload={
        "text": "@ops Review these notes", "thread_id": "file-stop-thread", "attachments": [manifest],
    })
    (task,) = driver.list_tasks(r.service.db_path, room_id=binding.room_id, status="queued")

    def inaccessible(*args, **kwargs):
        forbidden.append(True)
        raise RuntimeError("Private file access is no longer available")

    def stop_after_admission(_timeout=None):
        assert len(admissions) == len(staged) == len(preflights) == 1
        assert staged[0][0]["data"] == data
        assert admissions[0]["attachment_manifest_digest"]
        if rotated:
            refreshed = r.peer.refresh_grant(grant=r.old)
            r.service._rotate_route_grant(binding.room_id, "ops", refreshed["grant"],
                                          expected_grant_sha256=hashlib.sha256(r.old.encode()).hexdigest())
        monkeypatch.setattr(r.service, "_load_task_attachments", inaccessible)
        monkeypatch.setattr(r.service, "_refresh_peer_attachment_catalog", inaccessible)
        monkeypatch.setattr(r.peer, "stage_attachments", inaccessible)
        if retiring:
            r.service.begin_room_disband(binding.room_id)
        current = driver.get_task(r.service.db_path, task["identity"])
        try:
            observer = r.service._resolve_member_transport(binding, current)
            session = observer.resolve_exact(profile="ops", title=f"Group: {binding.room_id}", source="bot_room")
            assert observer.info(profile="ops", session_id=session["session_id"], source="bot_room")["active"]
        except RuntimeError as exc:
            errors.append(str(exc))
        try:
            r.service.stop_room(binding.room_id, cancel_id="file-stop", require_acknowledged=True)
        except RuntimeError as exc:
            errors.append(str(exc))
        if errors:
            r.service.runtime._stop.set()
        return False

    monkeypatch.setattr(r.service.runtime._wake, "wait", stop_after_admission)
    r.service.runtime._run_cycle()
    assert errors == [] and forbidden == []
    assert driver.get_task(r.service.db_path, task["identity"])["status"] == "cancelled"
    assert len(admissions) == len(stops) == 1
    if rotated:
        assert hosted_rooms.room_grant_is_revoked(r.service.db_path, claims=r.claims)
