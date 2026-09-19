"""An accepted peer observer adopts only a same-scope, CAS-published renewal."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_room_links, hosted_room_link_records, hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping
from gateway.hosted_room_execution_policy import execution_policy_mapping
from gateway.platforms import api_server
from tests.tui_gateway.test_groups_methods import home
from tests.tui_gateway.test_hosted_room_idle_grant_renewal import renewal
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


@pytest.mark.parametrize("maintenance", ["disabled", "poll", "racing_read"])
def test_active_peer_settles_once_across_exact_grant_retirement(renewal, monkeypatch, maintenance):
    r = renewal
    r.clock[0] = r.claims["expires_at"] - 360
    started = r.clock[0]
    r.peer.clock = lambda: r.clock[0]
    original_request = r.peer._request
    target = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    target._room_grant_secret = lambda: r.secret
    rejected, admissions, reads = [], [], []
    raced = False
    binding = r.service.bindings()[0]

    def request(path, *, method="GET", body=None, room_grant=None, **kwargs):
        nonlocal raced
        if path in {"/v1/runs", "/v1/runs/active-peer-run"}:
            if (maintenance == "racing_read" and not raced and method == "GET"
                    and r.clock[0] - started >= 60):
                raced = True
                r.service._renew_idle_peer_grants(binding, r.service.runtime._leases[binding.room_id])
            permission = "dispatch" if method == "POST" else "status"
            error = target._check_run_auth(SimpleNamespace(
                headers={"Authorization": f"HermesRoom {room_grant}"}, path=path, method=method,
            ), permission=permission)
            if error is not None:
                rejected.append(permission)
                raise PeerRunsHTTPError("target refused retired grant", status_code=error.status,
                                        error_code=json.loads(error.text)["error"]["code"])
            if method == "POST":
                admissions.append(body["hosted_room_dispatch"]["task_id"])
            else:
                reads.append(room_grant)
            complete = r.clock[0] - started >= 400
            return {"run_id": "active-peer-run", "status": "completed" if complete else "running",
                    "output": "Healthy peer result" if complete else ""}
        assert r.peer.timeout_seconds <= 1
        return original_request(path, method=method, body=body, room_grant=room_grant, **kwargs)

    monkeypatch.setattr(r.peer, "_request", request)
    if maintenance != "poll":
        r.service.runtime.maintain_leased_room = None
    r.service.send(room_id=binding.room_id, event_id="active-peer", payload={
        "text": "@ops Complete one healthy remote task", "thread_id": "active-peer-thread",
    })
    (task,) = driver.list_tasks(r.service.db_path, room_id=binding.room_id, status="queued")

    def tick(_timeout=None):
        r.clock[0] += 5
        assert r.clock[0] - started <= 450
        return False

    monkeypatch.setattr(r.service.runtime._wake, "wait", tick)
    r.service.runtime._run_cycle()
    assert driver.get_task(r.service.db_path, task["identity"])["status"] == "settled"
    r.service.runtime._run_cycle()
    events = hosted_rooms.read_events(r.service.db_path, room_id=binding.room_id)["events"]
    visible = [e for e in events if e["kind"] == "message.member" and e["payload"]["text"] == "Healthy peer result"]
    assert len(visible) == 1 and len(admissions) == 1
    assert len([e for e in events if e["kind"] == "turn.settled"]) == 1
    driver.require_active_lease(r.service.db_path, r.service.runtime._leases[binding.room_id], clock=lambda: r.clock[0])
    link = hosted_room_links.load_room_link(r.service.db_path, room_id=binding.room_id, member_id="ops")
    assert link.status == "ready"
    assert rejected == (["status"] if maintenance == "racing_read" else [])
    if maintenance != "disabled":
        assert link.grant != r.old and reads[-1] == link.grant
        assert hosted_rooms.room_grant_is_revoked(r.service.db_path, claims=r.claims)
    assert r.peer.timeout_seconds == 30


@pytest.mark.parametrize("change", ["same_scope", "url", "profile", "installation", "policy", "catalog", "cancel", "trace",
                                     "home", "membership", "epoch", "authority", "retired", "removed", "reauthorization"])
@pytest.mark.parametrize("retiring", [False, True])
def test_observer_adopts_only_same_scope_and_preserves_exact_cleanup(renewal, monkeypatch, change, retiring):
    r = renewal
    key = ("renewal-room", "ops")
    binding = r.service.bindings()[0]
    route = r.service.peer_routes[key]
    tracked = r.service._tracked_peer_client(*key, r.peer, route=route, binding=binding)
    refreshed = r.peer.refresh_grant(grant=r.old)
    r.service._rotate_route_grant(*key, refreshed["grant"], expected_grant_sha256=hashlib.sha256(r.old.encode()).hexdigest())
    stored = hosted_room_links.load_room_link(r.service.db_path, room_id=key[0], member_id=key[1])
    changed = {"url": {"target_url": "https://another.example"}, "profile": {"target_profile": "other"},
               "cancel": {"cancellation_scope_id": "another-cancel"}, "trace": {"trace_id": "another-trace"},
               "reauthorization": {"status": "needs_reauthorization"}}
    if change in {"installation", "policy", "catalog"}:
        catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
            installation_id="another-install" if change == "installation" else route.target_install_id,
            persistent_process=change != "catalog",
            execution_policy=execution_policy_mapping(target_profile="ops", config={"agent": {"max_turns": 7}})
            if change == "policy" else stored.catalog.execution_policy.as_mapping(),
        ))
        changed[change] = {"catalog": catalog}
    if change in changed:
        hosted_room_links.save_room_link(r.service.db_path, replace(stored, **changed[change]),
                                         expected_grant_sha256=hashlib.sha256(stored.grant.encode()).hexdigest())
    elif change == "home":
        r.service.peer_routes[key] = replace(r.service.peer_routes[key], home_install_id="another-home")
    elif change in {"retired", "removed"}:
        scope = {"room_id": key[0], "authority_gateway_id": binding.gateway_id, "authority_epoch": binding.authority_epoch}
        hosted_room_link_records.begin_room_link_retirement(r.service.db_path, **scope)
        if change == "removed":
            r.peer.revoke_grant_exact(grant=stored.grant)
            hosted_room_link_records.complete_room_link_retirement(r.service.db_path, **scope)
            hosted_room_link_records.delete_room_link_records(r.service.db_path, room_id=key[0])
    elif change != "same_scope":
        with sqlite3.connect(r.service.db_path) as conn:
            if change == "membership":
                members = r.service._room(key[0])["members"]
                members[1]["target"]["profile"] = "other"
                conn.execute("UPDATE hosted_rooms SET members_json=? WHERE room_id=?", (json.dumps(members), key[0]))
            elif change == "epoch":
                conn.execute("UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id=?", (key[0],))
            else:
                conn.execute("UPDATE hosted_rooms SET authority_gateway_id='another-home' WHERE room_id=?", (key[0],))
    if retiring and change not in {"retired", "removed"}:
        current_room = r.service._room(key[0])
        hosted_room_link_records.begin_room_link_retirement(
            r.service.db_path, room_id=key[0], authority_gateway_id=current_room["authority_gateway_id"],
            authority_epoch=current_room["authority_epoch"])
    sent = []
    monkeypatch.setattr(r.peer, "history", lambda **kw: sent.append(kw))
    if change in {"same_scope", "retired"}:
        tracked.history(room_id=key[0], profile="ops", session_id="accepted-run", grant=r.old)
        assert sent[0]["grant"] == stored.grant
        sent.clear()
    else:
        with pytest.raises(RuntimeError):
            tracked.history(room_id=key[0], profile="ops", session_id="accepted-run", grant=r.old)
        assert sent == []
    assert hosted_rooms.room_grant_is_revoked(r.service.db_path, claims=r.claims)
    monkeypatch.setattr(r.peer, "revoke_grant_exact", lambda **kw: sent.append(kw["grant"]))
    tracked.revoke_grant_exact(grant=r.old)
    assert sent == [r.old]  # Exact cleanup never substitutes the current bearer.
    if retiring or change in {"retired", "removed"}:
        with pytest.raises(RuntimeError, match="no longer current"):
            tracked.probe(grant=stored.grant)


@pytest.mark.parametrize("rotated", [False, True])
def test_disband_fence_allows_accepted_peer_reads_and_stop_without_new_work(renewal, monkeypatch, rotated):
    r = renewal
    binding = r.service.bindings()[0]
    r.peer.clock = lambda: r.clock[0]
    r.service.runtime.maintain_leased_room = None
    original_request = r.peer._request
    target = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    target._room_grant_secret = lambda: r.secret
    admissions, stops, errors, transports = [], [], [], []
    original_resolver = r.service.runtime.transport_resolver

    def resolve(*args):
        transport = original_resolver(*args)
        transports.append(transport)
        return transport

    def request(path, *, method="GET", body=None, room_grant=None, **kwargs):
        if path.startswith("/v1/runs"):
            permission = "stop" if path.endswith("/stop") else "dispatch" if method == "POST" else "status"
            error = target._check_run_auth(SimpleNamespace(
                headers={"Authorization": f"HermesRoom {room_grant}"}, path=path, method=method,
            ), permission=permission)
            assert error is None
            if permission == "dispatch":
                admissions.append(body["hosted_room_dispatch"])
            if permission == "stop":
                stops.append(room_grant)
            return {"run_id": "disband-peer-run", "status": "cancelled" if stops else "running"}
        return original_request(path, method=method, body=body, room_grant=room_grant, **kwargs)

    monkeypatch.setattr(r.peer, "_request", request)
    r.service.runtime.transport_resolver = resolve
    r.service.send(room_id=binding.room_id, event_id="disband-peer", payload={
        "text": "@ops Work until stopped", "thread_id": "disband-peer-thread",
    })
    (task,) = driver.list_tasks(r.service.db_path, room_id=binding.room_id, status="queued")

    def begin_disband(_timeout=None):
        assert len(admissions) == 1
        if rotated:
            refreshed = r.peer.refresh_grant(grant=r.old)
            r.service._rotate_route_grant(binding.room_id, "ops", refreshed["grant"],
                                          expected_grant_sha256=hashlib.sha256(r.old.encode()).hexdigest())
        hosted_room_link_records.begin_room_link_retirement(
            r.service.db_path, room_id=binding.room_id, authority_gateway_id=binding.gateway_id,
            authority_epoch=binding.authority_epoch)
        try:
            transports[0].history(profile="ops", session_id=transports[0]._session_id, source="bot_room")
        except RuntimeError as exc:
            errors.append(str(exc))
        try:
            r.service.stop_room(binding.room_id, cancel_id="disband-stop", require_acknowledged=True)
        except RuntimeError as exc:
            errors.append(str(exc))
        if errors:
            r.service.runtime._stop.set()
        return False

    monkeypatch.setattr(r.service.runtime._wake, "wait", begin_disband)
    r.service.runtime._run_cycle()
    assert errors == []
    assert driver.get_task(r.service.db_path, task["identity"])["status"] == "cancelled"
    assert len(admissions) == len(stops) == 1
    current = hosted_room_links.load_room_link(r.service.db_path, room_id=binding.room_id, member_id="ops")
    assert stops == [current.grant]
    assert (current.grant != r.old) is rotated
    if rotated:
        assert hosted_rooms.room_grant_is_revoked(r.service.db_path, claims=r.claims)
    refresh_count = len(r.peer.refreshes)
    for operation, kwargs in (("dispatch", {"dispatch": admissions[0]}),
                              ("recover_dispatch", {"dispatch": admissions[0]}), ("probe", {})):
        with pytest.raises(RuntimeError, match="no longer current"):
            getattr(transports[0].client, operation)(grant=r.old, **kwargs)
    with pytest.raises(hosted_rooms.HostedRoomError, match="registration is fenced"):
        r.service.register_peer_route(room_id=binding.room_id, member_id="ops",
                                      route=r.service.peer_routes[(binding.room_id, "ops")], client=r.peer,
                                      target_url=current.target_url, catalog=current.catalog)
    monkeypatch.setattr(r.peer, "_receipt", lambda *_args: None)
    with pytest.raises(PeerRunsHTTPError, match="accepted peer run receipt is unavailable"):
        transports[0].client.recover_dispatch(dispatch=admissions[0], grant=r.old, receipt_only=True)
    assert len(r.peer.refreshes) == refresh_count
    assert len(admissions) == 1


def test_ordinary_stop_without_a_proven_receipt_never_replays_admission(renewal, monkeypatch):
    r = renewal
    binding = r.service.bindings()[0]
    r.service.send(room_id=binding.room_id, event_id="unknown-stop", payload={
        "text": "@ops Stop an uncertain remote task", "thread_id": "unknown-stop-thread",
    })
    (task,) = driver.list_tasks(r.service.db_path, room_id=binding.room_id, status="queued")
    lease = r.service.runtime._ensure_lease(binding)
    driver.start_task(r.service.db_path, task["identity"], lease,
                      expected_cancel_generation=0, clock=lambda: r.clock[0])
    requests = []

    def no_remote_admission(path, **kwargs):
        requests.append(path)
        raise PeerRunsHTTPError("unknown remote result", retryable=True, ambiguous=True)

    monkeypatch.setattr(r.peer, "_request", no_remote_admission)
    assert not hosted_room_link_records.room_link_retirement_started(r.service.db_path, room_id=binding.room_id)
    r.service.stop_room(binding.room_id, cancel_id="ordinary-stop")
    assert driver.get_task(r.service.db_path, task["identity"])["status"] == "stopping"
    assert requests == []
