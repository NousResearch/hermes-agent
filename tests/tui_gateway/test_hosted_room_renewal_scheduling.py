"""Renewal shares active polling without preceding healthy work or Stop."""

import hashlib
from dataclasses import replace

import pytest

from gateway import hosted_room_driver as driver, hosted_room_links
from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
from tests.tui_gateway.test_groups_methods import home
from tests.tui_gateway.test_hosted_room_idle_grant_renewal import renewal
from tests.tui_gateway.test_hosted_room_driver_runtime import FakeSessionRPC
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


def local_task(r, *, complete):
    rpc = FakeSessionRPC(auto_complete=complete)
    r.service.rpc = r.service.runtime.rpc = rpc
    r.service.send(room_id="renewal-room", event_id="scheduling-test", payload={
        "text": "@local Complete one local test", "thread_id": "scheduling-thread",
    })
    (task,) = driver.list_tasks(r.service.db_path, room_id="renewal-room", status="queued")
    return task["identity"], rpc


@pytest.mark.parametrize("work", ["long", "queued", "stopping"])
def test_renewal_runs_during_active_work_but_after_stop_and_first_admission(renewal, monkeypatch, work):
    r = renewal
    r.clock[0] = r.claims["expires_at"] - (360 if work == "long" else 300)
    started = r.clock[0]
    identity, rpc = local_task(r, complete=work != "long")
    runtime = r.service.runtime
    binding = r.service.bindings()[0]
    expected = "cancelled" if work == "stopping" else "settled"
    if work == "long":
        def poll(_timeout=None):
            r.clock[0] += 5
            if r.clock[0] - started >= 400:
                rpc.complete(identity.task_id, content="Healthy local result")
            return False
        monkeypatch.setattr(runtime._wake, "wait", poll)
    else:
        if work == "stopping":
            lease = runtime._ensure_lease(binding)
            attempt = driver.start_task(r.service.db_path, identity, lease,
                                        expected_cancel_generation=0, clock=lambda: r.clock[0])
            sid = rpc.add_session(profile="default", title="Group: renewal-room", active=True, task_id=identity.task_id)
            rpc.states[sid]["execution_generation"] = attempt.execution_generation
            driver.begin_task_cancel(r.service.db_path, identity, cancel_id="pending-stop",
                                     expected_cancel_generation=0, clock=lambda: r.clock[0])

        def offline(_path, **_kwargs):
            assert driver.get_task(r.service.db_path, identity)["status"] == expected
            r.clock[0] += r.peer.timeout_seconds
            raise PeerRunsHTTPError("offline", retryable=True, not_admitted=True)
        monkeypatch.setattr(r.peer, "_request", offline)
    runtime._run_cycle()
    assert driver.get_task(r.service.db_path, identity)["status"] == expected
    driver.require_active_lease(r.service.db_path, runtime._leases[identity.room_id], clock=lambda: r.clock[0])
    if work == "long":
        assert r.clock[0] - started >= 400 and r.peer.refreshes
        link = hosted_room_links.load_room_link(r.service.db_path, room_id=identity.room_id, member_id="ops")
        assert link.status == "ready"
        claims = decode_room_grant(r.secret, link.grant, permission="dispatch")
        assert claims["status_expires_at"] == r.claims["status_expires_at"]
    else:
        assert r.clock[0] - started <= 2.001
    assert r.peer.timeout_seconds == 30


def test_room_scans_and_network_stay_throttled_when_horizon_prevents_extension(renewal, monkeypatch):
    r = renewal
    scans = []
    original = hosted_room_links.load_room_links_tolerant

    def scoped_read(db, *, room_id=None):
        scans.append(room_id)
        return original(db, room_id=room_id)
    monkeypatch.setattr(hosted_room_links, "load_room_links_tolerant", scoped_read)
    r.service.runtime._run_cycle()
    assert scans == ["renewal-room"]
    for _ in range(10):
        r.service.runtime._run_cycle()
    assert scans == ["renewal-room"] and not r.peer.refreshes
    hard = r.claims["status_expires_at"]
    r.clock[0] = hard - 180
    scope = {key: r.claims[key] for key in (
        "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id",
        "target_install_id", "target_profile", "execution_policy_digest", "permissions",
    )}
    grant = issue_room_grant(r.secret, grant_id="near-hard-expiry", issued_at=r.clock[0],
                             ttl_seconds=180, status_expires_at=hard, **scope)
    r.service._rotate_route_grant("renewal-room", "ops", grant,
                                 expected_grant_sha256=hashlib.sha256(r.old.encode()).hexdigest())
    for remaining in (180, 120, 60):
        r.clock[0] = hard - remaining
        r.service.runtime._run_cycle()
        count, scan_count = len(r.peer.refreshes), len(scans)
        r.service.runtime._run_cycle()
        assert len(r.peer.refreshes) == count and len(scans) == scan_count
        link = hosted_room_links.load_room_link(r.service.db_path, room_id="renewal-room", member_id="ops")
        claims = decode_room_grant(r.secret, link.grant, permission="dispatch")
        assert claims["expires_at"] == claims["status_expires_at"] == hard
    issued = len(r.peer.issued)
    r.clock[0] = hard
    r.service.runtime._run_cycle()
    link = hosted_room_links.load_room_link(r.service.db_path, room_id="renewal-room", member_id="ops")
    assert link.status == "needs_reauthorization" and len(r.peer.issued) == issued


def test_room_budget_defers_unvisited_routes_without_starving_them(renewal, monkeypatch):
    r = renewal
    route = r.service.peer_routes[("renewal-room", "ops")]
    catalog = hosted_room_links.load_room_link(r.service.db_path, room_id="renewal-room", member_id="ops").catalog
    for member in ("second", "third"):
        r.service.register_peer_route(room_id="renewal-room", member_id=member,
                                      route=replace(route, member_id=member), client=r.peer,
                                      target_url=r.peer.base_url, catalog=catalog)
    r.clock[0] = r.claims["expires_at"] - 300
    calls = []

    def offline(_path, **_kwargs):
        # No target authorization is simulated here: only request time and fair scheduling.
        calls.append(r.clock[0])
        r.clock[0] += r.peer.timeout_seconds
        raise PeerRunsHTTPError("offline", retryable=True, not_admitted=True)

    monkeypatch.setattr(r.peer, "_request", offline)
    start = r.clock[0]
    r.service.runtime._run_cycle()
    assert len(calls) == 2 and r.clock[0] - start <= 2
    assert ("renewal-room", "second") not in r.service._peer_renewals
    assert ("renewal-room", "third") not in r.service._peer_renewals
    for elapsed, member in ((5, "second"), (10, "third")):
        r.clock[0] = start + elapsed
        r.service.runtime._run_cycle()
        assert r.clock[0] - (start + elapsed) <= 2
        assert ("renewal-room", member) in r.service._peer_renewals
    assert len(calls) == 6
    assert ("renewal-room", "third") in r.service._peer_renewals
    driver.require_active_lease(r.service.db_path, r.service.runtime._leases["renewal-room"],
                                clock=lambda: r.clock[0])
