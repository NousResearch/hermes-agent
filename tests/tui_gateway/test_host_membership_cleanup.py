"""Lost private release cannot keep child subscriptions alive or consume capacity."""
from __future__ import annotations

import pytest

from tui_gateway import server
from tests.tui_gateway.test_conditional_activation import Peer, activate
from tests.tui_gateway import test_host_conditional as hosts


@pytest.fixture
def hosted(monkeypatch, tmp_path):
    yield from hosts.hosted.__wrapped__(monkeypatch, tmp_path)


@pytest.mark.parametrize("loss", ["full", "dropped", "unavailable"])
def test_lost_release_expires_child_liveness_and_recovers_subscription(hosted, monkeypatch, loss):
    _creator, binding, record, supervisor, _done = hosted
    hosts.control(supervisor, binding, "membership_freeze")
    peer = Peer()
    assert "result" in activate(peer, binding)
    assert hosts.control(supervisor, binding, "inspect")["members"] == 1
    send = supervisor._conditional_send_to_current

    def lose_cleanup(boot, payload, waiter=None):
        if payload.get("action") in {"release", "membership"}:
            if loss != "dropped":
                raise RuntimeError("conditional writer " + loss)
            return None
        return send(boot, payload, waiter)

    with monkeypatch.context() as patch:
        patch.setattr(supervisor, "_conditional_send_to_current", lose_cleanup)
        server._detach_session_transport(record, peer)
        assert peer not in record["host_bound_subscribers"]
        # The private release really did not reach the child.
        assert hosts.control(supervisor, binding, "inspect")["members"] == 1
        expired = hosts.control(supervisor, binding, "membership_expire")
        assert expired["members"] == 0
        assert expired["logical_live"] == 0
    recovered = Peer()
    assert "result" in activate(recovered, binding)
    assert hosts.control(supervisor, binding, "inspect")["members"] == 1
    server._detach_session_transport(record, recovered)


def test_expired_lease_and_delayed_confirmation_cannot_retain_liveness_or_capacity(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace

    from tui_gateway import host_conditional_membership as leases
    from tui_gateway.host_conditional import HostConditionalProtocol
    from tests.tui_gateway.test_conditional_activation import setup

    _creator, binding, record = setup(monkeypatch, tmp_path)
    clock = [100.0]
    monkeypatch.setattr(leases, "monotonic", lambda: clock[0])
    frames = []
    host = SimpleNamespace(_closed=threading.Event(), _boot_id="original", emit=frames.append)
    protocol = HostConditionalProtocol(host)
    protocol._membership.stop.set()
    protocol._origins[binding["session_id"]] = record
    peers = [leases.HostBoundPeer(binding["authenticated_owner"]) for _ in range(64)]
    try:
        for index, peer in enumerate(peers):
            server._attach_session_transport(record, peer)
            record.setdefault("bound_subscribers", {})[peer] = dict(binding)
            protocol._members[str(index)] = (peer, record)
        params = {"session_id": binding["session_id"], "expected_binding": binding}
        assert protocol._resolve(server, {"boot_id": "original", "params": params}) is None
        protocol._membership.poll()
        first = frames[-1]
        protocol.handle({"boot_id": "original", "action": "membership",
                         "challenge": first["challenge"], "members": list(protocol._members)})
        assert all(peer.confirmed for peer in peers)
        clock[0] = 110.0
        protocol._membership.poll()
        delayed = frames[-1]
        clock[0] = 126.0
        # Expiry removes liveness before any cleanup/transport-lock acquisition.
        assert all(not peer.write({}) for peer in peers)
        assert not any(peer in server._session_live_transports(record) for peer in peers)
        protocol.handle({"boot_id": "original", "action": "membership",
                         "challenge": first["challenge"], "members": list(protocol._members)})
        protocol.handle({"boot_id": "original", "action": "membership",
                         "challenge": delayed["challenge"], "members": list(protocol._members)})
        assert not protocol._members
        assert protocol._resolve(server, {"boot_id": "original", "params": params}) is not None
        protocol._executor.shutdown(wait=True)
        assert all(peer not in record["bound_subscribers"] for peer in peers)
    finally:
        protocol.close()


@pytest.mark.parametrize("boundary", ["expiry", "close"])
def test_paused_open_renewal_cannot_resurrect_an_observed_revocation(monkeypatch, boundary):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tui_gateway import host_conditional_membership as leases

    clock = [100.0]
    monkeypatch.setattr(leases, "monotonic", lambda: clock[0])
    peer = leases.HostBoundPeer("test:owner")
    observed_open, resume = threading.Event(), threading.Event()
    renewal_thread = []
    original_closed = leases.HostBoundPeer._closed.fget

    def paused_closed(current):
        closed = original_closed(current)
        if threading.current_thread() in renewal_thread and not closed:
            # Pause after the open observation has released its leaf lock.
            observed_open.set()
            assert resume.wait(5)
        return closed

    monkeypatch.setattr(leases.HostBoundPeer, "_closed", property(paused_closed))

    def renew():
        renewal_thread.append(threading.current_thread())
        peer.renew(130.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(renew)
        try:
            assert observed_open.wait(5)
            if boundary == "expiry":
                clock[0] = 116.0
            else:
                peer.close()
            assert peer._closed
            assert not peer.write({})
        finally:
            resume.set()
        future.result(timeout=5)
    assert peer._closed
    assert not peer.write({})
    assert not peer.confirmed
    peer.renew(200.0)
    assert peer._closed
    assert not peer.write({})
    assert not peer.confirmed
