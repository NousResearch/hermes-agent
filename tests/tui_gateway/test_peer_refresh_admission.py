"""Fresh preflight failures are not admissions; recovery preserves uncertainty."""

import hashlib
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    HostedMemberDispatch,
    PROTOCOL_VERSION,
    catalog_mapping,
    issue_room_grant,
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_status import _RouteStatusPeerClient


@pytest.fixture
def case():
    now = time.time()
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(installation_id="install-peer", persistent_process=True)
    )
    scope = dict(
        room_id="room-1",
        home_install_id="install-home",
        authority_gateway_id="install-home",
        authority_epoch=1,
        member_id="member-peer",
        target_install_id="install-peer",
        target_profile="reviewer",
        execution_policy_digest=catalog.execution_policy.policy_digest,
    )
    old = issue_room_grant(
        b"s" * 32,
        grant_id="old",
        issued_at=now - 3700,
        ttl_seconds=3600,
        status_expires_at=now + 10000,
        **scope,
    )
    new = issue_room_grant(
        b"s" * 32,
        grant_id="new",
        issued_at=now,
        ttl_seconds=3600,
        status_expires_at=now + 10000,
        **scope,
    )
    prompt = "Review the shared workshop plan"
    dispatch = HostedMemberDispatch(
        protocol_version=PROTOCOL_VERSION,
        task_id="task-1",
        execution_generation=1,
        source_event_seq=1,
        cancellation_scope_id="cancel-room-1",
        prompt=prompt,
        prompt_digest=hashlib.sha256(prompt.encode()).hexdigest(),
        capability_digest=catalog.catalog_digest,
        trace_id="trace-1",
        **scope,
    )
    return SimpleNamespace(old=old, new=new, dispatch=dispatch.as_mapping())


class Client:
    def __init__(self, *, failure=None, refreshed=None, admission_failure=None):
        self.failure, self.refreshed = failure, refreshed
        self.admission_failure = admission_failure
        self.calls, self.revoked = [], []

    def refresh_grant(self, **_kwargs):
        if self.failure:
            raise self.failure
        return {"grant": self.refreshed}

    def dispatch(self, **_kwargs):
        self.calls.append("dispatch")
        if self.admission_failure:
            raise self.admission_failure
        return {"status": "accepted"}

    def recover_dispatch(self, **_kwargs):
        self.calls.append("recover_dispatch")
        return {"status": "accepted"}

    def revoke_grant_exact(self, *, grant):
        self.revoked.append(grant)


def tracked(client, refreshed=lambda *_args: None):
    states = []
    result = _RouteStatusPeerClient(
        client,
        on_ready=lambda: states.append("ready"),
        on_reauthorization=lambda: states.append("reauthorize"),
        on_unavailable=lambda: states.append("unavailable"),
        on_refreshed=refreshed,
    )
    return result, states


@pytest.mark.parametrize("method", ["dispatch", "recover_dispatch"])
@pytest.mark.parametrize("kind", ["auth", "network", "missing-grant", "persistence"])
def test_preflight_failure_preserves_the_admission_phase(case, method, kind):
    failure = (
        PeerRunsHTTPError(
            "expired grant", status_code=401, error_code="invalid_room_grant"
        )
        if kind == "auth"
        else RuntimeError("refresh unavailable")
        if kind == "network"
        else None
    )
    peer = Client(
        failure=failure, refreshed="" if kind == "missing-grant" else case.new
    )
    persist_error = hosted_rooms.HostedRoomError("registration is fenced")

    def persist(*_args):
        if kind == "persistence":
            raise persist_error

    wrapper, states = tracked(peer, persist)
    with pytest.raises(Exception) as caught:
        getattr(wrapper, method)(dispatch=case.dispatch, grant=case.old)
    error = caught.value
    assert error.not_admitted is False
    assert error.dispatch_not_attempted is (method == "dispatch")
    if method == "recover_dispatch":
        assert error.ambiguous is True
    assert peer.calls == []
    if failure:
        assert error.__cause__ is failure and type(error) is type(failure)
    if kind == "auth":
        assert error.needs_reauthorization and states == ["reauthorize"]
    if kind == "persistence":
        assert error.__cause__ is persist_error and peer.revoked == [case.new]


@pytest.mark.parametrize("ambiguous,not_admitted", [(True, False), (False, True)])
def test_actual_admission_failures_keep_their_original_classification(
    case, ambiguous, not_admitted
):
    failure = PeerRunsHTTPError(
        "admission response",
        ambiguous=ambiguous,
        not_admitted=not_admitted,
        status_code=401,
        error_code="invalid_room_grant",
    )
    peer = Client(refreshed=case.new, admission_failure=failure)
    wrapper, _ = tracked(peer)
    with pytest.raises(PeerRunsHTTPError) as caught:
        wrapper.dispatch(dispatch=case.dispatch, grant=case.old)
    assert caught.value is failure
    assert failure.ambiguous is ambiguous and failure.not_admitted is not_admitted
    assert peer.calls == ["dispatch"]


def test_post_admission_status_callback_failure_is_not_marked_non_admitted(case):
    peer = Client(refreshed=case.new)
    wrapper, _ = tracked(peer)
    failure = RuntimeError("status receipt could not be saved")

    def fail_ready():
        raise failure

    wrapper._on_ready = fail_ready
    with pytest.raises(RuntimeError) as caught:
        wrapper.dispatch(dispatch=case.dispatch, grant=case.old)
    assert caught.value is failure
    assert not getattr(failure, "not_admitted", False)
    assert peer.calls == ["dispatch"]


def test_preflight_does_not_mutate_an_exception_held_by_another_call(case):
    prior = PeerRunsHTTPError("prior uncertain request", ambiguous=True)
    before = dict(prior.__dict__)
    peer = Client(failure=prior)
    wrapper, _ = tracked(peer)
    with pytest.raises(PeerRunsHTTPError) as caught:
        wrapper.dispatch(dispatch=case.dispatch, grant=case.old)
    assert caught.value is not prior and caught.value.__cause__ is prior
    assert caught.value.dispatch_not_attempted is True
    assert prior.__dict__ == before


@pytest.mark.parametrize("method", ["dispatch", "recover_dispatch"])
def test_read_only_exception_traits_use_a_controlled_phase_error(case, method):
    class FixedError(RuntimeError):
        status_code = 401
        error_code = "invalid_room_grant"
        needs_reauthorization = True

        @property
        def not_admitted(self):
            return False

    original = FixedError("refresh refused")
    peer = Client(failure=original)
    wrapper, _ = tracked(peer)
    with pytest.raises(PeerRunsHTTPError) as caught:
        getattr(wrapper, method)(dispatch=case.dispatch, grant=case.old)
    assert caught.value.__cause__ is original
    assert caught.value.not_admitted is False
    assert caught.value.dispatch_not_attempted is (method == "dispatch")
    assert caught.value.ambiguous is (method == "recover_dispatch")
    assert caught.value.needs_reauthorization
    assert peer.calls == []
