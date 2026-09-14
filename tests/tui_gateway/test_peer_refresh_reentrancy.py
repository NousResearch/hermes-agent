"""Adversarial phase checks with real clients and no HTTP transport."""

import pytest

from tests.tui_gateway.test_peer_refresh_admission import case
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_status import _RouteStatusPeerClient


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(
        "socket.socket.connect",
        lambda *_a, **_k: pytest.fail("network is forbidden"),
    )


@pytest.mark.parametrize("prior", ["accepted", "ambiguous"])
@pytest.mark.parametrize("next_method", ["dispatch", "recover_dispatch"])
def test_refresh_refusal_cannot_erase_prior_same_identity_admission(
    case, monkeypatch, prior, next_method
):
    peer = PeerRunsHTTPClient(
        base_url="https://peer.invalid", api_key="", target_profile="reviewer"
    )
    needs_refresh = [False]
    monkeypatch.setattr(
        "gateway.hosted_room_peer.room_grant_needs_dispatch_refresh",
        lambda *_args, **_kwargs: needs_refresh[0],
    )
    calls = []
    refresh_error = PeerRunsHTTPError(
        "refresh refused", status_code=401, error_code="invalid_room_grant"
    )

    def request(path, **kwargs):
        calls.append((path, kwargs))
        if path == "/v1/room-members/grants/refresh":
            raise refresh_error
        assert path == "/v1/runs"
        if prior == "ambiguous":
            raise PeerRunsHTTPError("reply lost", ambiguous=True)
        return {"run_id": "run-known-accepted"}

    monkeypatch.setattr(peer, "_request", request)
    wrapper = _RouteStatusPeerClient(
        peer,
        grant=case.new,
        on_ready=lambda **_kw: None,
        on_reauthorization=lambda **_kw: None,
        on_unavailable=lambda **_kw: None,
        on_refreshed=lambda *_args, **_kw: None,
    )
    if prior == "accepted":
        result = wrapper.dispatch(dispatch=case.dispatch, grant=case.new)
        assert result["run_id"] == "run-known-accepted"
        assert peer._runs[("task-1", 1)]["run_id"] == "run-known-accepted"
    else:
        with pytest.raises(PeerRunsHTTPError) as first:
            wrapper.dispatch(dispatch=case.dispatch, grant=case.new)
        assert first.value.ambiguous and not first.value.not_admitted
        assert len(calls) == 2
        assert calls[0][1]["headers"] == calls[1][1]["headers"]
        assert calls[0][1]["body"] == calls[1][1]["body"]

    attempted_posts = len(calls)
    needs_refresh[0] = True
    with pytest.raises(PeerRunsHTTPError) as second:
        getattr(wrapper, next_method)(dispatch=case.dispatch, grant=case.new)
    assert second.value.__cause__ is refresh_error
    assert not hasattr(refresh_error, "dispatch_not_attempted")
    assert second.value.needs_reauthorization
    assert len(calls) == attempted_posts + 1
    assert calls[-1][0] == "/v1/room-members/grants/refresh"
    assert not second.value.not_admitted, (
        prior,
        next_method,
        second.value.not_admitted,
        second.value.ambiguous,
    )
    if next_method == "recover_dispatch":
        assert second.value.ambiguous


def test_different_task_is_still_a_fresh_preflight(case, monkeypatch):
    peer = PeerRunsHTTPClient(
        base_url="https://peer.invalid", api_key="", target_profile="reviewer"
    )
    needs_refresh = [False]
    monkeypatch.setattr(
        "gateway.hosted_room_peer.room_grant_needs_dispatch_refresh",
        lambda *_a, **_k: needs_refresh[0],
    )

    def request(path, **_kwargs):
        if path == "/v1/room-members/grants/refresh":
            raise PeerRunsHTTPError(
                "refresh refused", status_code=401, error_code="invalid_room_grant"
            )
        assert path == "/v1/runs"
        return {"run_id": "first-task"}

    monkeypatch.setattr(peer, "_request", request)
    wrapper = _RouteStatusPeerClient(
        peer,
        grant=case.new,
        on_ready=lambda **_k: None,
        on_reauthorization=lambda **_k: None,
        on_unavailable=lambda **_k: None,
        on_refreshed=lambda *_a, **_k: None,
    )
    wrapper.dispatch(dispatch=case.dispatch, grant=case.new)
    needs_refresh[0] = True
    with pytest.raises(PeerRunsHTTPError) as caught:
        wrapper.dispatch(
            dispatch={**case.dispatch, "task_id": "task-2"}, grant=case.new
        )
    assert not caught.value.not_admitted
    assert caught.value.dispatch_not_attempted is True
