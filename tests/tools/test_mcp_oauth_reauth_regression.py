import json

import pytest

from tests.fakes.mcp_oauth_peer import (
    FakeOAuthMCPPeer,
    KnownCredentialLoss,
    OAuthArtifactState,
    OAuthFailurePoint,
    capture_oauth_state,
    raise_known_mutation,
    seed_old_oauth_state,
)
from tests.fakes.mcp_oauth_peer import InjectedOAuthFailure
from tools.mcp_dashboard_oauth import DashboardOAuthFlow


def test_seed_and_capture_old_oauth_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    seeded = seed_old_oauth_state(tmp_path, "reports")
    assert capture_oauth_state(tmp_path, "reports") == seeded
    assert seeded.labels() == ("OLD", "OLD", "OLD")
    assert seeded.safe_summary() == "token=OLD client=OLD metadata=OLD"


def test_safe_summary_never_contains_fake_secret_payloads():
    state = OAuthArtifactState(
        token=b'{"access_token":"OLD_ACCESS_TOKEN_FOR_TEST_ONLY"}',
        client=b'{"client_secret":"OLD_CLIENT_SECRET_FOR_TEST_ONLY"}',
        metadata=b'{"token_endpoint":"https://old-auth.invalid/token"}',
    )
    summary = state.safe_summary()
    assert summary == "token=OLD client=OLD metadata=OLD"
    assert "ACCESS_TOKEN" not in summary
    assert "CLIENT_SECRET" not in summary
    assert "old-auth.invalid" not in summary


def test_repr_never_contains_fake_secret_payloads():
    state = OAuthArtifactState(
        token=b'{"access_token":"OLD_ACCESS_TOKEN_FOR_TEST_ONLY"}',
        client=b'{"client_secret":"OLD_CLIENT_SECRET_FOR_TEST_ONLY"}',
        metadata=b'{"token_endpoint":"https://old-auth.invalid/token"}',
    )
    rendered = repr(state)
    assert "OLD_ACCESS_TOKEN_FOR_TEST_ONLY" not in rendered
    assert "OLD_CLIENT_SECRET_FOR_TEST_ONLY" not in rendered
    assert "old-auth.invalid" not in rendered


def test_repr_excludes_every_seeded_legacy_payload_value(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    state = seed_old_oauth_state(tmp_path, "reports")
    rendered = repr(state)

    def payload_values(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from payload_values(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from payload_values(nested)
        else:
            yield str(value)

    for artifact in (state.token, state.client, state.metadata):
        assert artifact is not None
        for payload_value in payload_values(json.loads(artifact)):
            assert payload_value not in rendered


def test_raise_known_mutation_rejects_unknown_corruption_shape():
    before = OAuthArtifactState(b"OLD_TOKEN", b"OLD_CLIENT", b"OLD_META")
    unexpected = OAuthArtifactState(None, None, b"PARTIAL_META")
    with pytest.raises(AssertionError, match="unexpected OAuth artifact state"):
        raise_known_mutation(
            before=before,
            after=unexpected,
            expected_labels=("MISSING", "PARTIAL", "PARTIAL"),
            surface="dashboard",
            failure_point=OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
        )


def test_raise_known_mutation_types_the_exact_known_bug():
    before = OAuthArtifactState(b"OLD_TOKEN", b"OLD_CLIENT", b"OLD_META")
    exact = OAuthArtifactState(None, b"PARTIAL_CLIENT", b"PARTIAL_META")
    with pytest.raises(KnownCredentialLoss, match="surface=dashboard"):
        raise_known_mutation(
            before=before,
            after=exact,
            expected_labels=("MISSING", "PARTIAL", "PARTIAL"),
            surface="dashboard",
            failure_point=OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
        )


@pytest.mark.parametrize("failure_point", list(OAuthFailurePoint))
def test_fake_peer_fails_after_exact_requested_event(tmp_path, monkeypatch, failure_point):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(failure_point)
    with pytest.raises(InjectedOAuthFailure) as caught:
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"}, connect_timeout=315)
    assert caught.value.point is failure_point
    assert peer.completed_events[-1] == failure_point.value
    assert peer.connect_timeouts == [315]


@pytest.mark.parametrize(
    ("failure_point", "expected_labels"),
    [
        (OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY, ("MISSING", "MISSING", "MISSING")),
        (OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY, ("MISSING", "MISSING", "PARTIAL")),
        (OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION, ("MISSING", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.AUTHORIZATION_URL_PUBLICATION, ("MISSING", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.CALLBACK_RECEIPT, ("MISSING", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.TOKEN_EXCHANGE, ("MISSING", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.TOKEN_PERSISTENCE, ("NEW", "PARTIAL", "PARTIAL")),
        (OAuthFailurePoint.MCP_INITIALIZATION, ("NEW", "PARTIAL", "PARTIAL")),
    ],
)
def test_fake_peer_persists_only_completed_stage_effects(tmp_path, monkeypatch, failure_point, expected_labels):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(failure_point)
    with pytest.raises(InjectedOAuthFailure):
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
    assert capture_oauth_state(tmp_path, "reports").labels() == expected_labels


_GH_76590 = pytest.mark.xfail(
    strict=True,
    raises=KnownCredentialLoss,
    reason="GH #76590: failed MCP OAuth reauthorization mutates active credentials",
)

_DASHBOARD_FAILURES = [
    pytest.param(
        OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY,
        None,
        id="before-write-preserves-old",
    ),
    pytest.param(
        OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY,
        ("MISSING", "MISSING", "PARTIAL"),
        marks=_GH_76590,
    ),
    pytest.param(
        OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
        ("MISSING", "PARTIAL", "PARTIAL"),
        marks=_GH_76590,
    ),
    pytest.param(
        OAuthFailurePoint.AUTHORIZATION_URL_PUBLICATION,
        ("MISSING", "PARTIAL", "PARTIAL"),
        marks=_GH_76590,
    ),
    pytest.param(
        OAuthFailurePoint.CALLBACK_RECEIPT,
        ("MISSING", "PARTIAL", "PARTIAL"),
        marks=_GH_76590,
    ),
    pytest.param(
        OAuthFailurePoint.TOKEN_EXCHANGE,
        ("MISSING", "PARTIAL", "PARTIAL"),
        marks=_GH_76590,
    ),
    pytest.param(
        OAuthFailurePoint.TOKEN_PERSISTENCE,
        ("NEW", "PARTIAL", "PARTIAL"),
        marks=_GH_76590,
    ),
    pytest.param(
        OAuthFailurePoint.MCP_INITIALIZATION,
        ("NEW", "PARTIAL", "PARTIAL"),
        marks=_GH_76590,
    ),
]


@pytest.mark.parametrize(("failure_point", "broken_labels"), _DASHBOARD_FAILURES)
def test_dashboard_failed_reauth_preserves_active_state(
    tmp_path, monkeypatch, failure_point, broken_labels
):
    from hermes_cli import mcp_config, web_server
    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    reset_manager_for_tests()
    before = seed_old_oauth_state(tmp_path, "reports")
    peer = FakeOAuthMCPPeer(failure_point)
    flow = DashboardOAuthFlow(
        flow_id=f"dashboard-{failure_point.value}",
        server_name="reports",
        profile=None,
        hermes_home=str(tmp_path),
        redirect_uri="https://dashboard.invalid/api/mcp/oauth/callback/reports",
    )
    monkeypatch.setattr(mcp_config, "_probe_single_server", peer.probe)
    monkeypatch.setattr(mcp_config, "_save_mcp_server", lambda *_args: True)

    web_server._run_dashboard_mcp_oauth(
        flow, {"url": "https://mcp.invalid/mcp", "auth": "oauth"}
    )

    assert peer.connect_timeouts == [315]
    assert flow.status == "error"
    assert flow.worker_done is True
    assert failure_point.value in (flow.error or "")
    assert "ACCESS_TOKEN_FOR_TEST_ONLY" not in (flow.error or "")
    after = capture_oauth_state(tmp_path, "reports")
    if broken_labels is None:
        assert after == before
    else:
        raise_known_mutation(
            before=before,
            after=after,
            expected_labels=broken_labels,
            surface="dashboard",
            failure_point=failure_point,
        )


from tests.fakes.mcp_oauth_peer import OAuthFailureKind, ProbeOutcome

_PRE_TOKEN_KINDED = [
    OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY,
    OAuthFailurePoint.AUTHORIZATION_SERVER_DISCOVERY,
    OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
    OAuthFailurePoint.TOKEN_EXCHANGE,
]


@pytest.mark.parametrize("failure_point", _PRE_TOKEN_KINDED)
def test_pre_token_failure_carries_default_definitive_kind(tmp_path, monkeypatch, failure_point):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(failure_point)
    with pytest.raises(InjectedOAuthFailure) as caught:
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
    assert caught.value.kind is OAuthFailureKind.DEFINITIVE
    assert caught.value.retry_after is None


@pytest.mark.parametrize("failure_point", _PRE_TOKEN_KINDED)
def test_pre_token_failure_reports_indeterminate_kind_and_retry_after(tmp_path, monkeypatch, failure_point):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(failure_point, kind=OAuthFailureKind.INDETERMINATE)
    with pytest.raises(InjectedOAuthFailure) as caught:
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
    assert caught.value.kind is OAuthFailureKind.INDETERMINATE
    assert isinstance(caught.value.retry_after, (int, float))


def test_authenticated_probe_returns_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(None)
    assert peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"}) == [("fake_tool", "Deterministic fake MCP tool")]


@pytest.mark.parametrize("outcome", [ProbeOutcome.REJECTED, ProbeOutcome.INDETERMINATE])
def test_probe_point_reports_requested_failing_outcome(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    peer = FakeOAuthMCPPeer(OAuthFailurePoint.MCP_INITIALIZATION, probe_outcome=outcome)
    with pytest.raises(InjectedOAuthFailure) as caught:
        peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
    assert caught.value.probe_outcome is outcome


def test_publication_and_callback_points_take_no_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    for point in (OAuthFailurePoint.AUTHORIZATION_URL_PUBLICATION, OAuthFailurePoint.CALLBACK_RECEIPT):
        peer = FakeOAuthMCPPeer(point)
        with pytest.raises(InjectedOAuthFailure) as caught:
            peer.probe("reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"})
        assert caught.value.kind is None
