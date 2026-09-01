import pytest

from tests.fakes.mcp_oauth_peer import (
    KnownCredentialLoss,
    OAuthArtifactState,
    OAuthFailurePoint,
    capture_oauth_state,
    raise_known_mutation,
    seed_old_oauth_state,
)


def test_seed_and_capture_old_oauth_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
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
