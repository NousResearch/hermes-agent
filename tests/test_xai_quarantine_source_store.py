"""Tests for xAI OAuth quarantine persistence cross-profile."""

import json
from pathlib import Path

import pytest

from hermes_cli.auth import AuthError, resolve_xai_oauth_runtime_credentials


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Set up a global root + an active profile under Path.home()/.hermes/profiles/coder.

    * Path.home() -> tmp_path
    * Global root -> tmp_path/.hermes
    * Profile     -> tmp_path/.hermes/profiles/coder (active, HERMES_HOME points here)
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    global_root = tmp_path / ".hermes"
    global_root.mkdir()
    profile_dir = global_root / "profiles" / "coder"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    
    return {"global": global_root, "profile": profile_dir}


def _make_auth_store(providers: dict | None = None) -> dict:
    store: dict = {"version": 1}
    if providers is not None:
        store["providers"] = providers
    return store


def test_terminal_refresh_failure_quarantines_to_source_store(profile_env, monkeypatch):
    """
    Test that a terminal refresh failure quarantines the grant in the store
    it was read from (the global root), rather than writing a shadowing
    stub into the active profile store.
    """
    # 1. Profile store is empty + root store holds the grant
    (profile_env["global"] / "auth.json").write_text(
        json.dumps(_make_auth_store(providers={
            "xai-oauth": {
                "auth_mode": "oauth_pkce",
                "tokens": {
                    "access_token": "expired_acc",
                    "refresh_token": "valid_ref",
                    "expires_in": 3600,
                },
                "discovery": {"token_endpoint": "http://mock/token"},
                "last_refresh": "2026-05-14T00:00:00Z",
            }
        }), indent=2)
    )
    (profile_env["profile"] / "auth.json").write_text(
        json.dumps(_make_auth_store(), indent=2)
    )

    # 2. Force a terminal refresh failure
    def _mock_refresh(*args, **kwargs):
        raise AuthError(
            provider="xai-oauth",
            code="xai_refresh_failed",
            message="invalid_grant: token revoked",
            relogin_required=True,
        )

    monkeypatch.setattr("hermes_cli.auth._refresh_xai_oauth_tokens", _mock_refresh)

    # force_refresh=True bypasses the expiry check to force a network refresh call
    with pytest.raises(AuthError) as exc_info:
        resolve_xai_oauth_runtime_credentials(force_refresh=True)

    assert exc_info.value.code == "xai_refresh_failed"

    # 3. Assert quarantine landed in ROOT store
    root_store = json.loads((profile_env["global"] / "auth.json").read_text())
    root_state = root_store.get("providers", {}).get("xai-oauth", {})
    assert "access_token" not in root_state.get("tokens", {})
    assert "refresh_token" not in root_state.get("tokens", {})
    assert root_state.get("last_auth_error", {}).get("reason") == "runtime_refresh_failure"
    assert root_state.get("last_auth_error", {}).get("relogin_required") is True

    # 4. Assert NO providers.xai-oauth stub exists in the profile store
    profile_store = json.loads((profile_env["profile"] / "auth.json").read_text())
    assert "xai-oauth" not in profile_store.get("providers", {})
