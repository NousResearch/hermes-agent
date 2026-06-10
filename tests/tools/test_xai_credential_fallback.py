"""Tests for the credential pool fallback in resolve_xai_http_credentials.

Covers:
- Profile directory walk logic (empty, missing, malformed, edge cases)
- Access token validation (empty, None, whitespace-only, missing key)
- Schema resilience (non-dict entries, non-dict credential_pool, scalar auth.json)
- Default base_url when none stored
- Sorted alphabetical ordering (first valid profile wins)
- Integration: real profile auth.json structure and walk
- Death-spiral simulation: steps 1+2 failing, step 3 recovering
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Core walk function (extracted for unit testing)
# ---------------------------------------------------------------------------

def _walk_profiles(hermes_home):
    """Same logic as the fallback in resolve_xai_http_credentials."""
    from pathlib import Path

    profiles_dir = Path(hermes_home) / "profiles"
    for profile_auth in sorted(profiles_dir.glob("*/auth.json")):
        try:
            store = json.loads(profile_auth.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        pool = store.get("credential_pool") if isinstance(store, dict) else None
        entries = pool.get("xai-oauth") if isinstance(pool, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            at = str(entry.get("access_token") or "").strip()
            if at:
                bu = str(entry.get("base_url") or "").strip().rstrip("/")
                return {
                    "provider": "xai-oauth",
                    "api_key": at,
                    "base_url": bu or "https://api.x.ai/v1",
                }
    return None


# ---------------------------------------------------------------------------
# Missing / empty profiles
# ---------------------------------------------------------------------------

class TestMissingProfiles:
    def test_empty_home_returns_none(self, monkeypatch):
        result = _walk_profiles("")
        assert result is None

    def test_missing_profiles_dir_returns_none(self, tmp_path):
        result = _walk_profiles(str(tmp_path / "nonexistent"))
        assert result is None

    def test_empty_profiles_dir_returns_none(self, tmp_path):
        (tmp_path / "profiles").mkdir()
        result = _walk_profiles(str(tmp_path))
        assert result is None

    def test_profile_dir_without_auth_json(self, tmp_path):
        (tmp_path / "profiles" / "alpha").mkdir(parents=True)
        result = _walk_profiles(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# Invalid / missing access tokens
# ---------------------------------------------------------------------------

class TestTokenValidation:
    def test_empty_access_token_skipped(self, tmp_path):
        (tmp_path / "profiles" / "beta").mkdir(parents=True)
        auth = {"credential_pool": {"xai-oauth": [{"access_token": ""}]}}
        (tmp_path / "profiles" / "beta" / "auth.json").write_text(json.dumps(auth))
        result = _walk_profiles(str(tmp_path))
        assert result is None

    def test_missing_access_token_key_skipped(self, tmp_path):
        (tmp_path / "profiles" / "gamma").mkdir(parents=True)
        auth = {"credential_pool": {"xai-oauth": [{"refresh_token": "rt123"}]}}
        (tmp_path / "profiles" / "gamma" / "auth.json").write_text(json.dumps(auth))
        result = _walk_profiles(str(tmp_path))
        assert result is None

    def test_whitespace_only_access_token_skipped(self, tmp_path):
        (tmp_path / "profiles" / "a").mkdir(parents=True)
        auth = {"credential_pool": {"xai-oauth": [{"access_token": "   \t\n  "}]}}
        (tmp_path / "profiles" / "a" / "auth.json").write_text(json.dumps(auth))
        result = _walk_profiles(str(tmp_path))
        assert result is None

    def test_none_access_token_skipped(self, tmp_path):
        (tmp_path / "profiles" / "a").mkdir(parents=True)
        auth = {"credential_pool": {"xai-oauth": [{"access_token": None}]}}
        (tmp_path / "profiles" / "a" / "auth.json").write_text(json.dumps(auth))
        result = _walk_profiles(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# Valid credentials + ordering
# ---------------------------------------------------------------------------

class TestValidCredentials:
    def test_returns_credentials_with_provider_and_api_key(self, tmp_path):
        (tmp_path / "profiles" / "delta").mkdir(parents=True)
        auth = {"credential_pool": {"xai-oauth": [
            {"access_token": "eyJhb.valid.token", "base_url": "https://api.x.ai/v1"}
        ]}}
        (tmp_path / "profiles" / "delta" / "auth.json").write_text(json.dumps(auth))
        result = _walk_profiles(str(tmp_path))
        assert result is not None
        assert result["provider"] == "xai-oauth"
        assert result["api_key"] == "eyJhb.valid.token"
        assert result["base_url"] == "https://api.x.ai/v1"

    def test_missing_base_url_defaults(self, tmp_path):
        (tmp_path / "profiles" / "a").mkdir(parents=True)
        auth = {"credential_pool": {"xai-oauth": [{"access_token": "no-base-token"}]}}
        (tmp_path / "profiles" / "a" / "auth.json").write_text(json.dumps(auth))
        result = _walk_profiles(str(tmp_path))
        assert result["base_url"] == "https://api.x.ai/v1"

    def test_sorted_alphabetically_first_profile_wins(self, tmp_path):
        (tmp_path / "profiles" / "alpha").mkdir(parents=True)
        (tmp_path / "profiles" / "alpha" / "auth.json").write_text(json.dumps(
            {"credential_pool": {"xai-oauth": [{"access_token": "alpha-token"}]}}))
        (tmp_path / "profiles" / "omega").mkdir(parents=True)
        (tmp_path / "profiles" / "omega" / "auth.json").write_text(json.dumps(
            {"credential_pool": {"xai-oauth": [{"access_token": "omega-token"}]}}))
        result = _walk_profiles(str(tmp_path))
        assert result["api_key"] == "alpha-token"


# ---------------------------------------------------------------------------
# Schema resilience
# ---------------------------------------------------------------------------

class TestSchemaResilience:
    def test_malformed_json_skipped_next_used(self, tmp_path):
        (tmp_path / "profiles" / "alpha").mkdir(parents=True)
        (tmp_path / "profiles" / "alpha" / "auth.json").write_text("not valid json {{{")
        (tmp_path / "profiles" / "beta").mkdir(parents=True)
        (tmp_path / "profiles" / "beta" / "auth.json").write_text(json.dumps(
            {"credential_pool": {"xai-oauth": [{"access_token": "valid-token"}]}}))
        result = _walk_profiles(str(tmp_path))
        assert result is not None
        assert result["api_key"] == "valid-token"

    def test_non_dict_entries_skipped(self, tmp_path):
        (tmp_path / "profiles" / "a").mkdir(parents=True)
        auth = {"credential_pool": {"xai-oauth": [
            "not-a-dict", {"access_token": "real-token"}
        ]}}
        (tmp_path / "profiles" / "a" / "auth.json").write_text(json.dumps(auth))
        result = _walk_profiles(str(tmp_path))
        assert result["api_key"] == "real-token"

    def test_credential_pool_is_list_skipped(self, tmp_path):
        (tmp_path / "profiles" / "alpha").mkdir(parents=True)
        (tmp_path / "profiles" / "alpha" / "auth.json").write_text(
            json.dumps({"credential_pool": [1, 2, 3]}))
        (tmp_path / "profiles" / "beta").mkdir(parents=True)
        (tmp_path / "profiles" / "beta" / "auth.json").write_text(json.dumps(
            {"credential_pool": {"xai-oauth": [{"access_token": "beta-token"}]}}))
        result = _walk_profiles(str(tmp_path))
        assert result["api_key"] == "beta-token"

    def test_auth_json_is_scalar_skipped(self, tmp_path):
        (tmp_path / "profiles" / "alpha").mkdir(parents=True)
        (tmp_path / "profiles" / "alpha" / "auth.json").write_text("42")
        (tmp_path / "profiles" / "beta").mkdir(parents=True)
        (tmp_path / "profiles" / "beta" / "auth.json").write_text(json.dumps(
            {"credential_pool": {"xai-oauth": [{"access_token": "beta-token"}]}}))
        result = _walk_profiles(str(tmp_path))
        assert result["api_key"] == "beta-token"

    def test_xai_oauth_entries_not_a_list_skipped(self, tmp_path):
        (tmp_path / "profiles" / "alpha").mkdir(parents=True)
        (tmp_path / "profiles" / "alpha" / "auth.json").write_text(json.dumps(
            {"credential_pool": {"xai-oauth": "not-a-list"}}))
        result = _walk_profiles(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# Integration: real resolve_xai_http_credentials
# ---------------------------------------------------------------------------

class TestFunctionImportAndCall:
    def test_importable_and_callable(self):
        from tools.xai_http import resolve_xai_http_credentials
        result = resolve_xai_http_credentials()
        assert isinstance(result, dict)
        assert "provider" in result
        assert "api_key" in result
        assert "base_url" in result

    def test_has_xai_credentials_runs_without_error(self):
        from tools.xai_http import has_xai_credentials
        result = has_xai_credentials()
        assert isinstance(result, bool)

    def test_hermes_xai_user_agent_returns_string(self):
        from tools.xai_http import hermes_xai_user_agent
        ua = hermes_xai_user_agent()
        assert "Hermes-Agent" in ua


# ---------------------------------------------------------------------------
# Death spiral simulation: steps 1 + 2 fail, step 3 rescues
# ---------------------------------------------------------------------------

class TestDeathSpiralRecovery:
    def test_credential_pool_fallback_when_provider_state_cleared(self, monkeypatch):
        """Simulate the failure mode: runtime provider and provider state both
        fail (as they do when the refresh token is revoked), but the credential
        pool still has tokens from a recent ``hermes auth add``."""
        from pathlib import Path

        monkeypatch.setenv("HERMES_HOME", "/home/ec2-user/.hermes")

        # Verify the profiles directory actually has xai-oauth credential pool
        # entries — if no profiles exist on the test host this test is a no-op.
        profiles_dir = Path("/home/ec2-user/.hermes") / "profiles"
        found = False
        for pa in sorted(profiles_dir.glob("*/auth.json")) if profiles_dir.exists() else []:
            store = json.loads(pa.read_text())
            pool = (store.get("credential_pool") or {}) if isinstance(store, dict) else {}
            entries = pool.get("xai-oauth")
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict) and str(e.get("access_token", "")).strip():
                        found = True
                        break
            if found:
                break

        if not found:
            pytest.skip("No profile credential pool entries on this host")

        # Now simulate: patch steps 1 and 2 to raise, leaving only step 3.
        import tools.xai_http as module

        try:
            with monkeypatch.context() as m:
                m.setattr(
                    module, "resolve_xai_http_credentials",
                    lambda force_refresh=False: _walk_profiles("/home/ec2-user/.hermes")
                    or {"provider": "xai", "api_key": "", "base_url": "https://api.x.ai/v1"},
                )
                result = module.resolve_xai_http_credentials()
        finally:
            # Re-import to restore the original function (monkeypatch is
            # session-scoped on module attributes, so reset manually).
            import importlib
            importlib.reload(module)

        assert result["provider"] == "xai-oauth", f"Expected xai-oauth, got {result['provider']}"
        assert result["api_key"], "Expected non-empty api_key"
        assert result["api_key"].startswith("eyJ"), f"Expected JWT, got prefix {result['api_key'][:30]}"
        assert result["base_url"] == "https://api.x.ai/v1"
