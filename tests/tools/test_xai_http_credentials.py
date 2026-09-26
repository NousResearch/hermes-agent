import pytest


def _set_xai_oauth_unavailable(monkeypatch):
    from hermes_cli import auth
    import hermes_cli.auth_xai as auth_xai

    monkeypatch.setattr(auth, "resolve_xai_oauth_runtime_credentials", lambda **_: {})
    monkeypatch.setattr(auth_xai, "resolve_xai_oauth_runtime_credentials", lambda **_: {})


def test_xai_credentials_fail_closed_without_profile_scope(tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_cli.config import invalidate_env_cache
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "foreign-xai-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://foreign.example/v1")
    _set_xai_oauth_unavailable(monkeypatch)
    invalidate_env_cache()
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            resolve_xai_http_credentials(force_refresh=True)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        invalidate_env_cache()


def test_xai_credentials_do_not_fall_back_to_environ_when_scope_has_no_key(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from hermes_cli.config import invalidate_env_cache
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("XAI_API_KEY", "foreign-xai-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://foreign.example/v1")
    _set_xai_oauth_unavailable(monkeypatch)
    invalidate_env_cache()
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope({})
    secret_scope.set_multiplex_active(True)
    try:
        credentials = resolve_xai_http_credentials(force_refresh=True)
        assert credentials == {
            "provider": "xai",
            "api_key": "",
            "base_url": "https://api.x.ai/v1",
        }
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        invalidate_env_cache()


# ---------------------------------------------------------------------------
# prefer_api_key opt-in (#88040 x_search / #87045 TTS): explicit API key wins
# over subscription OAuth, via the shared resolver — not per-caller inlining.
# ---------------------------------------------------------------------------


def _install_fake_oauth_pool(monkeypatch, oauth_token: str) -> None:
    from types import SimpleNamespace

    entry = SimpleNamespace(
        access_token=oauth_token,
        runtime_api_key=None,
        runtime_base_url=None,
        base_url="https://api.x.ai/v1",
    )

    class _FakePool:
        def select(self):
            return entry

        def try_refresh_matching(self, _hint):
            return entry

    def _fake_load_pool(provider_id):
        if provider_id == "xai-oauth":
            return _FakePool()
        raise KeyError(provider_id)

    monkeypatch.setattr("agent.credential_pool.load_pool", _fake_load_pool)


def test_prefer_api_key_wins_over_available_oauth(monkeypatch):
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name, default=None: {"XAI_API_KEY": "paid-key-x1"}.get(name, default),
    )
    _install_fake_oauth_pool(monkeypatch, "oauth-token-x1")

    creds = resolve_xai_http_credentials(prefer_api_key=True)
    assert creds == {
        "provider": "xai",
        "api_key": "paid-key-x1",
        "base_url": "https://api.x.ai/v1",
    }

    # Default order is unchanged: OAuth still wins without the opt-in.
    creds = resolve_xai_http_credentials()
    assert creds["provider"] == "xai-oauth"
    assert creds["api_key"] == "oauth-token-x1"


def test_prefer_api_key_falls_back_to_oauth_without_explicit_key(monkeypatch):
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value", lambda name, default=None: default
    )
    _install_fake_oauth_pool(monkeypatch, "oauth-token-x1")

    creds = resolve_xai_http_credentials(prefer_api_key=True)
    assert creds["provider"] == "xai-oauth"
    assert creds["api_key"] == "oauth-token-x1"


def test_prefer_api_key_honors_hermes_xai_base_url_with_validation(monkeypatch):
    """The preferred-key path reads the same override pair as the OAuth
    branch (HERMES_XAI_BASE_URL first, then XAI_BASE_URL) behind the same
    origin-pinning validation: an *.x.ai override is honored, a foreign
    origin is rejected in favor of the default."""
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name, default=None: {
            "XAI_API_KEY": "paid-key-x1",
            "HERMES_XAI_BASE_URL": "https://staging.x.ai/v1",
            "XAI_BASE_URL": "https://ignored.x.ai/v1",
        }.get(name, default),
    )
    _install_fake_oauth_pool(monkeypatch, "oauth-token-x1")

    creds = resolve_xai_http_credentials(prefer_api_key=True)
    assert creds["base_url"] == "https://staging.x.ai/v1"

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name, default=None: {
            "XAI_API_KEY": "paid-key-x1",
            "XAI_BASE_URL": "https://attacker.example/v1",
        }.get(name, default),
    )
    creds = resolve_xai_http_credentials(prefer_api_key=True)
    assert creds["base_url"] == "https://api.x.ai/v1"


def test_prefer_api_key_honors_profile_scope_only_key(tmp_path, monkeypatch):
    """A key present only in the active profile's secret scope (not in
    os.environ / .env) is honored on the preferred path — the read goes
    through resolve_provider_secret, not a raw env lookup."""
    from agent import secret_scope
    from hermes_cli.config import invalidate_env_cache
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    invalidate_env_cache()
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope({"XAI_API_KEY": "scoped-key-x1"})
    secret_scope.set_multiplex_active(True)
    try:
        creds = resolve_xai_http_credentials(prefer_api_key=True)
        assert creds["provider"] == "xai"
        assert creds["api_key"] == "scoped-key-x1"
        assert creds["base_url"] == "https://api.x.ai/v1"
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        invalidate_env_cache()


# ---------------------------------------------------------------------------
# #116155: `hermes auth remove xai` suppresses env:XAI_API_KEY, but a stale value
# still inherited by the gateway process environment overrode the working
# xai-oauth grant for x_search (prefer_api_key=True).
# ---------------------------------------------------------------------------


def _write_auth_store(tmp_path, monkeypatch, suppressed):
    import json
    import time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"xai-oauth": {"tokens": {
            "access_token": "oauth-live-token", "refresh_token": "r",
            "expires_at": int(time.time() * 1000) + 3_600_000}}},
        "credential_pool": {},
        "suppressed_sources": suppressed,
    }), encoding="utf-8")


@pytest.mark.parametrize("suppressed, expected_provider", [
    ({"xai": ["env:XAI_API_KEY"]}, "xai-oauth"),   # removed source is ignored -> OAuth serves
    ({}, "xai"),                                    # control: the env key still wins when not removed
])
def test_suppressed_env_key_does_not_override_oauth(tmp_path, monkeypatch, suppressed, expected_provider):
    from hermes_cli.config import invalidate_env_cache
    from tools.xai_http import resolve_xai_http_credentials

    _write_auth_store(tmp_path, monkeypatch, suppressed)
    monkeypatch.setenv("XAI_API_KEY", "dead-creditless-team-key")
    invalidate_env_cache()

    creds = resolve_xai_http_credentials(prefer_api_key=True)
    assert creds["provider"] == expected_provider
    assert (creds["api_key"] == "dead-creditless-team-key") is (expected_provider == "xai")


def test_suppression_is_scoped_to_its_provider(tmp_path, monkeypatch):
    """Suppressing xai's env source must not blind another provider's env read."""
    from tools.tool_backend_helpers import resolve_provider_secret

    _write_auth_store(tmp_path, monkeypatch, {"xai": ["env:XAI_API_KEY"]})
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    monkeypatch.setenv("XAI_API_KEY", "dead-key")
    assert resolve_provider_secret("ELEVENLABS_API_KEY", "elevenlabs") == "el-key"
    assert resolve_provider_secret("XAI_API_KEY", "xai") == ""


def test_api_key_fallback_branch_pins_base_url_origin(monkeypatch):
    """The final API-key fallback branch (no OAuth pool) must origin-pin the
    env override like the prefer_api_key and OAuth branches: a tampered
    XAI_BASE_URL / HERMES_XAI_BASE_URL (foreign host or non-HTTPS) is rejected
    in favor of the default so the credential is never sent elsewhere."""
    from tools.xai_http import resolve_xai_http_credentials

    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name, default=None: {
            "XAI_API_KEY": "paid-key-x1",
            "XAI_BASE_URL": "http://attacker.example/v1",
        }.get(name, default),
    )
    # No usable OAuth pool: force the fallback branch.
    import agent.credential_pool as credential_pool

    class _DeadPool:
        def select(self):
            raise RuntimeError("no pool")

        def try_refresh_matching(self, _hint):
            raise RuntimeError("no pool")

    monkeypatch.setattr(
        credential_pool, "load_pool", lambda provider_id: _DeadPool()
    )

    creds = resolve_xai_http_credentials()
    assert creds["provider"] == "xai"
    assert creds["api_key"] == "paid-key-x1"
    assert creds["base_url"] == "https://api.x.ai/v1"

    # A legitimate *.x.ai HTTPS override is still honored.
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value",
        lambda name, default=None: {
            "XAI_API_KEY": "paid-key-x1",
            "XAI_BASE_URL": "https://staging.x.ai/v1",
        }.get(name, default),
    )
    creds = resolve_xai_http_credentials()
    assert creds["base_url"] == "https://staging.x.ai/v1"
