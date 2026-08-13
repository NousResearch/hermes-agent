import pytest


def _set_xai_oauth_unavailable(monkeypatch):
    from hermes_cli import auth

    monkeypatch.setattr(auth, "resolve_xai_oauth_runtime_credentials", lambda **_: {})


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
