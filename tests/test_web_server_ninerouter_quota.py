from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone


def test_profile_settings_use_explicit_profile_home(monkeypatch, tmp_path):
    import hermes_constants
    import hermes_cli.web_server as web_server
    import agent.secret_scope as secret_scope

    profile_home = tmp_path / "profile-alpha"
    profile_home.mkdir()
    observed = []

    def fake_scope(home):
        observed.append(home)
        return {
            "HERMES_9ROUTER_MANAGEMENT_URL": "http://127.0.0.2:20128",
            "HERMES_9ROUTER_AUTH_COOKIE": "[REDACTED]",
            "HERMES_9ROUTER_DATA_DIR": str(profile_home / "router-data"),
        }

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path / "wrong-home")
    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", fake_scope)

    settings = web_server._ninerouter_profile_settings(
        "profile-alpha",
        profile_home=profile_home,
    )

    assert observed == [profile_home]
    assert settings["management_base_url"] == "http://127.0.0.2:20128"
    assert settings["auth_cookie"] == "[REDACTED]"
    assert settings["data_dir"] == str(profile_home / "router-data")
    assert settings["allow_default_data_dir"] is False


def test_usage_quota_serializes_route_breakdown_without_aggregate(monkeypatch, tmp_path):
    import hermes_cli.web_server as web_server
    from agent.account_usage import AccountUsageRoute, AccountUsageSnapshot
    import agent.ninerouter_usage as ninerouter_usage
    import agent.secret_scope as secret_scope
    import hermes_cli.runtime_provider as runtime_provider

    captured = {}

    @contextmanager
    def profile_scope(profile):
        captured["profile"] = profile
        yield tmp_path

    snapshot = AccountUsageSnapshot(
        provider="9router",
        source="9router_management_api",
        fetched_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
        title="9Router usage & quota",
        routes=(
            AccountUsageRoute(
                route="daily",
                provider="route-provider",
                account="quota-account",
                usage=12.0,
                limit=100.0,
                remaining=88.0,
                remaining_percent=88.0,
                unit="requests",
                reset_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
                status="reported",
                source="9router_management_api",
            ),
        ),
        scope="profile:quota-test",
    )

    def fake_fetch(**kwargs):
        captured["fetch"] = kwargs
        return snapshot

    monkeypatch.setattr(web_server, "_config_profile_scope", profile_scope)
    monkeypatch.setattr(web_server, "load_config", lambda: {"model": {"provider": "openai-api"}})
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda requested: {"base_url": "http://localhost:20128/v1", "api_key": "[REDACTED]"},
    )
    def fake_resolve(*args, **kwargs):
        captured["token_kwargs"] = kwargs
        return "[REDACTED]"

    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", lambda home: {})
    monkeypatch.setattr(ninerouter_usage, "fetch_ninerouter_account_usage", fake_fetch)
    monkeypatch.setattr(ninerouter_usage, "resolve_ninerouter_cli_token", fake_resolve)

    result = web_server._get_usage_quota("quota-test")

    assert captured["profile"] == "quota-test"
    assert captured["token_kwargs"] == {
        "data_dir": None,
        "use_process_env": False,
        "allow_default_data_dir": False,
    }
    assert captured["fetch"]["cli_token"] == "[REDACTED]"
    serialized = __import__("json").dumps(result)
    assert "x-9r-cli-token" not in serialized
    assert "cli-secret" not in serialized
    assert "[REDACTED]" not in serialized
    assert captured["fetch"]["base_url"] == "http://localhost:20128/v1"
    assert captured["fetch"]["scope"] == "profile:quota-test"
    assert "inference_api_key" not in captured["fetch"]
    assert "total" not in result
    assert len(result["providers"]) == 1
    provider = result["providers"][0]
    assert provider["provider"] == "9router"
    assert provider["routes"] == [
        {
            "route": "daily",
            "provider": "route-provider",
            "account": "quota-account",
            "usage": 12.0,
            "limit": 100.0,
            "remaining": 88.0,
            "remaining_percent": 88.0,
            "unit": "requests",
            "reset_at": "2026-09-14T00:00:00+00:00",
            "status": "reported",
            "source": "9router_management_api",
            "detail": None,
        }
    ]


def test_named_profile_without_scope_home_never_reads_global_home(monkeypatch):
    import hermes_constants
    import hermes_cli.runtime_provider as runtime_provider
    import hermes_cli.web_server as web_server
    import agent.ninerouter_usage as ninerouter_usage
    from agent.account_usage import AccountUsageSnapshot

    observed = {}

    @contextmanager
    def profile_scope(_profile):
        yield None

    def fail_global_home():
        raise AssertionError("named profile attempted to use global home")

    def fake_settings(profile, *, profile_home=None):
        observed["profile"] = profile
        observed["profile_home"] = profile_home
        return {
            "management_base_url": None,
            "auth_cookie": None,
            "data_dir": None,
            "allow_default_data_dir": False,
        }

    snapshot = AccountUsageSnapshot(
        provider="9router",
        source="9router_management_api",
        fetched_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
        unavailable_reason="profile binding unavailable",
        scope="profile:profile-alpha",
    )

    monkeypatch.setattr(web_server, "_config_profile_scope", profile_scope)
    monkeypatch.setattr(web_server, "_ninerouter_profile_settings", fake_settings)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", fail_global_home)
    monkeypatch.setattr(web_server, "load_config", lambda: {"model": {"provider": "openai-api"}})
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda requested: {"base_url": "http://127.0.0.1:20128/v1", "api_key": "[REDACTED]"},
    )
    fetch_called = {"value": False}

    def forbidden_fetch(**kwargs):
        fetch_called["value"] = True
        return snapshot

    monkeypatch.setattr(ninerouter_usage, "fetch_ninerouter_account_usage", forbidden_fetch)

    result = web_server._get_usage_quota("profile-alpha")

    assert observed == {"profile": "profile-alpha", "profile_home": None}
    assert fetch_called["value"] is False
    assert result["providers"][0]["available"] is False


def test_named_profile_without_management_credential_skips_quota_fetch(monkeypatch, tmp_path):
    import hermes_cli.web_server as web_server
    import hermes_cli.runtime_provider as runtime_provider
    import agent.ninerouter_usage as ninerouter_usage
    from agent.account_usage import AccountUsageRoute, AccountUsageSnapshot

    @contextmanager
    def profile_scope(_profile):
        yield tmp_path

    monkeypatch.setattr(web_server, "_config_profile_scope", profile_scope)
    monkeypatch.setattr(
        web_server,
        "_ninerouter_profile_settings",
        lambda profile, *, profile_home=None: {
            "management_base_url": "http://127.0.0.1:20128",
            "auth_cookie": None,
            "data_dir": None,
            "allow_default_data_dir": False,
        },
    )
    monkeypatch.setattr(web_server, "load_config", lambda: {"model": {"provider": "openai-api"}})
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda requested: {"base_url": "http://127.0.0.1:20128/v1", "api_key": None},
    )
    fetch_called = {"value": False}

    data_bearing_snapshot = AccountUsageSnapshot(
        provider="9router",
        source="9router_management_api",
        fetched_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
        title="9Router usage & quota",
        routes=(
            AccountUsageRoute(
                route="profile-secret-route",
                provider="route-provider",
                account="profile-account",
                usage=1.0,
                limit=10.0,
                remaining=9.0,
                remaining_percent=90.0,
                unit="requests",
                status="reported",
                source="9router_management_api",
            ),
        ),
        scope="profile:profile-alpha",
    )

    def forbidden_fetch(**_kwargs):
        fetch_called["value"] = True
        return data_bearing_snapshot

    monkeypatch.setattr(ninerouter_usage, "fetch_ninerouter_account_usage", forbidden_fetch)

    result = web_server._get_usage_quota("profile-alpha")

    assert fetch_called["value"] is False
    provider = result["providers"][0]
    assert provider["available"] is False
    assert provider["routes"] == []
    assert "credential" in provider["unavailable_reason"].lower()


def test_explicit_active_profile_uses_current_profile_quota_binding(monkeypatch, tmp_path):
    import hermes_cli.web_server as web_server
    import hermes_cli.runtime_provider as runtime_provider
    import agent.ninerouter_usage as ninerouter_usage
    import agent.secret_scope as secret_scope
    from agent.account_usage import AccountUsageRoute, AccountUsageSnapshot

    @contextmanager
    def profile_scope(_profile):
        yield tmp_path

    monkeypatch.setattr(web_server, "_config_profile_scope", profile_scope)
    monkeypatch.setattr(web_server, "get_process_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server, "load_config", lambda: {"model": {"provider": "openai-api"}})
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda requested: {"base_url": "http://127.0.0.1:20128/v1", "api_key": None},
    )
    monkeypatch.setattr(
        secret_scope,
        "build_profile_secret_scope",
        lambda home: {
            "HERMES_9ROUTER_MANAGEMENT_URL": "http://127.0.0.1:20128",
            "HERMES_9ROUTER_AUTH_COOKIE": "[REDACTED]",
            "HERMES_9ROUTER_DATA_DIR": str(tmp_path / "router-data"),
        },
    )
    captured = {}

    def fake_resolve(*args, **kwargs):
        captured["token_kwargs"] = kwargs
        return "[REDACTED]"

    monkeypatch.setattr(ninerouter_usage, "resolve_ninerouter_cli_token", fake_resolve)
    monkeypatch.setattr(
        ninerouter_usage,
        "fetch_ninerouter_account_usage",
        lambda **kwargs: AccountUsageSnapshot(
            provider="9router",
            source="9router_management_api",
            fetched_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
            title="9Router usage & quota",
            routes=(
                AccountUsageRoute(
                    route="active-profile-route",
                    provider="route-provider",
                    account="active-profile-account",
                    usage=1.0,
                    limit=10.0,
                    remaining=9.0,
                    remaining_percent=90.0,
                    unit="requests",
                    status="reported",
                    source="9router_management_api",
                ),
            ),
            scope="profile:coder",
        ),
    )

    result = web_server._get_usage_quota("coder")

    assert result["providers"][0]["available"] is True
    assert captured["token_kwargs"]["allow_default_data_dir"] is True


def test_profile_scope_display_label_is_fail_closed():
    import hermes_cli.web_server as web_server

    scope = web_server._ninerouter_scope(
        "prefix=https://user:[REDACTED]@profile.invalid"
    )

    assert scope == "profile:current"
    assert "[REDACTED]" not in scope


def test_openai_api_without_9router_runtime_is_fail_closed(monkeypatch):
    import hermes_cli.web_server as web_server
    import hermes_cli.runtime_provider as runtime_provider

    @contextmanager
    def profile_scope(_profile):
        yield

    monkeypatch.setattr(web_server, "_config_profile_scope", profile_scope)
    monkeypatch.setattr(web_server, "load_config", lambda: {"model": {"provider": "openai-api"}})
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda requested: {"base_url": None, "api_key": "[REDACTED]"},
    )

    result = web_server._get_usage_quota("quota-test")
    provider = result["providers"][0]

    assert provider["provider"] == "9router"
    assert provider["available"] is False
    assert provider["routes"] == []
    assert provider["unavailable_reason"]
    assert "[REDACTED]" not in provider["unavailable_reason"]
