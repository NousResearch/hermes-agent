"""Readiness must keep working for the launch profile after serving another home."""

import json
from pathlib import Path

import pytest

from agent import secret_scope
from tui_gateway import launch_profile_policy, server


@pytest.fixture
def profile_homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    secondary = root / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    token = secret_scope.set_secret_scope(None)
    for home in (root, secondary):
        (home / "config.yaml").write_text(
            "model:\n  provider: openai-codex\n  default: gpt-5.5\n", encoding="utf-8")
    (root / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {"openai-codex": [{
            "id": "readiness-fixture", "auth_type": "oauth", "source": "device_code",
            "access_token": "readiness-fixture-token", "priority": 0,
        }]},
    }), encoding="utf-8")
    try:
        yield root, secondary
    finally:
        secret_scope.reset_secret_scope(token)


def test_codex_launch_readiness_survives_secondary_profile_activation(profile_homes):
    """Use the real RPC and credential-pool resolver; no provider calls or resolver mocks."""
    def check(params):
        return server.handle_request({"id": "check", "method": "setup.runtime_check", "params": params})["result"]

    before = check({})
    assert before["ok"] is True, before
    server._profile_home("secondary")
    assert secret_scope.is_multiplex_active()
    for params in ({}, {"profile": "default"}):
        result = check(params)
        assert result["ok"] is True, result
        assert result["provider"] == before["provider"]
        assert secret_scope.current_secret_scope() is None

    # Root OAuth-pool borrowing is intentional; the named-profile result keeps its identity.
    secondary = check({"profile": "secondary"})
    assert secondary["ok"] is True, secondary
    assert secondary["profile"] == "secondary"
    assert secret_scope.current_secret_scope() is None


def test_readiness_scope_preserves_launch_snapshot_and_releases_on_error(profile_homes, monkeypatch):
    """The shared setup shell binds launch authority, not later ambient/secondary values."""
    root, secondary = profile_homes
    key = "HERMES_CODEX_BASE_URL"
    monkeypatch.setenv(key, "https://launch.example.invalid")
    (secondary / ".env").write_text(f"{key}=https://secondary.example.invalid\n", encoding="utf-8")
    server._profile_home("secondary")
    monkeypatch.setenv(key, "https://ambient.example.invalid")

    def probe(profile, stamp):
        from hermes_constants import get_hermes_home
        return {"value": secret_scope.get_secret(key), "home": get_hermes_home(), **stamp}

    for params, expected, home in (
        ({}, "https://launch.example.invalid", root),
        ({"profile": "default"}, "https://launch.example.invalid", root),
        ({"profile": "secondary"}, "https://secondary.example.invalid", secondary),
    ):
        result = server._readiness_check(1, params, probe)["result"]
        assert result["value"] == expected
        assert Path(result["home"]) == home
        assert secret_scope.current_secret_scope() is None

    def failing_probe(profile, stamp):
        assert secret_scope.get_secret(key) == "https://launch.example.invalid"
        raise ValueError("probe failed")

    with pytest.raises(ValueError, match="probe failed"):
        server._readiness_check(1, {}, failing_probe)
    assert secret_scope.current_secret_scope() is None
