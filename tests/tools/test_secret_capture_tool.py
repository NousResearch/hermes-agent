"""Behavior contracts for general masked secret capture."""

import json
from types import SimpleNamespace
from uuid import UUID

from tools.registry import registry
from tools.secret_capture_tool import _set_thread_secret_capture_callback


def test_registry_capture_never_returns_value_and_replaces_existing_secret():
    captured = []

    def callback(name, prompt, metadata):
        captured.append((name, prompt, metadata))
        return {"success": True, "stored_as": name, "validated": False, "secret": "do-not-return"}

    _set_thread_secret_capture_callback(callback)
    try:
        result = json.loads(registry.dispatch("secret_capture", {
            "var_name": "OPENROUTER_API_KEY",
            "prompt": "Replacement OpenRouter API key",
            "destination": "profile_env",
        }))
    finally:
        _set_thread_secret_capture_callback(None)

    assert captured == [(
        "OPENROUTER_API_KEY",
        "Replacement OpenRouter API key",
        {"destination": "profile_env", "source": "secret_capture"},
    )]
    assert result == {
        "success": True,
        "stored_as": "OPENROUTER_API_KEY",
        "destination": "profile_env",
    }
    assert "do-not-return" not in json.dumps(result)


def test_bitwarden_write_updates_in_process_without_putting_value_on_argv(monkeypatch, tmp_path):
    from agent.secret_sources import bitwarden, bitwarden_write

    organization_id = UUID("11111111-1111-1111-1111-111111111111")
    project_id = UUID("22222222-2222-2222-2222-222222222222")
    secret_id = UUID("33333333-3333-3333-3333-333333333333")
    existing = SimpleNamespace(
        id=secret_id,
        key="OPENROUTER_API_KEY",
        note="keep me",
        organization_id=organization_id,
        project_id=project_id,
    )
    calls = []

    class Secrets:
        def sync(self, org, last_synced):
            calls.append(("sync", org, last_synced))
            return SimpleNamespace(success=True, data=SimpleNamespace(secrets=[existing]))

        def update(self, *args):
            calls.append(("update", *args))
            return SimpleNamespace(success=True, data=SimpleNamespace(id=secret_id))

        def create(self, *args):
            raise AssertionError("existing secret should be updated")

    class Client:
        def __init__(self, settings):
            calls.append(("client", settings))

        def auth(self):
            return SimpleNamespace(login_access_token=lambda token: SimpleNamespace(
                success=True, data=SimpleNamespace()))

        def projects(self):
            return SimpleNamespace(get=lambda value: SimpleNamespace(
                success=True,
                data=SimpleNamespace(id=project_id, organization_id=organization_id),
            ))

        def secrets(self):
            return Secrets()

    fake_sdk = SimpleNamespace(
        BitwardenClient=Client,
        DeviceType=SimpleNamespace(SDK="sdk"),
        client_settings_from_dict=lambda settings: settings,
    )
    monkeypatch.setattr(bitwarden_write, "_bitwarden_config", lambda: {
        "enabled": True,
        "project_id": str(project_id),
        "access_token_env": "BWS_ACCESS_TOKEN",
    })
    monkeypatch.setattr(bitwarden_write, "get_secret", lambda name, default="": "machine-token")
    monkeypatch.setattr(bitwarden_write, "_load_sdk", lambda: fake_sdk)
    monkeypatch.setattr(bitwarden_write, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(bitwarden, "clear_caches", lambda home: calls.append(("clear", home)))

    result = bitwarden_write.store_bitwarden_secret("OPENROUTER_API_KEY", "captured-secret")

    assert result == {
        "success": True,
        "stored_as": "OPENROUTER_API_KEY",
        "validated": True,
        "operation": "updated",
    }
    update = next(call for call in calls if call[0] == "update")
    assert update[1:] == (
        organization_id,
        str(secret_id),
        "OPENROUTER_API_KEY",
        "captured-secret",
        "keep me",
        [project_id],
    )
