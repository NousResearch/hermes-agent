"""DirectSDK profiles must survive shared picker discovery and native selection."""

from pathlib import Path

import pytest


@pytest.fixture
def picker_env(monkeypatch, tmp_path):
    # Real registries/config/runtime resolution, isolated from the user's CLI login.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from providers import get_provider_profile
    profile = get_provider_profile("claude-subscription-directsdk-experimental")
    assert profile is not None
    # A stand-in CLI that reports a login and answers the picker handshake; never native or paid inference.
    import shutil
    fake = tmp_path / "claude"
    fake.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in auth) echo '{\"loggedIn\": true, \"authMethod\": \"claude.ai\", \"subscriptionType\": \"pro\"}'; exit 0;; esac\n"
        "read _req\n"
        "echo '{\"type\": \"control_response\", \"response\": {\"subtype\": \"success\", \"request_id\": \"x\", \"response\": {\"account\": {\"subscriptionType\": \"Claude Pro\"}, \"models\": ["
        "{\"value\": \"sonnet[1m]\", \"resolvedModel\": \"claude-sonnet-5[1m]\", \"description\": \"Sonnet 5\"},"
        "{\"value\": \"claude-fable-5-1[1m]\", \"resolvedModel\": \"claude-fable-5-1[1m]\", \"description\": \"Fable 5.1\"},"
        "{\"value\": \"haiku\", \"resolvedModel\": \"claude-haiku-4-5-20251001\", \"description\": \"Haiku 4.5\"}]}}}'\n")
    fake.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda cmd, *a, **kw: str(fake) if cmd == profile.process_command else None)
    import agent.models_dev as models_dev
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda *a, **kw: {})
    import hermes_cli.inventory as inventory
    monkeypatch.setattr(inventory, "_prewarm_pricing_async", lambda *a, **kw: None)
    return home, profile


def test_directsdk_picker_discovers_profile_catalog(picker_env, monkeypatch):
    home, profile = picker_env
    from hermes_cli.config import save_config
    from hermes_cli.main_provider_setup import _build_provider_picker_rows
    from hermes_cli.models import _PROVIDER_LABELS, list_available_providers, provider_model_ids
    from tui_gateway import server

    assert any(row["id"] == profile.name for row in list_available_providers())
    rows, _ = _build_provider_picker_rows({}, "", _PROVIDER_LABELS, {})
    assert any(row[0] == profile.name for row in rows)
    # The account's live picker (from the CLI handshake) is what the shared pickers list.
    live = ["claude-sonnet-5[1m]", "claude-fable-5-1[1m]", "claude-haiku-4-5-20251001"]
    assert provider_model_ids(profile.name) == live

    response = server._methods["model.options"](1, {})
    assert "error" not in response, response
    row = next(r for r in response["result"]["providers"] if r["slug"] == profile.name)
    assert set(live) <= set(row["models"])
    assert row["authenticated"]

    # Both native clients use model.options. The desktop explicit-only view
    # must keep a configured process provider without borrowing API credentials.
    save_config({"model": {"provider": profile.name, "default": profile.default_aux_model}})
    for explicit_only in (False, True):
        response = server._methods["model.options"](1, {"explicit_only": explicit_only})
        assert "error" not in response, response
        row = next(r for r in response["result"]["providers"] if r["slug"] == profile.name)
        assert set(live) <= set(row["models"])
        assert row["is_current"]
        assert row["authenticated"]

    # The setup picker must actually dispatch the newly visible process row.
    from hermes_cli import main, model_setup_flows
    from hermes_cli.config import load_config
    selected = "claude-fable-5-1[1m]"
    monkeypatch.setattr(main, "_pick_provider", lambda *a: profile.name)
    monkeypatch.setattr(model_setup_flows, "_pick_model_or_prompt", lambda *a, **kw: selected)
    main.select_provider_and_model()
    saved = load_config()["model"]
    assert saved["provider"] == profile.name
    assert saved["default"] == selected
    assert saved["api_mode"] == profile.api_mode
    assert saved["base_url"] == profile.base_url

    # An unavailable process must not overwrite an existing saved selection.
    before = (home / "config.yaml").read_bytes()
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd, *a, **kw: None)
    main.select_provider_and_model()
    assert (home / "config.yaml").read_bytes() == before


def test_directsdk_native_picker_selection_preserves_runtime(picker_env, monkeypatch):
    home, profile = picker_env
    from hermes_cli.models import provider_model_ids
    from hermes_cli.config import load_config, save_config
    from tui_gateway import server

    save_config(load_config())  # materialize defaults before checking session-only writes
    config_before = (home / "config.yaml").read_bytes()
    session = {"agent": None, "running": False}
    monkeypatch.setitem(server._sessions, "directsdk-picker", session)
    models = provider_model_ids(profile.name)
    assert models, "A registered process provider must expose selectable models"
    for model in models:
        response = server._methods["config.set"](2, {
            "session_id": "directsdk-picker", "key": "model",
            "value": f"{model} --provider {profile.name} --session",
            "confirm_expensive_model": True,
        })
        assert "error" not in response, response
        assert response["result"]["value"] == model
        runtime = session["model_override"]
        assert runtime["model"] == model
        assert runtime["provider"] == profile.name
        assert runtime["api_mode"] == profile.api_mode
        assert runtime["base_url"] == profile.base_url
    assert (home / "config.yaml").read_bytes() == config_before
