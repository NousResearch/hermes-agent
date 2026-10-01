"""A user-installed ``external_process`` provider must survive shared picker discovery and native selection.

Drives a fake process profile (registered like a ``$HERMES_HOME/plugins/<name>`` model-provider plugin
would) through ``list_available_providers``, ``provider_model_ids``, the TUI/Desktop ``model.options``
RPC, the ``hermes model`` setup picker and a session-scoped ``config.set model`` switch.
"""
import shutil
import sys
from pathlib import Path

import pytest

from providers import register_provider
from providers.base import ProviderProfile

LIVE = ["fake-large[1m]", "fake-credit[1m]", "fake-small"]


class _FakeProcessProfile(ProviderProfile):
    def fetch_models(self, **_):
        return list(LIVE)

    def discover_models(self, **_):
        return [{"id": m, "label": m, "note": "usage credits" if "credit" in m else ""} for m in LIVE]

    def setup_status(self, **_):
        return {"available": True, "logged_in": True, "plan": "Fake Pro", "detail": "", "login_command": None}


@pytest.fixture
def picker_env(monkeypatch, tmp_path):
    # Real registries/config/runtime resolution, isolated from the user's home.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    profile = _FakeProcessProfile(
        name="fake-process-provider", display_name="Fake Process Provider", auth_type="external_process",
        api_mode="chat_completions", base_url="process://fake-process-provider", process_command="fake-process-cli",
        default_aux_model="fake-small", fallback_models=("fake-large[1m]", "fake-small", "fake-pinned-only"),
        model_aliases={"large": "fake-large[1m]"})
    register_provider(profile)
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda cmd, *a, **kw: sys.executable if cmd == profile.process_command else real_which(cmd, *a, **kw))
    import agent.models_dev as models_dev
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda *a, **kw: {})
    import hermes_cli.inventory as inventory
    monkeypatch.setattr(inventory, "_prewarm_pricing_async", lambda *a, **kw: None)
    return home, profile


def test_process_provider_reaches_every_shared_picker(picker_env, monkeypatch):
    home, profile = picker_env
    from hermes_cli.config import save_config
    from hermes_cli.main_provider_setup import _build_provider_picker_rows
    from hermes_cli.models import _PROVIDER_LABELS, list_available_providers, provider_model_ids
    from tui_gateway import server

    assert any(row["id"] == profile.name for row in list_available_providers())
    rows, _ = _build_provider_picker_rows({}, "", _PROVIDER_LABELS, {})
    assert any(row[0] == profile.name for row in rows)
    # The account's live picker (the profile's own probe) is what the shared pickers list,
    # merged with the pinned catalog so a declared id the probe omits is still selectable.
    ids = provider_model_ids(profile.name)
    assert set(LIVE) <= set(ids) and "fake-pinned-only" in ids

    # GUI read path: a cold catalog cache must still list the pinned floor (never an empty row);
    # an explicit refresh blocks on the profile's probe and adds the live ids.
    response = server._methods["model.options"](1, {})
    assert "error" not in response, response
    row = next(r for r in response["result"]["providers"] if r["slug"] == profile.name)
    assert set(profile.fallback_models) <= set(row["models"])
    assert row["authenticated"]
    response = server._methods["model.options"](1, {"refresh": True})
    row = next(r for r in response["result"]["providers"] if r["slug"] == profile.name)
    assert set(LIVE) <= set(row["models"])

    # Both native clients use model.options. The desktop explicit-only view
    # must keep a configured process provider without borrowing API credentials.
    save_config({"model": {"provider": profile.name, "default": profile.default_aux_model}})
    for explicit_only in (False, True):
        response = server._methods["model.options"](1, {"explicit_only": explicit_only, "refresh": True})
        assert "error" not in response, response
        row = next(r for r in response["result"]["providers"] if r["slug"] == profile.name)
        assert set(LIVE) <= set(row["models"])
        assert row["is_current"]
        assert row["authenticated"]

    # The setup picker dispatches the process row through the generic plugin flow and persists it.
    from hermes_cli import main, model_setup_flows
    from hermes_cli.config import load_config
    selected = "fake-credit[1m]"
    seen = {}
    monkeypatch.setattr(main, "_pick_provider", lambda *a: profile.name)

    def fake_pick(model_list, prompt, **kwargs):
        seen.update(models=model_list, notes=kwargs.get("notes"))
        return selected

    monkeypatch.setattr(model_setup_flows, "_pick_model_or_prompt", fake_pick)
    main.select_provider_and_model()
    assert seen["notes"] == {"fake-credit[1m]": "usage credits"}
    saved = load_config()["model"]
    assert saved["provider"] == profile.name
    assert saved["default"] == selected
    assert saved["api_mode"] == profile.api_mode
    assert saved["base_url"] == profile.base_url

    # An unavailable process must not overwrite an existing saved selection.
    before = (home / "config.yaml").read_bytes()
    monkeypatch.setattr(shutil, "which", lambda cmd, *a, **kw: None)
    main.select_provider_and_model()
    assert (home / "config.yaml").read_bytes() == before


def test_native_picker_selection_preserves_process_runtime(picker_env, monkeypatch):
    home, profile = picker_env
    from hermes_cli.config import load_config, save_config
    from hermes_cli.models import provider_model_ids
    from tui_gateway import server

    save_config(load_config())  # materialize defaults before checking session-only writes
    config_before = (home / "config.yaml").read_bytes()
    session = {"agent": None, "running": False}
    monkeypatch.setitem(server._sessions, "process-picker", session)
    models = provider_model_ids(profile.name)
    assert models, "A registered process provider must expose selectable models"
    for model in models:
        response = server._methods["config.set"](2, {
            "session_id": "process-picker", "key": "model",
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
