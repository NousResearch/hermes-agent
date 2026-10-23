"""profiles.create inherits the launch profile's model together with the custom gateway backing it.

``_inherit_launch_model`` pins the launch ``model`` into the new profile through the same
validated path as ``/api/model/set``, and that validation runs inside the new profile's home. A
launch model on a custom ``providers:`` gateway is therefore only valid once the gateway
definition is already in the profile — writing it after the pin never ran (#101885 / #94071).
"""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli.config import DEFAULT_CONFIG, atomic_config_write, read_user_config_raw
import tui_gateway.server as srv


@pytest.mark.parametrize(
    "schema,provider",
    [
        ("providers", "custom:my-gateway"),
        ("custom_providers", "custom:my-gateway"),
    ],
)
def test_inherit_launch_model_carries_a_custom_provider_gateway(
    monkeypatch, tmp_path, schema, provider
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("FAKE_GW_TOKEN", "gw-FAKE-111")
    endpoint = {
        "name": "my-gateway",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "${FAKE_GW_TOKEN}",
    }
    unrelated = {"name": "unrelated", "base_url": "http://127.0.0.1:11435/v1"}
    catalog = (
        {"my-gateway": endpoint, "unrelated": unrelated}
        if schema == "providers"
        else [endpoint, unrelated]
    )
    atomic_config_write(
        home / "config.yaml",
        {
            "model": {"provider": provider, "default": "my-finetune"},
            schema: catalog,
            "_config_version": DEFAULT_CONFIG["_config_version"],
        },
    )
    profile = home / "profiles" / "scout"
    profile.mkdir(parents=True)
    retained = {
        "name": "retained",
        "base_url": "http://127.0.0.1:11436/v1",
        "api_key": "${RETAINED_TOKEN}",
    }
    # Same identity under another key: runtime takes the first alias match, so it must not survive.
    stale = {"name": "my-gateway", "base_url": "http://127.0.0.1:11437/v1"}
    retained_catalog = (
        {"stale": stale, "retained": retained}
        if schema == "providers"
        else [stale, retained]
    )
    atomic_config_write(
        profile / "config.yaml",
        {
            "tts": {"provider": "edge"},
            schema: retained_catalog,
            "_config_version": DEFAULT_CONFIG["_config_version"],
        },
    )

    assert srv._inherit_launch_model(profile) is True

    cfg = yaml.safe_load((profile / "config.yaml").read_text(encoding="utf-8"))
    assert (cfg["model"]["provider"], cfg["model"]["default"]) == (
        provider,
        "my-finetune",
    )
    expected = (
        {"my-gateway": endpoint, "retained": retained}
        if schema == "providers"
        else [endpoint, retained]
    )
    assert cfg[schema] == expected
    # The gateway travels as the launch file wrote it: the ref, never the launch secret.
    copied = (
        cfg["providers"]["my-gateway"]
        if schema == "providers"
        else cfg["custom_providers"][0]
    )
    assert copied["api_key"] == "${FAKE_GW_TOKEN}"


@pytest.mark.parametrize("written", ["http://127.0.0.1:11434/v1", "${LAB_URL}"])
def test_inherit_launch_model_pins_a_bare_custom_endpoint(monkeypatch, tmp_path, written):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("CUSTOM_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LAB_URL", "http://127.0.0.1:11434/v1")
    # Hermetic: the pin probes the endpoint's /models.
    monkeypatch.setattr("hermes_cli.models.probe_api_models", lambda *a, **k: {"models": ["llama3"]})
    atomic_config_write(
        home / "config.yaml",
        {
            "model": {"provider": "custom", "default": "llama3", "base_url": written},
            "_config_version": DEFAULT_CONFIG["_config_version"],
        },
    )
    profile = home / "profiles" / "scout"
    profile.mkdir(parents=True)
    # Voice mirroring writes config.yaml before inheritance runs (#85755).
    atomic_config_write(
        profile / "config.yaml",
        {"tts": {"provider": "edge"}, "_config_version": DEFAULT_CONFIG["_config_version"]},
    )

    assert srv._inherit_launch_model(profile) is True

    model = read_user_config_raw(profile / "config.yaml")["model"]
    # The launch file's spelling travels: a ${VAR} URL stays a ref, never the launch value.
    assert (model["provider"], model["default"], model["base_url"]) == ("custom", "llama3", written)


def test_inherit_launch_model_preserves_explicit_profile_selection(
    monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    atomic_config_write(
        home / "config.yaml", {"model": {"provider": "nous", "default": "launch-model"}}
    )
    profile = home / "profiles" / "scout"
    profile.mkdir(parents=True)
    selected = {"model": {"provider": "openrouter", "default": "profile-model"}}
    atomic_config_write(profile / "config.yaml", selected)
    before = read_user_config_raw(profile / "config.yaml")

    assert srv._inherit_launch_model(profile) is False
    assert read_user_config_raw(profile / "config.yaml") == before
