"""profiles.create exposes the launch profile's model and custom gateway through inheritance.

The profile owns only its overrides.  The effective configuration must still expose the launch
model and its custom provider, without persisting an expanded launch secret into the child.
"""

from __future__ import annotations

import tui_gateway.server as srv
from hermes_cli.config import load_config, read_user_config_raw


def test_inherit_launch_model_carries_a_custom_provider_gateway(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("FAKE_GW_TOKEN", "gw-FAKE-111")
    (home / "config.yaml").write_text(
        "model:\n  provider: my-gateway\n  default: my-finetune\n"
        "providers:\n  my-gateway:\n    api: https://llm.internal.example.com/v1\n    key_env: GW_KEY\n    api_key: ${FAKE_GW_TOKEN}\n"
        "  unrelated:\n    api: https://other.example.com/v1\n")
    profile = home / "profiles" / "scout"
    profile.mkdir(parents=True)

    assert srv._inherit_launch_model(profile) is True

    with srv._hermes_home_scope(profile):
        cfg = load_config()
        child_raw = read_user_config_raw() or {}
    assert (cfg["model"]["provider"], cfg["model"]["default"]) == ("my-gateway", "my-finetune")
    assert "my-gateway" in cfg["providers"]
    assert cfg["providers"]["my-gateway"]["api_key"] == "${FAKE_GW_TOKEN}"
    # The child is a delta: the launch template stays in the parent and its resolved secret is
    # never persisted into the child.
    assert "my-gateway" not in child_raw.get("providers", {})
    assert "model" not in child_raw
    assert not (profile / "config.yaml").exists()
