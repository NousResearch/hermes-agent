"""profiles.create inherits the launch profile's model together with the custom gateway backing it.

``_inherit_launch_model`` pins the launch ``model`` into the new profile through the same
validated path as ``/api/model/set``, and that validation runs inside the new profile's home. A
launch model on a custom ``providers:`` gateway is therefore only valid once the gateway
definition is already in the profile — writing it after the pin never ran (#101885 / #94071).
"""

from __future__ import annotations

import yaml

import tui_gateway.server as srv


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

    cfg = yaml.safe_load((profile / "config.yaml").read_text())
    assert (cfg["model"]["provider"], cfg["model"]["default"]) == ("my-gateway", "my-finetune")
    assert set(cfg["providers"]) == {"my-gateway"}
    # The gateway travels as the launch file wrote it: the ref, never the launch secret.
    assert cfg["providers"]["my-gateway"]["api_key"] == "${FAKE_GW_TOKEN}"
