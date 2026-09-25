"""Gateway ``plugins.manage`` — manifest ``config_schema`` rendered as settings fields (#46600, #87934).

Drives the real discovery + config writer against a temp HERMES_HOME: ``list`` carries each user
plugin's schema with the current values, and ``settings`` writes through the same
``plugins.entries.<id>.settings`` namespace ``ctx.get_config`` reads — never a secret into config.yaml.
"""

import pytest

from tui_gateway import server

MANIFEST = """\
name: demo-plugin
version: 1.0.0
config_schema:
  api_url: {type: str, default: "https://example.invalid", description: "Service endpoint"}
  retries: {type: int, default: 3}
  verbose: {type: bool, default: false}
  mode: {type: str, choices: [fast, careful], default: fast}
  api_key: {type: secret, description: "Token"}
"""


@pytest.fixture
def plugins_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "plugins" / "demo-plugin").mkdir(parents=True)
    (home / "plugins" / "demo-plugin" / "plugin.yaml").write_text(MANIFEST, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  entries:\n    demo-plugin:\n      settings:\n        retries: 7\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _manage(**params):
    return server.handle_request({"id": "1", "method": "plugins.manage", "params": params})


def test_list_carries_schema_fields_with_current_values_and_no_secret_values(plugins_home, monkeypatch):
    monkeypatch.setenv("DEMO_PLUGIN_API_KEY", "shh")

    rows = _manage(action="list")["result"]["plugins"]
    row = next(r for r in rows if r["key"] == "demo-plugin")
    fields = {f["key"]: f for f in row["settings_schema"]}

    assert fields["api_url"]["type"] == "string" and fields["api_url"]["value"] == "https://example.invalid"
    assert fields["retries"]["type"] == "number" and fields["retries"]["value"] == 7  # config.yaml wins over default
    assert fields["verbose"]["type"] == "boolean" and fields["verbose"]["value"] is False
    assert fields["mode"]["type"] == "enum" and fields["mode"]["choices"] == ["fast", "careful"]
    assert fields["api_key"] == {"key": "api_key", "type": "secret", "label": "api_key", "description": "Token",
                                 "required": False, "env": "DEMO_PLUGIN_API_KEY", "has_value": True}


def test_settings_writes_the_plugin_namespace_and_refuses_secrets_and_bad_types(plugins_home):
    resp = _manage(action="settings", key="demo-plugin", values={"api_url": "https://real.invalid", "retries": 2, "mode": "careful"})

    assert resp["result"]["ok"] is True and sorted(resp["result"]["written"]) == ["api_url", "mode", "retries"]
    from hermes_cli.config import load_config_readonly
    assert load_config_readonly()["plugins"]["entries"]["demo-plugin"]["settings"] == {
        "api_url": "https://real.invalid", "retries": 2, "mode": "careful"}
    refreshed = {f["key"]: f["value"] for f in resp["result"]["plugin"]["settings_schema"] if "value" in f}
    assert refreshed["retries"] == 2 and refreshed["mode"] == "careful"

    for values in ({"api_key": "leak"}, {"retries": "two"}, {"mode": "reckless"}, {"unknown": 1}):
        assert _manage(action="settings", key="demo-plugin", values=values)["error"]["code"] == 4021
    assert "api_key" not in (plugins_home / "config.yaml").read_text(encoding="utf-8")
