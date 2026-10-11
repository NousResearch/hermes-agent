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


SECTIONED_MANIFEST = """\
name: sectioned-plugin
version: 1.0.0
config_schema:
  enabled: {type: bool, default: true, section: Behaviour}
  mode: {type: str, choices: [rotate, compact], default: rotate, section: Behaviour}
  bar_width: {type: int, default: 10, group: Appearance}
  emoji: {type: str, default: "x", section: Appearance}
  legacy: {type: bool, default: false}
"""


@pytest.fixture
def sectioned_home(tmp_path, monkeypatch):
    home = tmp_path / "sectioned-home"
    (home / "plugins" / "sectioned-plugin").mkdir(parents=True)
    (home / "plugins" / "sectioned-plugin" / "plugin.yaml").write_text(SECTIONED_MANIFEST, encoding="utf-8")
    (home / "config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _sectioned_fields(sectioned_home):
    rows = _manage(action="list")["result"]["plugins"]
    row = next(r for r in rows if r["key"] == "sectioned-plugin")
    return row["settings_schema"]


def test_section_heading_is_carried_on_the_wire(sectioned_home):
    fields = {f["key"]: f for f in _sectioned_fields(sectioned_home)}

    assert fields["enabled"]["section"] == "Behaviour"
    assert fields["mode"]["section"] == "Behaviour"
    # `group` is an accepted alias so manifests do not have to guess the spelling.
    assert fields["bar_width"]["section"] == "Appearance"
    assert fields["emoji"]["section"] == "Appearance"
    # A key that declares neither stays flat — and carries no empty-string section.
    assert "section" not in fields["legacy"]


def test_undeclared_section_is_absent_not_empty(sectioned_home):
    """Older Hermes build a field dict without the key at all; a missing section must
    stay missing rather than becoming `""`, so a naive client does not render a blank heading."""
    for field in _sectioned_fields(sectioned_home):
        if "section" in field:
            assert field["section"], field


def test_section_does_not_leak_into_the_saved_value(sectioned_home):
    """`section` is presentation metadata; writing settings must store only the value."""
    resp = _manage(action="settings", key="sectioned-plugin", values={"bar_width": 20})
    assert resp["result"]["ok"] is True

    from hermes_cli.config import load_config_readonly
    saved = load_config_readonly()["plugins"]["entries"]["sectioned-plugin"]["settings"]
    assert saved == {"bar_width": 20}
