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
  tier: {type: str, choices: [{value: t1, label: "Tier one"}, t2]}
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
    assert "choice_labels" not in fields["mode"]  # unlabelled choices: the pre-label wire shape exactly
    assert fields["tier"]["choices"] == ["t1", "t2"] and fields["tier"]["choice_labels"] == ["Tier one", "t2"]
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

    assert _manage(action="settings", key="demo-plugin", values={"tier": "t1"})["result"]["written"] == ["tier"]
    for values in ({"api_key": "leak"}, {"retries": "two"}, {"mode": "reckless"}, {"unknown": 1},
                   {"tier": "Tier one"}):  # a label is display text, never a saved value
        assert _manage(action="settings", key="demo-plugin", values=values)["error"]["code"] == 4021
    assert "api_key" not in (plugins_home / "config.yaml").read_text(encoding="utf-8")


# ── choices_from: dynamic choices resolved from the plugin's own package ─────────────────────────
CHOICES_PY = """\
import time
from pathlib import Path

Path(__file__).with_name("imported.marker").touch()


def good(context):
    return [{"value": "m1", "label": "Model one"}, "m2", context["plugin_id"] + ":" + context["settings"]["region"]]


def boom(context):
    raise RuntimeError("catalog offline")


def bad_shape(context):
    return {"m1": "Model one"}


def hang(context):
    time.sleep(3)
    return ["late"]
"""


def _dynamic_plugin(tmp_path, monkeypatch, *, func, static=None, enabled=True):
    home = tmp_path / "hermes-home"
    plugin = home / "plugins" / "dyn-plugin"
    plugin.mkdir(parents=True)
    static_yaml = f", choices: {static}" if static else ""
    (plugin / "plugin.yaml").write_text(
        f"name: dyn-plugin\nversion: 1.0.0\nconfig_schema:\n"
        f"  engine: {{type: str, choices_from: 'catalog:{func}'{static_yaml}}}\n", encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (plugin / "catalog.py").write_text(CHOICES_PY, encoding="utf-8")
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: [{'dyn-plugin' if enabled else ''}]\n"
        "  entries:\n    dyn-plugin:\n      settings:\n        region: eu\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return plugin


def _engine_field():
    rows = _manage(action="list")["result"]["plugins"]
    return next(f for f in next(r for r in rows if r["key"] == "dyn-plugin")["settings_schema"] if f["key"] == "engine")


def test_choices_from_renders_and_validates_the_plugins_own_choices(tmp_path, monkeypatch):
    _dynamic_plugin(tmp_path, monkeypatch, func="good", static="[fallback]")

    field = _engine_field()

    # Called with the plugin id and its current settings; labels ride alongside the values.
    assert field["type"] == "enum" and field["choices"] == ["m1", "m2", "dyn-plugin:eu"]
    assert field["choice_labels"] == ["Model one", "m2", "dyn-plugin:eu"]
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "m2"})["result"]["written"] == ["engine"]
    # Save re-resolves the dynamic list: the static entry is not offered, so it is refused.
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "fallback"})["error"]["code"] == 4021


@pytest.mark.parametrize("func", ["boom", "bad_shape", "hang"])
def test_failing_choices_from_falls_back_to_static_choices_then_free_text(tmp_path, monkeypatch, func):
    monkeypatch.setattr("hermes_cli.plugins_settings._CHOICES_TIMEOUT_SECS", 0.2)
    _dynamic_plugin(tmp_path / "static", monkeypatch, func=func, static="[fallback]")

    field = _engine_field()
    assert field["type"] == "enum" and field["choices"] == ["fallback"]
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "m1"})["error"]["code"] == 4021
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "fallback"})["result"]["ok"] is True

    from hermes_cli.plugins import _reset_plugin_managers_for_tests
    _reset_plugin_managers_for_tests()
    _dynamic_plugin(tmp_path / "free", monkeypatch, func=func)

    field = _engine_field()
    assert field["type"] == "string" and "choices" not in field
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "anything"})["result"]["ok"] is True


def test_choices_from_never_runs_for_a_disabled_plugin(tmp_path, monkeypatch):
    plugin = _dynamic_plugin(tmp_path, monkeypatch, func="good", static="[fallback]", enabled=False)

    assert _engine_field()["choices"] == ["fallback"]
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "fallback"})["result"]["ok"] is True
    assert not (plugin / "imported.marker").exists()  # the plugin's module was never imported
