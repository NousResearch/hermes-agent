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
    monkeypatch.setattr("hermes_cli.plugins_settings._CHOICES_LIST_BUDGET_SECS", 0.2)
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


# ── choices_from cost on the list path: cache, aggregate budget, invalidation (review F1/F4/F6) ─────
COUNTING_PY = """\
import os
import time
from pathlib import Path

CALLS = Path(__file__).with_name("calls.log")


def _record(name):
    with CALLS.open("a", encoding="utf-8") as fh:
        fh.write(name + "\\n")


def _slow():
    time.sleep(float(os.environ.get("DYN_CHOICES_DELAY", "0")))
    gate = os.environ.get("DYN_CHOICES_GATE")  # block until the test creates this file
    deadline = time.monotonic() + 10
    while gate and not os.path.exists(gate) and time.monotonic() < deadline:
        time.sleep(0.01)


def slow_a(context):
    _record("a")
    _slow()
    return [{"value": "a1", "label": "A one"}, "a2"]


def slow_b(context):
    _record("b")
    _slow()
    return ["b1", "b2"]


def boom(context):
    _record("boom")
    raise RuntimeError("catalog offline")


def dupes(context):
    return [{"value": "d1", "label": "first"}, {"value": "d1", "label": "second"}, "d2", "d2"]


def per_home(context):
    name = Path(context["hermes_home"]).name
    _record(name)
    if name == "work":
        _slow()
    return [name]
"""


@pytest.fixture(autouse=True)
def _fresh_choices_state():
    from hermes_cli import plugins_settings as ps
    with ps._CHOICES_LOCK:
        ps._CHOICES_CACHE.clear()
        ps._CHOICES_GENERATION.clear()
    yield


def _counting_plugin(home, fields, *, settings="{}"):
    plugin = home / "plugins" / "dyn-plugin"
    plugin.mkdir(parents=True)
    schema = "".join(f"  {key}: {{type: str, choices_from: 'catalog:{func}', choices: [fallback]}}\n"
                     for key, func in fields.items())
    (plugin / "plugin.yaml").write_text(f"name: dyn-plugin\nversion: 1.0.0\nconfig_schema:\n{schema}", encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (plugin / "catalog.py").write_text(COUNTING_PY, encoding="utf-8")
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: [dyn-plugin]\n  entries:\n    dyn-plugin:\n      settings: {settings}\n", encoding="utf-8")
    return plugin


def _calls(plugin):
    log = plugin / "calls.log"
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def _dyn_fields(**params):
    rows = _manage(action="list", **params)["result"]["plugins"]
    return {f["key"]: f for f in next(r for r in rows if r["key"] == "dyn-plugin")["settings_schema"]}


def _timed_resolution(plugin):
    """Time only the settings-field build the list RPC runs (the rest of a list is plugin-independent)."""
    import time
    from hermes_cli import plugins_settings as ps
    start = time.monotonic()
    fields = ps.plugin_settings_fields_many([("dyn-plugin", plugin)])[0]
    return {f["key"]: f for f in fields}, time.monotonic() - start


def _wait_idle():
    import time
    from hermes_cli import plugins_settings as ps
    deadline = time.monotonic() + 10
    while ps._CHOICES_INFLIGHT and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not ps._CHOICES_INFLIGHT


def test_listing_resolves_slow_fields_concurrently_then_from_cache_until_a_save(tmp_path, monkeypatch):
    """The reviewer's probe shape: two slow ``choices_from`` fields. Serial + uncached cost the sum on every
    list; now a cold build costs about one delay and a warm one does not call the plugin at all."""
    monkeypatch.setattr("hermes_cli.plugins_settings._CHOICES_LIST_BUDGET_SECS", 3.0)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    plugin = _counting_plugin(tmp_path / "home", {"engine": "slow_a", "voice": "slow_b"})
    assert _dyn_fields()["engine"]["choices"] == ["a1", "a2"]  # loads the plugin; warms the cache
    assert sorted(_calls(plugin)) == ["a", "b"]  # the two fields run concurrently: any order

    from hermes_cli import plugins_settings as ps
    with ps._CHOICES_LOCK:
        ps._CHOICES_CACHE.clear()
    monkeypatch.setenv("DYN_CHOICES_DELAY", "0.5")
    cold, cold_secs = _timed_resolution(plugin)
    assert cold["engine"]["choices"] == ["a1", "a2"] and cold["voice"]["choices"] == ["b1", "b2"]
    assert cold_secs < 0.9, cold_secs  # ~0.5 (concurrent), not 1.0 (serial)

    warm, warm_secs = _timed_resolution(plugin)
    assert warm == cold and warm_secs < 0.2, warm_secs
    assert _dyn_fields()["voice"]["choices"] == ["b1", "b2"]  # the list RPC hits the same cache
    assert sorted(_calls(plugin)) == ["a", "a", "b", "b"]

    monkeypatch.setenv("DYN_CHOICES_DELAY", "0")
    # A save validates against a fresh call (not the cache), then drops the plugin's cached results, so
    # the refreshed row in the reply resolves again.
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "a2"})["result"]["ok"] is True
    assert sorted(_calls(plugin)[4:]) == ["a", "a", "b"]
    _dyn_fields()
    assert len(_calls(plugin)) == 7  # the refreshed row's results are cached again


def test_listing_budget_is_aggregate_and_late_results_are_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins_settings._CHOICES_LIST_BUDGET_SECS", 0.3)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    plugin = _counting_plugin(tmp_path / "home", {"engine": "slow_a", "voice": "slow_b"})
    _dyn_fields()
    from hermes_cli import plugins_settings as ps
    with ps._CHOICES_LOCK:
        ps._CHOICES_CACHE.clear()
    gate = tmp_path / "release"
    monkeypatch.setenv("DYN_CHOICES_GATE", str(gate))

    cold, cold_secs = _timed_resolution(plugin)
    assert cold["engine"]["choices"] == ["fallback"] and cold["voice"]["choices"] == ["fallback"]
    assert cold_secs < 0.55, cold_secs  # one 0.3s budget for both fields, not one deadline each

    gate.touch()  # the abandoned workers now finish and store their results
    _wait_idle()
    warm, warm_secs = _timed_resolution(plugin)
    assert warm["engine"]["choices"] == ["a1", "a2"] and warm["voice"]["choices"] == ["b1", "b2"]
    assert warm_secs < 0.2, warm_secs


def test_failing_choices_from_is_negative_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    plugin = _counting_plugin(tmp_path / "home", {"engine": "boom"})

    assert _dyn_fields()["engine"]["choices"] == ["fallback"]
    assert _dyn_fields()["engine"]["choices"] == ["fallback"]
    assert _calls(plugin).count("boom") == 1  # the failure is remembered, not retried per list

    from hermes_cli import plugins_settings as ps
    with ps._CHOICES_LOCK:  # expire it: the next list retries
        ps._CHOICES_CACHE.update({slot: (0.0, None) for slot in ps._CHOICES_CACHE})
    _dyn_fields()
    assert _calls(plugin).count("boom") == 2


def test_duplicate_choice_values_keep_the_first_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    _counting_plugin(tmp_path / "home", {"engine": "dupes"})

    field = _dyn_fields()["engine"]
    assert field["choices"] == ["d1", "d2"] and field["choice_labels"] == ["first", "d2"]


def test_stored_value_no_longer_offered_still_renders_and_saves_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    _counting_plugin(tmp_path / "home", {"engine": "slow_a"}, settings="{engine: retired}")

    field = _dyn_fields()["engine"]
    assert field["value"] == "retired"
    assert field["choices"] == ["a1", "a2", "retired"]
    assert field["choice_labels"] == ["A one", "a2", "retired (unavailable)"]

    assert _manage(action="settings", key="dyn-plugin", values={"engine": "retired"})["result"]["ok"] is True
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "other-gone"})["error"]["code"] == 4021
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "a1"})["result"]["ok"] is True
    # Once changed away, the retired value is a NEW value again and must be offered.
    assert _manage(action="settings", key="dyn-plugin", values={"engine": "retired"})["error"]["code"] == 4021


def test_choices_from_is_scoped_per_profile_a_b_a(tmp_path, monkeypatch):
    """Two profiles in one process: a hung call in ``work`` must not block or answer for ``other``, and
    each profile's cache holds its own results (A -> B -> A)."""
    monkeypatch.setattr("hermes_cli.plugins_settings._CHOICES_LIST_BUDGET_SECS", 0.3)
    gate = tmp_path / "release"
    monkeypatch.setenv("DYN_CHOICES_GATE", str(gate))
    root = tmp_path / "hermes_home"
    root.mkdir()
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    work = _counting_plugin(root / "profiles" / "work", {"engine": "per_home"})
    other = _counting_plugin(root / "profiles" / "other", {"engine": "per_home"})

    assert _dyn_fields(profile="work")["engine"]["choices"] == ["fallback"]  # A: still running
    assert _dyn_fields(profile="other")["engine"]["choices"] == ["other"]  # B: not blocked by A's slot
    assert _dyn_fields(profile="work")["engine"]["choices"] == ["fallback"]  # A: in flight, never B's answer
    gate.touch()
    _wait_idle()
    assert _dyn_fields(profile="work")["engine"]["choices"] == ["work"]  # A: its own late result, cached
    assert _dyn_fields(profile="other")["engine"]["choices"] == ["other"]
    assert _calls(work) == ["work"] and _calls(other) == ["other"]
