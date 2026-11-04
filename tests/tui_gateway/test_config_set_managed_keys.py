"""``config.set`` refuses a key the administrator's managed scope pins, like ``hermes config set``.

Before, both TUI paths wrote the pinned key and answered success while the managed value kept
winning on every read. A managed scope that pins something else must not block other keys.
"""
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

from hermes_cli import managed_scope
from tui_gateway import server


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from hermes_cli import managed_scope
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (managed / "config.yaml").write_text("display:\n  skin: corp\n  sections:\n    tools: collapsed\n"
                                         "custom_prompt: policy\nmodel:\n  default: acme/pinned\n")
    (home / "config.yaml").write_text("display:\n  tui_theme: dark\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(server, "_hermes_home", home)
    managed_scope.invalidate_managed_cache()
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield home
    managed_scope.invalidate_managed_cache()
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _set(**params):
    return server._methods["config.set"](1, params)


@pytest.mark.parametrize(("key", "value", "pinned"), [
    ("skin", "mono", "display.skin"),  # one-key funnel (_write_config_keys)
    ("details_mode", "expanded", "display.sections.tools"),  # fan-out to every section leaf
    ("prompt", "mine", "custom_prompt"),  # its own raw save
])
def test_tui_config_set_refuses_a_managed_key_and_writes_nothing(homes, key, value, pinned):
    before = (homes / "config.yaml").read_text()
    answer = _set(key=key, value=value)
    assert answer["error"]["code"] == 4030, answer
    assert pinned in answer["error"]["message"] and "managed by your administrator" in answer["error"]["message"]
    assert (homes / "config.yaml").read_text() == before


@pytest.mark.parametrize("running", [True, False], ids=["deferred", "immediate"])
@pytest.mark.parametrize(("pick", "pinned"), [
    ("other/model --global", "model.default"),
    ("other/model --global --reasoning high", "agent.reasoning_effort"),  # only the effort is pinned
])
def test_tui_persisting_model_pick_is_refused_before_anything_changes(homes, monkeypatch, running, pick, pinned):
    """Refused before the live switch, the model persist or the mid-turn queue, not after a partial success."""
    if pinned == "agent.reasoning_effort":
        (homes.parent / "managed" / "config.yaml").write_text("agent:\n  reasoning_effort: low\n")
        managed_scope.invalidate_managed_cache()
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: pytest.fail("switched before refusal"))
    agent = SimpleNamespace(model="old/model", provider="nous", reasoning_config=None)
    server._sessions["managed-pick"] = session = {"session_key": "managed-pick", "agent": agent, "running": running}
    before = (homes / "config.yaml").read_text()
    try:
        answer = _set(key="model", value=pick, session_id="managed-pick")
    finally:
        server._sessions.pop("managed-pick", None)
    assert answer["error"]["code"] == 4030, answer
    assert pinned in answer["error"]["message"]
    assert session == {"session_key": "managed-pick", "agent": agent, "running": running}  # nothing queued
    assert vars(agent) == {"model": "old/model", "provider": "nous", "reasoning_config": None}
    assert (homes / "config.yaml").read_text() == before


def test_tui_config_set_still_writes_a_key_the_scope_does_not_pin(homes):
    assert _set(key="theme", value="light")["result"] == {"key": "theme", "value": "light"}
    assert yaml.safe_load((homes / "config.yaml").read_text())["display"]["tui_theme"] == "light"
