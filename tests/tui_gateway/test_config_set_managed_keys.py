"""``config.set`` refuses a key the administrator's managed scope pins, like ``hermes config set``.

Before, both TUI paths wrote the pinned key and answered success while the managed value kept
winning on every read. A managed scope that pins something else must not block other keys.
"""
import pytest
import hermes_yaml as yaml

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


def test_tui_persisting_model_pick_mid_turn_is_refused_not_queued(homes):
    """A running turn queues the pick and acks it; a persisting one must be refused before that."""
    server._sessions["managed-running"] = session = {"session_key": "managed-running", "agent": None, "running": True}
    try:
        answer = _set(key="model", value="other/model --global", session_id="managed-running")
    finally:
        server._sessions.pop("managed-running", None)
    assert answer["error"]["code"] == 4030, answer
    assert "model.default" in answer["error"]["message"]
    assert "pending_model_switch" not in session


def test_tui_config_set_still_writes_a_key_the_scope_does_not_pin(homes):
    assert _set(key="theme", value="light")["result"] == {"key": "theme", "value": "light"}
    assert yaml.safe_load((homes / "config.yaml").read_text())["display"]["tui_theme"] == "light"
