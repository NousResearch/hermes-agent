"""`config.set`/`config.get` round-trip for mirrored desktop settings.

The two keys are `desktop.pluginDecisions` and `desktop.keybinds`.
These two keys are the write-through cache behind the localStorage-origin-change
incident: the desktop renderer pushes its live value on every change and self-heals
from this store when its own localStorage comes up empty (e.g. after the renderer
origin moved from file:// to http://127.0.0.1:47891, d428df2a65). This test proves
the backend half of that contract: a value written via config.set is exactly what
config.get reads back, bad shapes are refused (4002) without writing anything, and
an unset key reads back as {} rather than raising or fabricating a default.
"""

import pytest
import hermes_yaml as yaml

from tui_gateway import server


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Point the server's config read/write at a temp file."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield tmp_path / "config.yaml"
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _set(key, value):
    return server._methods["config.set"](1, {"key": key, "value": value})


def _get(key):
    return server._methods["config.get"](1, {"key": key})


def test_plugin_decisions_round_trip(config_home):
    decisions = {"kanban": True, "zebra-notes": False}

    set_reply = _set("desktop.pluginDecisions", decisions)
    assert set_reply["result"] == {"key": "desktop.pluginDecisions", "value": decisions}

    get_reply = _get("desktop.pluginDecisions")
    assert get_reply["result"] == {"value": decisions}

    on_disk = yaml.safe_load(config_home.read_text())
    assert on_disk["desktop"]["pluginDecisions"] == decisions


def test_keybinds_round_trip(config_home):
    binds = {"sidebar.toggle": ["mod+b"], "session.new": []}

    set_reply = _set("desktop.keybinds", binds)
    assert set_reply["result"] == {"key": "desktop.keybinds", "value": binds}

    get_reply = _get("desktop.keybinds")
    assert get_reply["result"] == {"value": binds}

    on_disk = yaml.safe_load(config_home.read_text())
    assert on_disk["desktop"]["keybinds"] == binds


def test_unset_key_reads_back_empty_not_fabricated(config_home):
    assert _get("desktop.pluginDecisions")["result"] == {"value": {}}
    assert _get("desktop.keybinds")["result"] == {"value": {}}
    assert not config_home.exists()


@pytest.mark.parametrize("key,bad_value", [
    ("desktop.pluginDecisions", {"kanban": "yes"}),       # value must be bool
    ("desktop.pluginDecisions", ["kanban"]),               # must be an object
    ("desktop.keybinds", {"sidebar.toggle": "mod+b"}),     # must be a list
    ("desktop.keybinds", {"sidebar.toggle": [1, 2]}),      # combos must be strings
])
def test_bad_shape_refused_without_writing(config_home, key, bad_value):
    reply = _set(key, bad_value)
    assert reply["error"]["code"] == 4002
    assert not config_home.exists()


def test_writing_one_key_does_not_disturb_the_other_or_unrelated_sections(config_home):
    _set("desktop.pluginDecisions", {"kanban": True})
    _set("density", "on")  # unrelated existing config.set key, same config.yaml
    _set("desktop.keybinds", {"sidebar.toggle": ["mod+b"]})

    on_disk = yaml.safe_load(config_home.read_text())
    assert on_disk["desktop"]["pluginDecisions"] == {"kanban": True}
    assert on_disk["desktop"]["keybinds"] == {"sidebar.toggle": ["mod+b"]}
    assert on_disk["display"]["tui_compact"] is True

    # Unrelated sections are untouched.
    assert "approvals" not in on_disk
    assert "model" not in on_disk
