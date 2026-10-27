"""Adding to config.yaml keeps every section banner on top of the section it introduces.

ruamel files the comment lines between two sections on the LAST entry of the section above, so a
key appended to that section used to land below the next section's banner (the layout that made a
line-based tool write a duplicate ``enabled:`` key and break the whole file).
"""

import pytest

import hermes_yaml as yaml

DOC = (
    "plugins:\n"
    "  clone_timeout_seconds: 300  # eol\n"
    "  hosts:\n"
    "    - a\n"
    "\n"
    "# ==== Model Configuration ====\n"
    "model:\n"
    "  default: some-model\n"
)
BANNER = "\n\n# ==== Model Configuration ====\nmodel:\n"


def _save(path, mutate):
    from utils import atomic_roundtrip_yaml_save

    state = yaml.safe_load(DOC)
    mutate(state)
    atomic_roundtrip_yaml_save(path, state)
    return state


def _update(path, key, value):
    from utils import atomic_roundtrip_yaml_update

    atomic_roundtrip_yaml_update(path, key, value)


@pytest.mark.parametrize("write", [
    pytest.param(lambda p: _save(p, lambda s: s["plugins"].__setitem__("enabled", ["foo"])), id="save-new-key"),
    pytest.param(lambda p: _save(p, lambda s: s["plugins"]["hosts"].append("b")), id="save-list-append"),
    pytest.param(lambda p: _save(p, lambda s: s["plugins"].__setitem__("hosts", "b")), id="save-list-to-scalar"),
    pytest.param(lambda p: _update(p, "plugins.enabled", ["foo"]), id="update-new-key"),
    pytest.param(lambda p: _update(p, "plugins.entries.foo.settings.mode", "x"), id="update-new-branch"),
])
def test_added_entry_lands_above_the_next_sections_banner(tmp_path, write):
    path = tmp_path / "config.yaml"
    path.write_text(DOC, encoding="utf-8")

    write(path)

    text = path.read_text(encoding="utf-8-sig")
    assert text.count("# ==== Model Configuration ====") == 1, text
    assert BANNER in text, text
    assert "clone_timeout_seconds: 300  # eol\n" in text, text
    # Everything the writer added sits inside plugins:, before the banner.
    plugins_block = text.split(BANNER, 1)[0]
    assert yaml.safe_load(plugins_block)["plugins"] == yaml.safe_load(text)["plugins"]


@pytest.mark.parametrize("parent", ["null", "casual"])
def test_new_branch_under_a_null_or_scalar_section_keeps_its_place(tmp_path, parent):
    """Setting ``display.personality`` over ``display: null`` turns that entry into a mapping where
    it stands; it must not move ``display`` below ``model``."""
    path = tmp_path / "config.yaml"
    tail = "\n# ==== Model Configuration ====\nmodel:\n  default: m\n"
    path.write_text(f"display: {parent}\n" + tail, encoding="utf-8")

    _update(path, "display.personality", "kawaii")

    assert path.read_text(encoding="utf-8-sig") == "display:\n  personality: kawaii\n" + tail


def test_plugin_selection_publication_keeps_the_next_sections_banner(tmp_path, monkeypatch):
    """``hermes plugins enable`` commits through the PM publication writer, not save_config."""
    from pm.publication import PluginSelection

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(DOC, encoding="utf-8")

    proposed = PluginSelection({"home": str(tmp_path), "enabled": ["foo"], "disabled": []}).proposed.decode()

    assert BANNER in proposed, proposed
    assert yaml.safe_load(proposed.split(BANNER, 1)[0])["plugins"]["enabled"] == ["foo"]


_EVICTION_TAIL = "\n# ==== Model Configuration ====\nmodel:\n  default: some-model\n"


@pytest.mark.parametrize(("plugins_block", "rejoin", "expected"), [
    pytest.param("  enabled:\n    - foo\n    - bar\n", False,
                 {"enabled": ["foo"], "disabled": ["bar"]}, id="park-last-enabled"),
    pytest.param("  enabled:\n    - bar\n  disabled:\n    - baz\n", False,
                 {"enabled": [], "disabled": ["baz", "bar"]}, id="append-to-disabled"),
    pytest.param("  enabled:\n    - foo\n  disabled:\n    - bar\n"
                 "  disabled_by_update:\n    bar:\n      enabled: true\n      reason: x\n", True,
                 {"enabled": ["foo", "bar"], "disabled": []}, id="rejoin-empties-disabled"),
])
def test_update_plugin_eviction_keeps_the_next_sections_banner(tmp_path, plugins_block, rejoin, expected):
    """``hermes update`` parks (and later restores) a plugin that no longer builds through its own
    writer; that edit must not move, drop or corrupt the banner that follows ``plugins:``."""
    from pm.plugin_eviction import PluginEviction

    (tmp_path / "plugins" / "bar").mkdir(parents=True)
    (tmp_path / "config.yaml").write_text("plugins:\n" + plugins_block + _EVICTION_TAIL, encoding="utf-8")
    entry = (tmp_path / "plugins", "bar", tmp_path / "plugins" / "bar")

    if rejoin:
        eviction = PluginEviction([], {}, [entry], configs={})
    else:
        eviction = PluginEviction([entry], {entry[2].resolve(): "requires Python >=3.99"}, configs={})
    proposed = eviction.edits[0][2].decode()

    assert proposed.count("# ==== Model Configuration ====") == 1, proposed
    assert "\n\n# ==== Model Configuration ====\nmodel:\n" in proposed, proposed
    above = yaml.safe_load(proposed.split("# ==== Model Configuration ====", 1)[0])["plugins"]
    assert {key: above[key] for key in expected} == expected
    assert above == yaml.safe_load(proposed)["plugins"]
