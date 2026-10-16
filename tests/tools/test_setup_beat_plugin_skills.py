"""The build beat names a plugin's skill by the name it registered under.

A first task writes a plugin skill readably (``blender:blender``), but a portable plugin's skills
register under a collision-resistant namespace (``agent-plugin-blender-<sha8>:blender``). In a live run
the build beat sent the readable form, ``skill_view("blender:blender")`` missed, and the model rendered
headless without the skill. Real plugin discovery against a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json

import pytest

from agent import onboarding_flow as flow
from hermes_cli import plugins as plugins_mod
from tools.setup_beat_tool import _result

PC = flow.Facts(os="windows", arch="amd64", machine_kind="PC", nvidia_platform="discrete")


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "plugins").mkdir(parents=True)
    (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["blender"]}}))
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    plugins_mod._reset_plugin_managers_for_tests()
    yield home
    plugins_mod._reset_plugin_managers_for_tests()


def _install_blender_from_catalog(home):
    """What a catalog install leaves: the portable plugin tree and the installer-owned record."""
    from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1

    plugin = home / "plugins" / "blender"
    (plugin / "skills" / "blender").mkdir(parents=True)
    (plugin / "plugin.json").write_text(json.dumps({"$schema": PLUGIN_SCHEMA_V1, "name": "blender"}))
    (plugin / "skills" / "blender" / "SKILL.md").write_text(
        "---\nname: blender\ndescription: Drive Blender.\n---\nBody.\n")
    (home / "plugins" / ".install-metadata.json").write_text(json.dumps({"blender": {
        "source": "https://example.com/blender.git",
        "catalog": {"name": "blender", "repo": "https://example.com/blender.git", "sha": "0" * 40}}}))


def _build_skills_line() -> str:
    record = flow.new_record(PC)
    record["picks"]["installed"] = ["blender"]
    reminder = json.loads(_result(flow.Step(record=record, result={}, reminder="build_blender")))["reminder"]
    return next(line for line in reminder.split("\n") if line.startswith("Load these skills"))


def test_the_build_beat_names_an_installed_plugin_skill_by_its_registered_name(home):
    _install_blender_from_catalog(home)
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load(force=True)
    [registered] = [skill["name"] for skill in manager.list_plugin_skill_metadata()]
    assert registered.startswith("agent-plugin-blender-") and registered.endswith(":blender")

    line = _build_skills_line()

    assert registered in line
    assert "blender:blender" not in line.replace(registered, "")


def test_a_plugin_skill_that_is_not_installed_is_dropped(home):
    plugins_mod.get_plugin_manager().discover_and_load(force=True)

    assert "blender" not in _build_skills_line()
