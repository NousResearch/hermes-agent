import json

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from tools.skills_tool import skill_view


def test_area_leaf_name_resolves_to_plugin_leaf(tmp_path, monkeypatch):
    pm = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", pm)
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path / "empty")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    area = tmp_path / "messenger"
    (area / "kakao-talk").mkdir(parents=True)
    (area / "SKILL.md").write_text("---\nname: messenger\ndescription: router\n---\nrouter\n")
    (area / "kakao-talk" / "SKILL.md").write_text("---\nname: kakao-talk\ndescription: leaf\n---\nLEAF-BODY\n")
    pm._plugin_skills["bs:messenger"] = {"path": area / "SKILL.md", "plugin": "bs", "bare_name": "messenger", "description": ""}

    out = json.loads(skill_view("messenger/kakao-talk"))
    assert out["success"] is True and "LEAF-BODY" in out["content"]
    assert json.loads(skill_view("messenger/missing"))["success"] is False
