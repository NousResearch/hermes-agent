"""Plugin skills must be recoverable after an unqualified lookup (#119976)."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def installed_plugin(tmp_path, monkeypatch):
    from hermes_cli import plugins
    from tools import skills_tool

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    local = home / "skills"
    local.mkdir()
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", local)
    skills_tool._SKILLS_CACHE.clear()
    namespace = "agent-plugin-example-a1b2c3d4"
    plugin = home / "plugins" / namespace
    skill = plugin / "skills" / "example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: example\ndescription: A fixture.\n---\nPlugin content.\n", encoding="utf-8")
    (plugin / "plugin.yaml").write_text(f"name: {namespace}\nversion: 1.0.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\ndef register(ctx):\n"
        "    ctx.register_skill('example', Path(__file__).parent / 'skills/example/SKILL.md')\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(f"plugins:\n  enabled: [{namespace}]\n", encoding="utf-8")
    plugins.discover_plugins(force=True)
    assert plugins.get_plugin_manager().find_plugin_skill(f"{namespace}:example") == skill
    return local, namespace


@pytest.mark.parametrize("query", ["example", "example:example", "missing"])
def test_not_found_names_a_loadable_plugin_even_after_twenty_local_skills(installed_plugin, query):
    from tools.registry import registry
    from tools import skills_tool  # registers the real tool handler

    local, namespace = installed_plugin
    for index in range(25):
        folder = local / f"aaa-{index:02}"
        folder.mkdir()
        (folder / "SKILL.md").write_text(f"---\nname: {folder.name}\n---\nLocal.\n", encoding="utf-8")
    skills_tool._SKILLS_CACHE.clear()
    result = json.loads(registry.dispatch("skill_view", {"name": query}))
    qualified = f"{namespace}:example"
    assert result["success"] is False
    assert qualified in result["available_skills"]
    loaded = json.loads(registry.dispatch("skill_view", {"name": qualified}))
    assert loaded["success"] is True
    assert "Plugin content." in loaded["content"]


def test_local_resolution_and_missing_result_survive_plugin_listing_failure(installed_plugin, monkeypatch):
    from hermes_cli import plugins
    from tools.skills_tool import skill_view

    local, _ = installed_plugin
    folder = local / "example"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: example\n---\nLocal winner.\n", encoding="utf-8")
    assert "Local winner." in json.loads(skill_view("example", preprocess=False))["content"]
    def unavailable():
        raise RuntimeError("fixture discovery failure")
    monkeypatch.setattr(plugins, "discover_plugins", unavailable)
    result = json.loads(skill_view("absent", preprocess=False))
    assert result["success"] is False
    assert "not found" in result["error"]
    assert "example" in result["available_skills"]
