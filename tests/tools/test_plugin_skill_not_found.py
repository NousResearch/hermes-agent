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
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: tmp_path / "no-bundled")
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
@pytest.mark.parametrize("local_count", [0, 25])
def test_not_found_names_a_loadable_plugin_even_after_twenty_local_skills(installed_plugin, query, local_count):
    from tools.registry import registry
    from tools import skills_tool  # registers the real tool handler

    local, namespace = installed_plugin
    for index in range(local_count):
        folder = local / f"aaa-{index:02}"
        folder.mkdir()
        (folder / "SKILL.md").write_text(f"---\nname: {folder.name}\n---\nLocal.\n", encoding="utf-8")
    if not local_count:
        local.rmdir()
    skills_tool._SKILLS_CACHE.clear()
    result = json.loads(registry.dispatch("skill_view", {"name": query}))
    qualified = f"{namespace}:example"
    assert result["success"] is False
    assert qualified in result.get("available_skills", [])
    loaded = json.loads(registry.dispatch("skill_view", {"name": qualified}))
    assert loaded["success"] is True
    assert "Plugin content." in loaded["content"]


def test_local_resolution_and_missing_result_survive_plugin_listing_failure(installed_plugin, monkeypatch):
    from hermes_cli import plugins
    from tools.skills_tool import skill_view

    local, namespace = installed_plugin
    manager = plugins.get_plugin_manager()
    from hermes_cli.plugins import PluginContext, PluginManifest
    context = PluginContext(PluginManifest(name=namespace, version="1.0.0", description="fixture", source="user"), manager)
    skill_md = manager.find_plugin_skill(f"{namespace}:example")
    for index in range(25):
        context.register_skill(f"aaa-{index:02}", skill_md)
    suggestions = json.loads(skill_view("example", preprocess=False))["available_skills"]
    assert suggestions[0] == f"{namespace}:example"
    assert len(suggestions) <= 20
    monkeypatch.setattr("tools.skills_tool._is_skill_disabled", lambda name: ":" in name)
    assert not json.loads(skill_view("absent", preprocess=False))["available_skills"]
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
