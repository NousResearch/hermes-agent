"""`usage_report()` covers plugin-provided skills, which live outside the skills directory."""

from pathlib import Path

import pytest


@pytest.fixture
def usage_mod(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills" / "local-skill").mkdir(parents=True)
    (home / "skills" / "local-skill" / "SKILL.md").write_text("---\nname: local-skill\ndescription: d\n---\nbody\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib
    import hermes_agent.tools.skill_usage as mod
    importlib.reload(mod)
    return mod


def test_usage_report_lists_plugin_skills_with_their_counts(usage_mod, tmp_path, monkeypatch):
    from hermes_cli import plugins
    md = tmp_path / "plug" / "skills" / "model-3d" / "SKILL.md"
    md.parent.mkdir(parents=True)
    md.write_text("---\nname: model-3d\ndescription: d\n---\nbody\n")
    manager = plugins.PluginManager()
    manager._plugin_skills["meshlike:model-3d"] = {
        "path": md, "plugin": "meshlike", "bare_name": "model-3d", "description": "d"}
    monkeypatch.setattr(plugins, "discover_plugins", lambda force=False: None)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    usage_mod.bump_use("meshlike:model-3d")
    usage_mod.bump_use("meshlike:model-3d")

    rows = {r["name"]: r for r in usage_mod.usage_report()}
    assert rows["meshlike:model-3d"]["use_count"] == 2
    assert rows["meshlike:model-3d"]["provenance"] == "plugin"
    assert rows["local-skill"]["provenance"] == "agent"
    # A plugin skill is never handed to the curator: listing it must not make it a managed row.
    assert "meshlike:model-3d" not in {r["name"] for r in usage_mod.curated_report()}
