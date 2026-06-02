"""Tests for tools/skills_list.py — enriched L0 desktop catalog."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_skills(tmp_path, monkeypatch):
    """Point the bundled-skills resolver at an empty temp dir + isolate caches."""
    bundled = tmp_path / "bundled-empty"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS_DIR", str(bundled))
    yield tmp_path


def _write_skill(root: Path, name: str, frontmatter_extra: str = "", body: str = "Body text") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = skill_dir / "SKILL.md"
    fm = f"name: {name}\ndescription: Desc for {name}.\n"
    if frontmatter_extra:
        fm += frontmatter_extra
    md.write_text(f"---\n{fm}---\n\n{body}\n", encoding="utf-8")
    return md


def _call_list_skills(local_dir: Path, *, registered_toolsets=None, plugin_skills=None):
    from tools import skills_list as mod

    registered_toolsets = registered_toolsets or []
    plugin_skills = plugin_skills or {}

    class _FakeRegistry:
        def get_registered_toolset_names(self):
            return list(registered_toolsets)

        def get_registered_toolset_aliases(self):
            return {}

    class _FakePM:
        _plugin_skills = plugin_skills

    with patch.object(mod, "_local_skills_dir", return_value=local_dir), \
         patch.object(mod, "_external_skills_dirs", return_value=[]), \
         patch.object(mod, "_bundled_skills_dirs", return_value=[]), \
         patch.object(mod, "_disabled_names", return_value=set()), \
         patch.object(mod, "_registered_toolsets",
                      lambda: set(registered_toolsets)), \
         patch("hermes_cli.plugins.get_plugin_manager", return_value=_FakePM(), create=True), \
         patch("hermes_cli.plugins._get_disabled_plugins", return_value=[], create=True):
        return json.loads(mod.list_skills())


def test_l0_shape_includes_required_fields(isolated_skills):
    local = isolated_skills / "local"
    local.mkdir()
    _write_skill(local, "alpha")

    result = _call_list_skills(local)
    assert result["success"] is True
    assert result["sources"] == ["bundled", "plugin", "user"]
    assert result["count"] == 1
    entry = result["skills"][0]
    for field in (
        "id", "slug", "name", "description",
        "source_badge", "requires_toolsets", "fallback_for_toolsets",
    ):
        assert field in entry
    assert entry["source_badge"] == "user"
    assert entry["slug"] == "alpha"
    assert entry["requires_toolsets"] == []
    assert entry["fallback_for_toolsets"] == []


def test_requires_toolsets_hides_skill_when_gate_missing(isolated_skills):
    local = isolated_skills / "local"
    local.mkdir()
    _write_skill(
        local,
        "messaging-triage",
        frontmatter_extra=(
            "metadata:\n"
            "  hermes:\n"
            "    requires_toolsets: [messaging]\n"
        ),
    )

    # No messaging toolset registered → must be hidden.
    hidden = _call_list_skills(local, registered_toolsets=[])
    assert hidden["count"] == 0

    # When the messaging toolset is registered → visible.
    visible = _call_list_skills(local, registered_toolsets=["messaging"])
    assert visible["count"] == 1
    assert visible["skills"][0]["requires_toolsets"] == ["messaging"]


def test_three_source_categories_present(isolated_skills, tmp_path):
    from tools import skills_list as mod

    local = tmp_path / "user"
    local.mkdir()
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    _write_skill(local, "user-skill")
    _write_skill(bundled, "bundled-skill")

    plugin_path = tmp_path / "plugin-skill" / "SKILL.md"
    plugin_path.parent.mkdir()
    plugin_path.write_text(
        "---\nname: plugin-skill\ndescription: From plugin.\n---\n\nbody\n",
        encoding="utf-8",
    )
    plugin_skills = {
        "demo:plugin-skill": {
            "path": plugin_path,
            "plugin": "demo",
            "bare_name": "plugin-skill",
            "description": "From plugin.",
        }
    }

    class _FakeRegistry:
        def get_registered_toolset_names(self):
            return []

        def get_registered_toolset_aliases(self):
            return {}

    class _FakePM:
        _plugin_skills = plugin_skills

    with patch.object(mod, "_local_skills_dir", return_value=local), \
         patch.object(mod, "_external_skills_dirs", return_value=[]), \
         patch.object(mod, "_bundled_skills_dirs", return_value=[bundled]), \
         patch.object(mod, "_disabled_names", return_value=set()), \
         patch.object(mod, "_registered_toolsets", lambda: set()), \
         patch("hermes_cli.plugins.get_plugin_manager", return_value=_FakePM(), create=True), \
         patch("hermes_cli.plugins._get_disabled_plugins", return_value=[], create=True):
        result = json.loads(mod.list_skills())

    badges = sorted({e["source_badge"] for e in result["skills"]})
    assert badges == ["bundled", "plugin", "user"]


def test_registered_in_inventory():
    """The tool must appear in HERMES_TOOL_INVENTORY (toolsets.py + model_tools)."""
    from toolsets import _HERMES_CORE_TOOLS, TOOLSETS
    from model_tools import _LEGACY_TOOLSET_MAP

    assert "list_skills" in _HERMES_CORE_TOOLS
    assert "list_skills" in TOOLSETS["skills"]["tools"]
    assert "list_skills" in _LEGACY_TOOLSET_MAP["skills_tools"]


def test_handler_registered_in_registry():
    """Registry-side check: importing the module registers the handler."""
    import tools.skills_list  # noqa: F401 - import side-effect registers
    from tools.registry import registry

    entry = registry.get_entry("list_skills")
    assert entry is not None
    assert entry.toolset == "skills"
