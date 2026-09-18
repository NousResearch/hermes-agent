"""Regression tests for exact-name recovery of archived skills."""

import json
from pathlib import Path
from unittest.mock import patch

from tools.skills_tool import skill_view, skills_list


def _make_archived_skill(skills_dir: Path, name: str) -> None:
    skill_dir = skills_dir / ".archive" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Archived test skill.\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def test_archived_skill_miss_includes_recovery_actions(tmp_path: Path) -> None:
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_archived_skill(tmp_path, "remember-me")

        result = json.loads(skill_view("remember-me"))
        listed = json.loads(skills_list())
        inspected = json.loads(skill_view(result["archived_path"]))

    assert result["success"] is False
    assert result["archived"] is True
    assert result["archived_path"] == ".archive/remember-me"
    assert result["restore_command"] == "hermes curator restore remember-me"
    assert 'skill_view(name=".archive/remember-me")' in result["hint"]
    assert "remember-me" not in {skill["name"] for skill in listed["skills"]}
    assert inspected["success"] is True


def test_unknown_skill_keeps_generic_active_catalog_hint(tmp_path: Path) -> None:
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_archived_skill(tmp_path, "-unsafe;name")
        result = json.loads(skill_view("never-installed"))
        unsafe = json.loads(skill_view("-unsafe;name"))

    assert result["success"] is False
    assert result["hint"] == "Use skills_list to see all available skills"
    assert "archived" not in result
    assert unsafe["hint"] == "Use skills_list to see all available skills"
    assert "restore_command" not in unsafe
