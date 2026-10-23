"""Test that _find_skill searches project-local skill dirs.

Bug: _find_skill only called get_all_skills_dirs() which explicitly excludes
project dirs (see its docstring). skill_manage(patch/write_file/delete) would
answer "Skill '<name>' not found" for every .hermes/skills project skill,
while skill_view saw them fine.

Fix: _find_skill now prepends get_project_skills_dirs() (higher precedence)
before get_all_skills_dirs(), mirroring skill_commands.py and prompt_builder.py.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import _find_skill, _patch_skill


MINIMAL_SKILL = """\
---
name: proj-skill
description: A project-local skill for testing.
---

# Proj Skill

Step 1: Do the thing.
"""


class TestFindSkillIncludesProjectDirs:
    """_find_skill must search get_project_skills_dirs() with higher precedence."""

    def test_find_skill_finds_project_skill(self, tmp_path):
        """A skill that lives only in a project dir must be found by _find_skill."""
        proj_skills = tmp_path / "project" / ".hermes" / "skills"
        proj_skills.mkdir(parents=True)
        skill_dir = proj_skills / "proj-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(MINIMAL_SKILL, encoding="utf-8")

        profile_skills = tmp_path / "profile" / "skills"
        profile_skills.mkdir(parents=True)

        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", profile_skills),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[profile_skills]),
            patch("agent.skill_utils.get_project_skills_dirs", return_value=[proj_skills]),
            patch("tools.skill_manager_tool.get_project_skills_dirs", return_value=[proj_skills]),
        ):
            result = _find_skill("proj-skill")

        assert result is not None, "project skill must be found by _find_skill"
        assert result["path"] == skill_dir

    def test_find_skill_project_dir_takes_precedence_over_profile(self, tmp_path):
        """When the same name exists in both project and profile dirs, project wins."""
        proj_skills = tmp_path / "project" / ".hermes" / "skills"
        proj_skills.mkdir(parents=True)
        proj_skill_dir = proj_skills / "shared-skill"
        proj_skill_dir.mkdir()
        (proj_skill_dir / "SKILL.md").write_text(
            MINIMAL_SKILL.replace("proj-skill", "shared-skill").replace("project-local", "project version"),
            encoding="utf-8",
        )

        profile_skills = tmp_path / "profile" / "skills"
        profile_skills.mkdir(parents=True)
        profile_skill_dir = profile_skills / "shared-skill"
        profile_skill_dir.mkdir()
        (profile_skill_dir / "SKILL.md").write_text(
            MINIMAL_SKILL.replace("proj-skill", "shared-skill").replace("project-local", "profile version"),
            encoding="utf-8",
        )

        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", profile_skills),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[profile_skills]),
            patch("agent.skill_utils.get_project_skills_dirs", return_value=[proj_skills]),
            patch("tools.skill_manager_tool.get_project_skills_dirs", return_value=[proj_skills]),
        ):
            result = _find_skill("shared-skill")

        assert result is not None
        assert result["path"] == proj_skill_dir, "project dir must shadow same-named profile skill"

    def test_patch_succeeds_on_project_skill(self, tmp_path):
        """End-to-end: skill_manage(action='patch') must succeed on a project skill.

        Before the fix this raised "Skill 'proj-skill' not found in active profile"
        because _find_skill never searched the project dirs.
        """
        proj_skills = tmp_path / "project" / ".hermes" / "skills"
        proj_skills.mkdir(parents=True)
        skill_dir = proj_skills / "proj-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(MINIMAL_SKILL, encoding="utf-8")

        profile_skills = tmp_path / "profile" / "skills"
        profile_skills.mkdir(parents=True)

        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", profile_skills),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[profile_skills]),
            patch("agent.skill_utils.get_project_skills_dirs", return_value=[proj_skills]),
            patch("tools.skill_manager_tool.get_project_skills_dirs", return_value=[proj_skills]),
        ):
            result = _patch_skill("proj-skill", "Do the thing.", "Do the patched thing.")

        assert result.get("success") is True, (
            f"patch on project skill should succeed but got: {result}"
        )
        updated = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "Do the patched thing." in updated

    def test_find_skill_returns_none_when_no_project_dirs(self, tmp_path):
        """When get_project_skills_dirs() returns [], _find_skill degrades gracefully."""
        profile_skills = tmp_path / "profile" / "skills"
        profile_skills.mkdir(parents=True)

        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", profile_skills),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[profile_skills]),
            patch("agent.skill_utils.get_project_skills_dirs", return_value=[]),
            patch("tools.skill_manager_tool.get_project_skills_dirs", return_value=[]),
        ):
            result = _find_skill("nonexistent-skill")

        assert result is None
