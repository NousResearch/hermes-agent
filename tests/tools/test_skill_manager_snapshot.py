"""Tests for skill manager snapshot preservation and validation."""

import json
import os
from unittest.mock import patch


VALID_SKILL_CONTENT = """\
---
name: snapshot-skill
description: A test skill for snapshot cache behavior.
---

# Snapshot Skill

Use this skill to test cache invalidation.
"""


def test_skill_manage_preserves_snapshot(tmp_path):
    from tools.skill_manager_tool import skill_manage

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    with patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir), \
         patch("tools.skill_manager_tool._security_scan_skill", return_value=None), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]), \
         patch("agent.prompt_builder.clear_skills_system_prompt_cache") as clear_cache:
        result = json.loads(
            skill_manage(
                action="create",
                name="snapshot-skill",
                content=VALID_SKILL_CONTENT,
            )
        )

    assert result["success"] is True
    clear_cache.assert_called_once_with(clear_snapshot=False)


def test_snapshot_manifest_detects_mtime_change(tmp_path):
    from agent import prompt_builder

    hermes_home = tmp_path / "hermes-home"
    skills_dir, skill_md = _create_skill_file(tmp_path)
    snapshot = _snapshot_for(skills_dir, skill_md)

    with patch.object(prompt_builder, "get_hermes_home", return_value=hermes_home):
        _write_snapshot(prompt_builder, snapshot)

        stat = skill_md.stat()
        os.utime(
            skill_md,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000),
        )
        assert skill_md.stat().st_mtime_ns != snapshot["manifest"][str(skill_md.relative_to(skills_dir))][0]

        assert prompt_builder._load_skills_snapshot(skills_dir) is None


def test_snapshot_manifest_valid_when_unchanged(tmp_path):
    from agent import prompt_builder

    hermes_home = tmp_path / "hermes-home"
    skills_dir, skill_md = _create_skill_file(tmp_path)
    snapshot = _snapshot_for(skills_dir, skill_md)

    with patch.object(prompt_builder, "get_hermes_home", return_value=hermes_home):
        _write_snapshot(prompt_builder, snapshot)

        assert prompt_builder._load_skills_snapshot(skills_dir) == snapshot


def _create_skill_file(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "snapshot-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(VALID_SKILL_CONTENT, encoding="utf-8")
    return skills_dir, skill_md


def _snapshot_for(skills_dir, skill_md):
    stat = skill_md.stat()
    return {
        "version": 1,
        "manifest": {
            str(skill_md.relative_to(skills_dir)): [stat.st_mtime_ns, stat.st_size],
        },
        "skills": [
            {
                "name": "snapshot-skill",
                "description": "A test skill for snapshot cache behavior.",
                "path": str(skill_md),
            }
        ],
        "category_descriptions": {},
    }


def _write_snapshot(prompt_builder, snapshot):
    snapshot_path = prompt_builder._skills_prompt_snapshot_path()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
