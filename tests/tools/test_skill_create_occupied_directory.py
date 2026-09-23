"""Ownership contracts for creating skills in occupied paths."""

from contextlib import contextmanager
from unittest.mock import patch

from tools.skill_manager_tool import _create_skill


VALID_CONTENT = """\
---
name: new-skill
description: Use when testing skill creation.
---

# New Skill

Create the thing.
"""


@contextmanager
def _skill_dir(tmp_path):
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


def test_occupied_directory_is_preserved_when_scan_blocks(tmp_path):
    with _skill_dir(tmp_path), patch("tools.skill_manager_tool._security_scan_skill", return_value="blocked"):
        target = tmp_path / "new-skill"
        target.mkdir()
        data = target / "data.bin"
        data.write_bytes(b"keep me")
        result = _create_skill("new-skill", VALID_CONTENT)

    assert result["success"] is False
    assert data.read_bytes() == b"keep me"
    assert target.is_dir()
    assert not (target / "SKILL.md").exists()


def test_occupied_directory_is_preserved_when_scan_does_not_block(tmp_path):
    with _skill_dir(tmp_path), patch("tools.skill_manager_tool._security_scan_skill", return_value=None):
        target = tmp_path / "new-skill"
        target.mkdir()
        data = target / "data.bin"
        data.write_bytes(b"keep me")
        result = _create_skill("new-skill", VALID_CONTENT)

    assert result["success"] is False
    assert data.read_bytes() == b"keep me"
    assert target.is_dir()
    assert not (target / "SKILL.md").exists()


def test_preexisting_skill_file_is_preserved_when_index_does_not_find_it(tmp_path):
    with _skill_dir(tmp_path), patch("tools.skill_manager_tool._find_skill", return_value=None):
        target = tmp_path / "new-skill"
        target.mkdir()
        skill_md = target / "SKILL.md"
        skill_md.write_bytes(b"original")
        result = _create_skill("new-skill", VALID_CONTENT)

    assert result["success"] is False
    assert skill_md.read_bytes() == b"original"


def test_fresh_directory_is_removed_when_scan_blocks(tmp_path):
    with _skill_dir(tmp_path), patch("tools.skill_manager_tool._security_scan_skill", return_value="blocked"):
        result = _create_skill("new-skill", VALID_CONTENT)

    assert result["success"] is False
    assert not (tmp_path / "new-skill").exists()


def test_fresh_directory_is_created_when_scan_does_not_block(tmp_path):
    with _skill_dir(tmp_path), patch("tools.skill_manager_tool._security_scan_skill", return_value=None):
        result = _create_skill("new-skill", VALID_CONTENT)

    assert result["success"] is True
    assert (tmp_path / "new-skill" / "SKILL.md").exists()


def test_category_directory_collision_is_refused(tmp_path):
    with _skill_dir(tmp_path), patch("tools.skill_manager_tool._security_scan_skill", return_value=None):
        category = tmp_path / "category"
        category.mkdir()
        nested = category / "nested-skill.md"
        nested.write_bytes(b"nested")
        result = _create_skill("category", VALID_CONTENT)

    assert result["success"] is False
    assert nested.read_bytes() == b"nested"
    assert not (category / "SKILL.md").exists()


def test_file_occupying_skill_path_is_not_deleted(tmp_path):
    with _skill_dir(tmp_path):
        target = tmp_path / "new-skill"
        target.write_bytes(b"do not delete")
        result = _create_skill("new-skill", VALID_CONTENT)

    assert result["success"] is False
    assert target.read_bytes() == b"do not delete"
