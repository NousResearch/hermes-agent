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

        # An EMPTY pre-existing directory (leftover of an earlier failed create) is a valid target,
        # but a blocked scan removes only what create wrote and leaves the directory in place.
        empty = tmp_path / "empty-skill"
        empty.mkdir()
        empty_result = _create_skill("empty-skill", VALID_CONTENT)

    assert result["success"] is False
    assert data.read_bytes() == b"keep me"
    assert target.is_dir()
    assert not (target / "SKILL.md").exists()

    assert empty_result["success"] is False
    assert empty.is_dir()
    assert not any(empty.iterdir())


def test_category_directory_collision_is_refused(tmp_path):
    with _skill_dir(tmp_path), patch("tools.skill_manager_tool._security_scan_skill", return_value=None):
        category = tmp_path / "category"
        category.mkdir()
        nested = category / "nested-skill.md"
        nested.write_bytes(b"nested")
        result = _create_skill("category", VALID_CONTENT)

        # Empty pre-existing directory + clean scan: create succeeds (retry after a leftover works).
        empty = tmp_path / "empty-skill"
        empty.mkdir()
        empty_result = _create_skill("empty-skill", VALID_CONTENT)

    assert result["success"] is False
    assert nested.read_bytes() == b"nested"
    assert not (category / "SKILL.md").exists()

    assert empty_result["success"] is True
    assert (empty / "SKILL.md").exists()
