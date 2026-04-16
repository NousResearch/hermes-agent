"""Tests for skill content size limits.

Agent writes (create/edit/patch/write_file) are constrained to
MAX_SKILL_CONTENT_CHARS (100k) and MAX_SKILL_FILE_BYTES (1 MiB).
Hand-placed and hub-installed skills have no hard limit.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import (
    MAX_SKILL_CONTENT_CHARS,
    MAX_SKILL_FILE_BYTES,
    _validate_content_size,
    skill_manage,
)


@pytest.fixture(autouse=True)
def isolate_skills(tmp_path, monkeypatch):
    """Redirect SKILLS_DIR to a temp directory."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", skills_dir)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return skills_dir


def _make_skill_content(body_chars: int) -> str:
    """Generate valid SKILL.md content with a body of the given character count."""
    frontmatter = (
        "---\n"
        "name: test-skill\n"
        "description: A test skill\n"
        "---\n"
    )
    body = "# Test Skill\n\n" + ("x" * max(0, body_chars - 15))
    return frontmatter + body


class TestValidateContentSize:
    """Unit tests for _validate_content_size."""

    def test_within_limit(self):
        assert _validate_content_size("a" * 1000) is None

    def test_at_limit(self):
        assert _validate_content_size("a" * MAX_SKILL_CONTENT_CHARS) is None

    def test_over_limit(self):
        err = _validate_content_size("a" * (MAX_SKILL_CONTENT_CHARS + 1))
        assert err is not None
        assert "100,001" in err
        assert "100,000" in err

    def test_custom_label(self):
        err = _validate_content_size("a" * (MAX_SKILL_CONTENT_CHARS + 1), label="references/api.md")
        assert "references/api.md" in err


class TestCreateSkillSizeLimit:
    """create action rejects oversized content."""

    def test_create_within_limit(self, isolate_skills):
        content = _make_skill_content(5000)
        result = json.loads(skill_manage(action="create", name="small-skill", content=content))
        assert result["success"] is True

    def test_create_over_limit(self, isolate_skills):
        content = _make_skill_content(MAX_SKILL_CONTENT_CHARS + 100)
        result = json.loads(skill_manage(action="create", name="huge-skill", content=content))
        assert result["success"] is False
        assert "100,000" in result["error"]

    def test_create_at_limit(self, isolate_skills):
        # Content at exactly the limit should succeed
        frontmatter = "---\nname: edge-skill\ndescription: Edge case\n---\n# Edge\n\n"
        body_budget = MAX_SKILL_CONTENT_CHARS - len(frontmatter)
        content = frontmatter + ("x" * body_budget)
        assert len(content) == MAX_SKILL_CONTENT_CHARS
        result = json.loads(skill_manage(action="create", name="edge-skill", content=content))
        assert result["success"] is True


class TestEditSkillSizeLimit:
    """edit action rejects oversized content."""

    def test_edit_over_limit(self, isolate_skills):
        # Create a small skill first
        small = _make_skill_content(1000)
        json.loads(skill_manage(action="create", name="grow-me", content=small))

        # Try to edit it to be oversized
        big = _make_skill_content(MAX_SKILL_CONTENT_CHARS + 100)
        # Fix the name in frontmatter
        big = big.replace("name: test-skill", "name: grow-me")
        result = json.loads(skill_manage(action="edit", name="grow-me", content=big))
        assert result["success"] is False
        assert "100,000" in result["error"]


class TestPatchSkillSizeLimit:
    """patch action checks resulting size, not just the new_string."""

    def test_patch_that_would_exceed_limit(self, isolate_skills):
        # Create a skill near the limit
        near_limit = _make_skill_content(MAX_SKILL_CONTENT_CHARS - 50)
        json.loads(skill_manage(action="create", name="near-limit", content=near_limit))

        # Patch that adds enough to go over
        result = json.loads(skill_manage(
            action="patch",
            name="near-limit",
            old_string="# Test Skill",
            new_string="# Test Skill\n" + ("y" * 200),
        ))
        assert result["success"] is False
        assert "100,000" in result["error"]

    def test_patch_that_reduces_size_on_oversized_skill(self, isolate_skills, tmp_path):
        """Patches that shrink an already-oversized skill should succeed."""
        # Manually create an oversized skill (simulating hand-placed)
        skill_dir = tmp_path / "skills" / "bloated"
        skill_dir.mkdir(parents=True)
        oversized = _make_skill_content(MAX_SKILL_CONTENT_CHARS + 5000)
        oversized = oversized.replace("name: test-skill", "name: bloated")
        (skill_dir / "SKILL.md").write_text(oversized, encoding="utf-8")
        assert len(oversized) > MAX_SKILL_CONTENT_CHARS

        # Patch that removes content to bring it under the limit.
        # Use replace_all to replace the repeated x's with a shorter string.
        result = json.loads(skill_manage(
            action="patch",
            name="bloated",
            old_string="x" * 100,
            new_string="y",
            replace_all=True,
        ))
        # Should succeed because the result is well within limits
        assert result["success"] is True

    def test_patch_supporting_file_size_limit(self, isolate_skills):
        """Patch on a supporting file also checks size."""
        small = _make_skill_content(1000)
        json.loads(skill_manage(action="create", name="with-ref", content=small))
        # Create a supporting file
        json.loads(skill_manage(
            action="write_file",
            name="with-ref",
            file_path="references/data.md",
            file_content="# Data\n\nSmall content.",
        ))
        # Try to patch it to be oversized
        result = json.loads(skill_manage(
            action="patch",
            name="with-ref",
            old_string="Small content.",
            new_string="x" * (MAX_SKILL_CONTENT_CHARS + 100),
            file_path="references/data.md",
        ))
        assert result["success"] is False
        assert "references/data.md" in result["error"]


class TestWriteFileSizeLimit:
    """write_file action enforces both char and byte limits."""

    def test_write_file_over_char_limit(self, isolate_skills):
        small = _make_skill_content(1000)
        json.loads(skill_manage(action="create", name="file-test", content=small))

        result = json.loads(skill_manage(
            action="write_file",
            name="file-test",
            file_path="references/huge.md",
            file_content="x" * (MAX_SKILL_CONTENT_CHARS + 1),
        ))
        assert result["success"] is False
        assert "100,000" in result["error"]

    def test_write_file_within_limit(self, isolate_skills):
        small = _make_skill_content(1000)
        json.loads(skill_manage(action="create", name="file-ok", content=small))

        result = json.loads(skill_manage(
            action="write_file",
            name="file-ok",
            file_path="references/normal.md",
            file_content="# Normal\n\n" + ("x" * 5000),
        ))
        assert result["success"] is True


class TestHandPlacedSkillsNoLimit:
    """Skills dropped directly on disk are not constrained."""

    def test_oversized_handplaced_skill_loads(self, isolate_skills, tmp_path):
        """A hand-placed 200k skill can still be read via skill_view."""
        from tools.skills_tool import skill_view

        skill_dir = tmp_path / "skills" / "manual-giant"
        skill_dir.mkdir(parents=True)
        huge = _make_skill_content(200_000)
        huge = huge.replace("name: test-skill", "name: manual-giant")
        (skill_dir / "SKILL.md").write_text(huge, encoding="utf-8")

        result = json.loads(skill_view("manual-giant"))
        assert "content" in result
        # The full content is returned — no truncation at the storage layer
        assert len(result["content"]) > MAX_SKILL_CONTENT_CHARS
