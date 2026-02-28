"""Tests for path traversal prevention in skill_view."""

import json
import sys
import types
import pytest
from pathlib import Path
from unittest.mock import patch

# Stub out heavy optional dependencies so tools/__init__.py doesn't blow up
from unittest.mock import MagicMock
for mod_name in ("firecrawl", "fal_client", "tavily"):
    sys.modules.setdefault(mod_name, MagicMock())

from tools.skills_tool import skill_view


@pytest.fixture()
def fake_skills(tmp_path):
    """Create a fake skills directory with one skill and a sensitive file outside."""
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "test-skill"
    skill_dir.mkdir(parents=True)

    # Create SKILL.md
    (skill_dir / "SKILL.md").write_text("# Test Skill\nA test skill.")

    # Create a legitimate file inside the skill
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "api.md").write_text("API docs here")

    # Create a sensitive file outside skills dir (simulating .env)
    (tmp_path / ".env").write_text("SECRET_API_KEY=sk-12345")

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        yield {"skills_dir": skills_dir, "skill_dir": skill_dir, "tmp_path": tmp_path}


class TestPathTraversalBlocked:
    def test_dotdot_in_file_path(self, fake_skills):
        result = json.loads(skill_view("test-skill", file_path="../../.env"))
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    def test_dotdot_nested(self, fake_skills):
        result = json.loads(skill_view("test-skill", file_path="references/../../.env"))
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    def test_legitimate_file_still_works(self, fake_skills):
        result = json.loads(skill_view("test-skill", file_path="references/api.md"))
        assert result["success"] is True
        assert "API docs here" in result["content"]

    def test_no_file_path_shows_skill(self, fake_skills):
        result = json.loads(skill_view("test-skill"))
        assert result["success"] is True
