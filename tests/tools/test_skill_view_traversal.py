"""Tests for path traversal prevention in skill_view.

Regression tests for issue #220: skill_view file_path parameter allowed
reading arbitrary files (e.g., ~/.hermes/.env) via path traversal.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

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
    (tmp_path / ".env").write_text("SECRET_API_KEY=sk-do-not-leak")

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        yield {"skills_dir": skills_dir, "skill_dir": skill_dir, "tmp_path": tmp_path}


class TestPathTraversalBlocked:
    def test_dotdot_in_file_path(self, fake_skills):
        """Direct .. traversal should be rejected."""
        result = json.loads(skill_view("test-skill", file_path="../../.env"))
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    def test_dotdot_nested(self, fake_skills):
        """Nested .. traversal should also be rejected."""
        result = json.loads(skill_view("test-skill", file_path="references/../../../.env"))
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    def test_legitimate_file_still_works(self, fake_skills):
        """Valid paths within the skill directory should work normally."""
        result = json.loads(skill_view("test-skill", file_path="references/api.md"))
        assert result["success"] is True
        assert "API docs here" in result["content"]

    def test_no_file_path_shows_skill(self, fake_skills):
        """Calling skill_view without file_path should return the SKILL.md."""
        result = json.loads(skill_view("test-skill"))
        assert result["success"] is True

    def test_symlink_escape_blocked(self, fake_skills):
        """Symlinks pointing outside the skill directory should be blocked."""
        skill_dir = fake_skills["skill_dir"]
        secret = fake_skills["tmp_path"] / "secret.txt"
        secret.write_text("TOP SECRET DATA")

        symlink = skill_dir / "evil-link"
        try:
            symlink.symlink_to(secret)
        except OSError:
            pytest.skip("Symlinks not supported")

        result = json.loads(skill_view("test-skill", file_path="evil-link"))
        # The resolve() check should catch the symlink escaping
        assert result["success"] is False
        assert "escapes" in result["error"].lower() or "boundary" in result["error"].lower()

    def test_sensitive_file_not_leaked(self, fake_skills):
        """Even if traversal somehow passes, sensitive content must not leak."""
        result = json.loads(skill_view("test-skill", file_path="../../.env"))
        assert result["success"] is False
        assert "sk-do-not-leak" not in result.get("content", "")
        assert "sk-do-not-leak" not in json.dumps(result)
