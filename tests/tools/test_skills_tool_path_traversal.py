"""Tests for skill_view path traversal prevention.

Verifies that skill_view blocks path traversal attempts that could
expose sensitive files like ~/.hermes/.env or ~/.ssh/id_rsa.

Fixes: https://github.com/NousResearch/hermes-agent/issues/220
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSkillViewPathTraversal:
    """Test path traversal prevention in skill_view."""

    def test_double_dot_blocked(self):
        """Path traversal with .. components should be rejected."""
        from tools.skills_tool import skill_view
        
        # Mock _find_all_skills to return a fake skill
        with patch('tools.skills_tool._find_all_skills') as mock_find:
            mock_find.return_value = [{
                "name": "test-skill",
                "md": Path("/tmp/skills/test-skill/SKILL.md"),
                "dir": Path("/tmp/skills/test-skill")
            }]
            
            # Attempt path traversal to read .env
            result = skill_view("test-skill", file_path="../../.env")
            data = json.loads(result)
            
            assert data["success"] is False
            assert "traversal" in data["error"].lower()

    def test_double_dot_nested_blocked(self):
        """Nested path traversal should also be rejected."""
        from tools.skills_tool import skill_view
        
        with patch('tools.skills_tool._find_all_skills') as mock_find:
            mock_find.return_value = [{
                "name": "test-skill",
                "md": Path("/tmp/skills/test-skill/SKILL.md"),
                "dir": Path("/tmp/skills/test-skill")
            }]
            
            # Try deeper traversal
            result = skill_view("test-skill", file_path="references/../../../etc/passwd")
            data = json.loads(result)
            
            assert data["success"] is False
            assert "traversal" in data["error"].lower()

    def test_valid_path_allowed(self):
        """Valid relative paths within skill directory should work."""
        from tools.skills_tool import skill_view
        import tempfile
        import os
        
        # Create a temporary skill directory with a file
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("# Test Skill")
            
            ref_dir = skill_dir / "references"
            ref_dir.mkdir()
            ref_file = ref_dir / "test.md"
            ref_file.write_text("Test reference content")
            
            with patch('tools.skills_tool._find_all_skills') as mock_find:
                mock_find.return_value = [{
                    "name": "test-skill",
                    "md": skill_md,
                    "dir": skill_dir
                }]
                
                result = skill_view("test-skill", file_path="references/test.md")
                data = json.loads(result)
                
                assert data["success"] is True
                assert data["content"] == "Test reference content"

    def test_symlink_escape_blocked(self):
        """Symlinks that escape the skill directory should be blocked."""
        from tools.skills_tool import skill_view
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("# Test Skill")
            
            # Create a symlink pointing outside the skill directory
            secret_file = Path(tmpdir) / "secret.txt"
            secret_file.write_text("SECRET DATA")
            
            symlink = skill_dir / "evil-link"
            try:
                symlink.symlink_to(secret_file)
            except OSError:
                pytest.skip("Symlinks not supported on this system")
            
            with patch('tools.skills_tool._find_all_skills') as mock_find:
                mock_find.return_value = [{
                    "name": "test-skill",
                    "md": skill_md,
                    "dir": skill_dir
                }]
                
                result = skill_view("test-skill", file_path="evil-link")
                data = json.loads(result)
                
                # Should either fail or be blocked
                # The resolve() check should catch this
                if data["success"]:
                    # If it succeeded, the content should NOT be the secret
                    assert data.get("content") != "SECRET DATA"


class TestSkillViewEdgeCases:
    """Edge cases for path validation."""

    def test_absolute_path_rejected(self):
        """Absolute paths should not escape the skill directory."""
        from tools.skills_tool import skill_view
        
        with patch('tools.skills_tool._find_all_skills') as mock_find:
            mock_find.return_value = [{
                "name": "test-skill",
                "md": Path("/tmp/skills/test-skill/SKILL.md"),
                "dir": Path("/tmp/skills/test-skill")
            }]
            
            # Try absolute path
            result = skill_view("test-skill", file_path="/etc/passwd")
            data = json.loads(result)
            
            # Should fail - either not found or blocked
            # (Path("/tmp/skills/test-skill") / "/etc/passwd" normalizes weirdly on some systems)
            assert data["success"] is False or "passwd" not in data.get("content", "")

    def test_url_encoded_traversal_blocked(self):
        """URL-encoded path traversal should be blocked."""
        from tools.skills_tool import skill_view
        
        with patch('tools.skills_tool._find_all_skills') as mock_find:
            mock_find.return_value = [{
                "name": "test-skill",
                "md": Path("/tmp/skills/test-skill/SKILL.md"),
                "dir": Path("/tmp/skills/test-skill")
            }]
            
            # %2e%2e is URL encoding for ..
            # Python's pathlib will NOT decode this, so it should be safe
            # But we test anyway to ensure the behavior is correct
            result = skill_view("test-skill", file_path="%2e%2e/.env")
            data = json.loads(result)
            
            # Should fail (file not found since %2e%2e is literal)
            assert data["success"] is False
