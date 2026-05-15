"""Test that skill_view registers required env vars in the passthrough registry."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.env_passthrough as _ep_mod
from tools.env_passthrough import clear_env_passthrough, is_env_passthrough


@pytest.fixture(autouse=True)
def _clean_passthrough():
    clear_env_passthrough()
    _ep_mod._config_passthrough = None
    yield
    clear_env_passthrough()
    _ep_mod._config_passthrough = None


def _create_skill(tmp_path, name, frontmatter_extra=""):
    """Create a minimal skill directory with SKILL.md."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: Test skill\n"
        f"{frontmatter_extra}"
        f"---\n\n"
        f"# {name}\n\n"
        f"Test content.\n"
    )
    return skill_dir


class TestSkillViewRegistersPassthrough:
    def test_available_env_vars_registered(self, tmp_path, monkeypatch):
        """When a skill declares an unmanaged required env var that IS set,
        it should be registered in the passthrough."""
        _create_skill(
            tmp_path,
            "test-skill",
            frontmatter_extra=(
                "required_environment_variables:\n"
                "  - name: CUSTOM_SKILL_API_KEY\n"
                "    prompt: Enter your custom API key\n"
            ),
        )
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR", tmp_path
        )
        # Set the env var so it's "available"
        monkeypatch.setenv("CUSTOM_SKILL_API_KEY", "test-value-123")

        # Patch the secret capture callback to not prompt
        with patch("tools.skills_tool._secret_capture_callback", None):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view(name="test-skill"))

        assert result["success"] is True
        assert is_env_passthrough("CUSTOM_SKILL_API_KEY")

    def test_persisted_managed_skill_env_vars_not_registered(self, tmp_path, monkeypatch):
        """Managed skill secrets can satisfy setup checks without becoming subprocess passthrough vars."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        _create_skill(
            tmp_path,
            "test-skill",
            frontmatter_extra=(
                "required_environment_variables:\n"
                "  - name: NOTION_API_KEY\n"
                "    prompt: Enter your Notion API key\n"
            ),
        )
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path)

        from hermes_cli.config import save_env_value

        save_env_value("NOTION_API_KEY", "persisted-value-123")
        monkeypatch.delenv("NOTION_API_KEY", raising=False)

        with patch("tools.skills_tool._secret_capture_callback", None):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view(name="test-skill"))

        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["missing_required_environment_variables"] == []
        assert not is_env_passthrough("NOTION_API_KEY")

    def test_missing_env_vars_not_registered(self, tmp_path, monkeypatch):
        """When a skill declares required_environment_variables but the var is NOT set,
        it should NOT be registered in the passthrough."""
        _create_skill(
            tmp_path,
            "test-skill",
            frontmatter_extra=(
                "required_environment_variables:\n"
                "  - name: NONEXISTENT_SKILL_KEY_XYZ\n"
                "    prompt: Enter your key\n"
            ),
        )
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR", tmp_path
        )
        monkeypatch.delenv("NONEXISTENT_SKILL_KEY_XYZ", raising=False)

        with patch("tools.skills_tool._secret_capture_callback", None):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view(name="test-skill"))

        assert result["success"] is True
        assert not is_env_passthrough("NONEXISTENT_SKILL_KEY_XYZ")

    def test_no_env_vars_skill_no_registration(self, tmp_path, monkeypatch):
        """Skills without required_environment_variables shouldn't register anything."""
        _create_skill(tmp_path, "simple-skill")
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR", tmp_path
        )

        with patch("tools.skills_tool._secret_capture_callback", None):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view(name="simple-skill"))

        assert result["success"] is True
        from tools.env_passthrough import get_all_passthrough
        assert len(get_all_passthrough()) == 0
