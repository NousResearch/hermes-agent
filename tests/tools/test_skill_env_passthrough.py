"""Test that skill_view registers required env vars in the passthrough registry."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.env_passthrough as _ep_mod
from tools.env_passthrough import clear_env_passthrough, is_env_passthrough

# Real bundled AgentMail skill — its CLI-first workflow runs `agentmail` through
# the sandboxed terminal, which strips secrets unless the skill declares them.
_AGENTMAIL_SKILL_SRC = Path(__file__).resolve().parents[2] / "skills" / "email" / "agentmail"


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
        """When a skill declares required_environment_variables and the var IS set,
        it should be registered in the passthrough."""
        _create_skill(
            tmp_path,
            "test-skill",
            frontmatter_extra=(
                "required_environment_variables:\n"
                "  - name: TENOR_API_KEY\n"
                "    prompt: Enter your Tenor API key\n"
            ),
        )
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR", tmp_path
        )
        # Set the env var so it's "available"
        monkeypatch.setenv("TENOR_API_KEY", "test-value-123")

        # Patch the secret capture callback to not prompt
        with patch("tools.skills_tool._secret_capture_callback", None):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view(name="test-skill"))

        assert result["success"] is True
        assert is_env_passthrough("TENOR_API_KEY")

    def test_remote_backend_persisted_env_vars_registered(self, tmp_path, monkeypatch):
        """Remote-backed skills still register locally available env vars."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        _create_skill(
            tmp_path,
            "test-skill",
            frontmatter_extra=(
                "required_environment_variables:\n"
                "  - name: TENOR_API_KEY\n"
                "    prompt: Enter your Tenor API key\n"
            ),
        )
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path)

        from hermes_cli.config import save_env_value

        save_env_value("TENOR_API_KEY", "persisted-value-123")
        monkeypatch.delenv("TENOR_API_KEY", raising=False)

        with patch("tools.skills_tool._secret_capture_callback", None):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view(name="test-skill"))

        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["missing_required_environment_variables"] == []
        assert is_env_passthrough("TENOR_API_KEY")

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


class TestAgentMailKeyPassthrough:
    """The real AgentMail skill must declare AGENTMAIL_API_KEY so a profile-stored
    key reaches the sandboxed terminal, while staying usable (self-signup) without
    one."""

    def _install_real_skill(self, tmp_path, monkeypatch):
        dest = tmp_path / "email" / "agentmail"
        shutil.copytree(_AGENTMAIL_SKILL_SRC, dest)
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path)
        return dest

    def test_agentmail_api_key_registered_for_sandbox_when_set(self, tmp_path, monkeypatch):
        """A profile-stored AGENTMAIL_API_KEY must pass through to the terminal
        sandbox — otherwise the documented CLI workflow fails authentication."""
        self._install_real_skill(tmp_path, monkeypatch)
        monkeypatch.setenv("AGENTMAIL_API_KEY", "am_test_key_123")

        with patch("tools.skills_tool._secret_capture_callback", None):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view(name="agentmail"))

        assert result["success"] is True
        assert is_env_passthrough("AGENTMAIL_API_KEY"), (
            "AGENTMAIL_API_KEY was set but not registered for sandbox passthrough — "
            "the skill's frontmatter must declare it under required_environment_variables"
        )

    def test_agentmail_key_is_optional_so_self_signup_still_works(self, tmp_path, monkeypatch):
        """With no key present the skill must NOT be blocked as setup_needed on the
        key — the CLI self-signup flow obtains one without a pre-provisioned key."""
        self._install_real_skill(tmp_path, monkeypatch)
        monkeypatch.delenv("AGENTMAIL_API_KEY", raising=False)

        with patch("tools.skills_tool._secret_capture_callback", None):
            from tools.skills_tool import skill_view

            result = json.loads(skill_view(name="agentmail"))

        assert result["success"] is True
        # `optional: true` means a missing key must NOT be reported as a blocking
        # requirement — that is what keeps the CLI self-signup path viable.
        assert "AGENTMAIL_API_KEY" not in result["missing_required_environment_variables"], (
            "AGENTMAIL_API_KEY must be declared optional so a missing key does not "
            "block the self-signup path"
        )
        assert result["setup_needed"] is False, (
            "an absent optional key must not force the skill into setup_needed"
        )
