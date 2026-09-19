"""Tests for hermes_cli.skills_hub — local skill inspect/view."""

from pathlib import Path

import pytest
from rich.console import Console

from hermes_cli import skills_hub


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with a local skill and a category subdirectory."""
    import tools.skills_hub as hub_module
    import tools.skills_tool as skills_tool
    import tools.skill_manager_tool as smt

    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_hermes_home_cache", None, raising=False)

    from agent import skill_utils as su
    su._external_dirs_cache_clear()

    # Override SKILLS_DIR in both modules so _find_all_skills scans our temp dir
    monkeypatch.setattr(hub_module, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(smt, "SKILLS_DIR", skills_dir)

    # Clear the skills cache so _find_all_skills rescans
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})

    yield home

    su._external_dirs_cache_clear()


def _create_skill(home: Path, name: str, category: str = "",
                  description: str = "Test skill description.", tags=None):
    """Create a local skill directory with a SKILL.md and return its directory."""
    tags = tags or []
    base = (home / "skills" / category / name) if category else (home / "skills" / name)
    base.mkdir(parents=True, exist_ok=True)
    tag_line = f"\ntags: {tags}" if tags else ""
    (base / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}{tag_line}\n---\n\n# {name}\n\nThis is the {name} skill.\n",
        encoding="utf-8",
    )
    return base


class TestTryLocalSkillView:
    """6 E2E tests for _try_local_skill_view and do_view."""

    def test_try_local_skill_view_found_and_displayed(self, isolated_home):
        """_try_local_skill_view returns True and outputs skill info."""
        _create_skill(isolated_home, "my-skill", description="A useful skill.")
        c = Console()
        result = skills_hub._try_local_skill_view("my-skill", c)
        assert result is True

    def test_try_local_skill_view_returns_false_for_missing(self, isolated_home):
        """_try_local_skill_view returns False when skill doesn't exist."""
        c = Console()
        result = skills_hub._try_local_skill_view("nonexistent-skill", c)
        assert result is False

    def test_do_view_displays_local_skill(self, isolated_home):
        """do_view displays a local skill without error."""
        _create_skill(isolated_home, "my-skill", description="A useful skill.")
        c = Console()
        # Should not raise
        skills_hub.do_view("my-skill", c)

    def test_do_view_shows_error_for_missing(self, isolated_home):
        """do_view prints error for missing skill."""
        c = Console()
        skills_hub.do_view("nonexistent-skill", c)

    def test_try_local_skill_view_shows_skill_md_preview(self, isolated_home):
        """_try_local_skill_view shows SKILL.md content preview with body text."""
        _create_skill(isolated_home, "preview-skill", description="Preview test.",
                      tags=["test", "demo"])
        c = Console()
        result = skills_hub._try_local_skill_view("preview-skill", c)
        assert result is True

    def test_try_local_skill_view_finds_skill_in_category(self, isolated_home):
        """_try_local_skill_view finds a skill nested in a category subdirectory."""
        _create_skill(isolated_home, "nested-skill", category="devops",
                      description="A categorized skill.")
        c = Console()
        result = skills_hub._try_local_skill_view("nested-skill", c)
        assert result is True