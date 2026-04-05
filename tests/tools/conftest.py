"""conftest for tests/tools/ — shared fixtures."""

from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def patch_skills_dirs(tmp_path: Path, monkeypatch):
    """Patch get_all_skills_dirs to return tmp_path for all skill-manager tests.

    skill_manager_tool refactored _find_skill to call agent.skill_utils.get_all_skills_dirs()
    instead of using SKILLS_DIR directly.  Tests that patch SKILLS_DIR still create
    skills in tmp_path but _find_skill would search the real ~/.hermes/skills/ and miss them.

    This fixture patches get_all_skills_dirs to always return [tmp_path].
    _resolve_skill_dir and _create_skill use SKILLS_DIR which tests patch directly.
    """
    monkeypatch.setattr(
        "agent.skill_utils.get_all_skills_dirs",
        lambda: [tmp_path],
    )


@contextmanager
def patch_skills_dir_for_test(tmp_path: Path):
    """Patch both SKILLS_DIR and get_all_skills_dirs simultaneously.

    Use this inside tests that already use `with patch("...SKILLS_DIR", tmp_path):`
    to also patch get_all_skills_dirs, which _find_skill calls internally.

    Example:
        with patch_skills_dir_for_test(tmp_path):
            _create_skill("my-skill", content)
            result = _delete_skill("my-skill")
        assert result["success"] is True
    """
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield
