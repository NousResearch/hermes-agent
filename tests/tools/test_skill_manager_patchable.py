"""Tests for PR #5317 -- _find_skill() uses patchable module-level SKILLS_DIR.

The fix changes _find_skill() to use the module-level SKILLS_DIR variable as
the first search directory instead of get_all_skills_dirs()[0].  This makes
the primary search path patchable via unittest.mock.patch, enabling tests to
redirect skill discovery to a controlled tmp directory.

Tests verify:
- Patching SKILLS_DIR redirects _find_skill to the tmp directory
- _create_skill + _find_skill roundtrip works under a patched SKILLS_DIR
- External dirs (index 1+ from get_all_skills_dirs()) are still searched
- Original bug: without the fix, patching SKILLS_DIR wouldn't help
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import tools.skill_manager_tool as smt
from tools.skill_manager_tool import (
    _create_skill,
    _find_skill,
    _resolve_skill_dir,
    SKILLS_DIR,
)


# ---------------------------------------------------------------------------
# Minimal valid skill content
# ---------------------------------------------------------------------------


VALID_CONTENT = """\
---
name: test-skill
description: A test skill for patchability tests.
---

# Test Skill

Step 1: Do the patched thing.
"""


# ---------------------------------------------------------------------------
# Helper: plant a SKILL.md manually in a given directory
# ---------------------------------------------------------------------------


def _plant_skill(skills_dir: Path, name: str) -> Path:
    """Create a bare-minimum skill directory under skills_dir."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: planted skill.\n---\n\n# {name}\n"
    )
    return skill_dir


# ---------------------------------------------------------------------------
# Test: SKILLS_DIR is patchable
# ---------------------------------------------------------------------------


class TestSkillsDirPatchable:
    def test_find_skill_uses_patched_skills_dir(self, tmp_path):
        """_find_skill() finds a skill when SKILLS_DIR is patched to tmp_path."""
        fake_skills_dir = tmp_path / "skills"
        fake_skills_dir.mkdir()

        _plant_skill(fake_skills_dir, "my-test-skill")

        with (
            patch.object(smt, "SKILLS_DIR", fake_skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[fake_skills_dir]),
        ):
            result = _find_skill("my-test-skill")

        assert result is not None
        assert result["path"] == fake_skills_dir / "my-test-skill"

    def test_find_skill_returns_none_when_not_present(self, tmp_path):
        """_find_skill() returns None when skill not present in patched SKILLS_DIR."""
        fake_skills_dir = tmp_path / "skills"
        fake_skills_dir.mkdir()

        with (
            patch.object(smt, "SKILLS_DIR", fake_skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[fake_skills_dir]),
        ):
            result = _find_skill("nonexistent-skill")

        assert result is None

    def test_skills_dir_not_in_real_home(self, tmp_path):
        """With a patched SKILLS_DIR, real ~/.hermes/skills is not searched for the primary slot."""
        fake_skills_dir = tmp_path / "isolated"
        fake_skills_dir.mkdir()

        # Plant same skill name in fake_skills_dir and nowhere else
        _plant_skill(fake_skills_dir, "isolated-skill")

        with (
            patch.object(smt, "SKILLS_DIR", fake_skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[fake_skills_dir]),
        ):
            result = _find_skill("isolated-skill")

        # Must be found via fake_skills_dir, not the real home
        assert result is not None
        assert str(fake_skills_dir) in str(result["path"])


# ---------------------------------------------------------------------------
# Test: _create_skill + _find_skill roundtrip
# ---------------------------------------------------------------------------


class TestCreateFindRoundtrip:
    def test_roundtrip_with_patched_skills_dir(self, tmp_path):
        """Creating a skill then finding it works under a patched SKILLS_DIR."""
        fake_skills_dir = tmp_path / "skills"
        fake_skills_dir.mkdir()

        with (
            patch.object(smt, "SKILLS_DIR", fake_skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[fake_skills_dir]),
        ):
            result = _create_skill("roundtrip-skill", VALID_CONTENT)
            assert result["success"] is True, f"create failed: {result}"

            found = _find_skill("roundtrip-skill")

        assert found is not None
        assert found["path"].name == "roundtrip-skill"

    def test_roundtrip_skill_md_exists(self, tmp_path):
        """After create+find, the SKILL.md file is present."""
        fake_skills_dir = tmp_path / "skills"
        fake_skills_dir.mkdir()

        with (
            patch.object(smt, "SKILLS_DIR", fake_skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[fake_skills_dir]),
        ):
            _create_skill("has-skill-md", VALID_CONTENT)
            found = _find_skill("has-skill-md")

        assert found is not None
        skill_md = found["path"] / "SKILL.md"
        assert skill_md.exists()

    def test_roundtrip_with_category(self, tmp_path):
        """Skills created with a category are found by _find_skill."""
        fake_skills_dir = tmp_path / "skills"
        fake_skills_dir.mkdir()

        cat_content = """\
---
name: cat-skill
description: A categorised skill.
---

# Cat Skill

Step 1: Do something categorised.
"""

        with (
            patch.object(smt, "SKILLS_DIR", fake_skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[fake_skills_dir]),
        ):
            result = _create_skill("cat-skill", cat_content, category="my-category")
            assert result["success"] is True, f"create failed: {result}"

            found = _find_skill("cat-skill")

        assert found is not None
        assert found["path"].name == "cat-skill"


# ---------------------------------------------------------------------------
# Test: External dirs (index 1+) are still searched
# ---------------------------------------------------------------------------


class TestExternalDirsStillSearched:
    def test_skill_in_external_dir_is_found(self, tmp_path):
        """Skills in external dirs (not SKILLS_DIR) are still discovered."""
        primary_dir = tmp_path / "primary"
        external_dir = tmp_path / "external"
        primary_dir.mkdir()
        external_dir.mkdir()

        # Plant skill ONLY in the external dir
        _plant_skill(external_dir, "external-skill")

        with (
            patch.object(smt, "SKILLS_DIR", primary_dir),
            patch(
                "agent.skill_utils.get_all_skills_dirs",
                return_value=[primary_dir, external_dir],
            ),
        ):
            result = _find_skill("external-skill")

        assert result is not None
        assert result["path"] == external_dir / "external-skill"

    def test_primary_takes_precedence_over_external(self, tmp_path):
        """When same skill name exists in both primary and external, primary wins."""
        primary_dir = tmp_path / "primary"
        external_dir = tmp_path / "external"
        primary_dir.mkdir()
        external_dir.mkdir()

        # Plant skill in both dirs
        _plant_skill(primary_dir, "shared-skill")
        _plant_skill(external_dir, "shared-skill")

        with (
            patch.object(smt, "SKILLS_DIR", primary_dir),
            patch(
                "agent.skill_utils.get_all_skills_dirs",
                return_value=[primary_dir, external_dir],
            ),
        ):
            result = _find_skill("shared-skill")

        # Primary dir should win
        assert result is not None
        assert str(primary_dir) in str(result["path"])

    def test_multiple_external_dirs_searched(self, tmp_path):
        """Skills in the second external dir are found when not in the first."""
        primary_dir = tmp_path / "primary"
        ext1 = tmp_path / "ext1"
        ext2 = tmp_path / "ext2"
        for d in (primary_dir, ext1, ext2):
            d.mkdir()

        # Plant skill only in ext2
        _plant_skill(ext2, "deep-external-skill")

        with (
            patch.object(smt, "SKILLS_DIR", primary_dir),
            patch(
                "agent.skill_utils.get_all_skills_dirs",
                return_value=[primary_dir, ext1, ext2],
            ),
        ):
            result = _find_skill("deep-external-skill")

        assert result is not None
        assert result["path"] == ext2 / "deep-external-skill"


# ---------------------------------------------------------------------------
# Test: Regression – old behaviour wouldn't help patching SKILLS_DIR
# ---------------------------------------------------------------------------


class TestOriginalBugRegression:
    """
    Before the fix, _find_skill() called get_all_skills_dirs() and used its
    first element (get_hermes_home() / 'skills') regardless of the module-level
    SKILLS_DIR.  Patching SKILLS_DIR had no effect on _find_skill().

    After the fix, _find_skill() constructs:
        search_dirs = [SKILLS_DIR] + list(all_dirs[1:])
    so patching SKILLS_DIR *does* affect the primary search path.
    """

    def test_patching_skills_dir_is_effective(self, tmp_path):
        """
        Verify the fix: patching smt.SKILLS_DIR redirects _find_skill's
        primary search to the patched value.
        """
        patched_dir = tmp_path / "patched_skills"
        patched_dir.mkdir()
        _plant_skill(patched_dir, "patched-skill")

        # With the fix applied, patching SKILLS_DIR should make _find_skill
        # search patched_dir first.
        with (
            patch.object(smt, "SKILLS_DIR", patched_dir),
            patch(
                "agent.skill_utils.get_all_skills_dirs",
                return_value=[patched_dir],
            ),
        ):
            result = _find_skill("patched-skill")

        # The fix works: skill is found via the patched path
        assert result is not None, (
            "Expected _find_skill to respect the patched SKILLS_DIR "
            "(PR #5317 fix not effective)"
        )
        assert result["path"] == patched_dir / "patched-skill"

    def test_without_fix_model_patching_skills_dir_not_in_search_path(self, tmp_path):
        """
        Illustrate the original bug: if _find_skill used get_all_skills_dirs()[0]
        directly (ignoring SKILLS_DIR), patching SKILLS_DIR would not help.

        We simulate the old (unfixed) behaviour to document what it would look like,
        then confirm the fixed code avoids this by using [SKILLS_DIR] + list(all_dirs[1:]).
        """
        real_home_skills = tmp_path / "real_hermes_skills"
        patched_dir = tmp_path / "patched_skills"
        real_home_skills.mkdir()
        patched_dir.mkdir()

        # Skill only in patched_dir, NOT in real_home_skills
        _plant_skill(patched_dir, "only-in-patched")

        # Old (unfixed) _find_skill would iterate get_all_skills_dirs() directly,
        # where [0] == real_home_skills -- it would miss the patched_dir skill.
        def old_find_skill(name):
            """Simulates the unfixed _find_skill that ignores module-level SKILLS_DIR."""
            from agent.skill_utils import get_all_skills_dirs
            for skills_dir in get_all_skills_dirs():
                if not skills_dir.exists():
                    continue
                for skill_md in skills_dir.rglob("SKILL.md"):
                    if skill_md.parent.name == name:
                        return {"path": skill_md.parent}
            return None

        # Old code: even with SKILLS_DIR patched, get_all_skills_dirs()[0] is real_home_skills
        with (
            patch.object(smt, "SKILLS_DIR", patched_dir),
            patch(
                "agent.skill_utils.get_all_skills_dirs",
                return_value=[real_home_skills],  # Old code only sees real_home_skills
            ),
        ):
            old_result = old_find_skill("only-in-patched")

        # Old code misses the skill (demonstrates the original bug)
        assert old_result is None, "Old code should miss skill in patched SKILLS_DIR"

        # New (fixed) _find_skill: uses [SKILLS_DIR] + list(all_dirs[1:])
        # so patched_dir IS in the search path
        with (
            patch.object(smt, "SKILLS_DIR", patched_dir),
            patch(
                "agent.skill_utils.get_all_skills_dirs",
                return_value=[real_home_skills],  # get_all_skills_dirs still returns old first dir
            ),
        ):
            new_result = _find_skill("only-in-patched")

        # New code finds the skill because SKILLS_DIR is used as primary
        assert new_result is not None, "Fixed code should find skill via patched SKILLS_DIR"
        assert new_result["path"] == patched_dir / "only-in-patched"
