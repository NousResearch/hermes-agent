"""Regression test: skill discovery follows symlinked category directories (#35184)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from tools.skill_manager_tool import _find_skill


def test_find_skill_follows_symlinks():
    """#35184: _find_skill should discover skills under symlinked
    category directories, not just direct subdirectories.

    pathlib.Path.rglob() does not follow symlinks into directories,
    so skills in symlinked categories were invisible.  The fix uses
    iter_skill_index_files() which calls os.walk(followlinks=True).
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Create a real skills dir
        skills_dir = Path(tmp) / "skills"
        skills_dir.mkdir()

        # Create a skill in a real subdirectory
        real_cat = skills_dir / "real-cat"
        real_cat.mkdir()
        skill_dir = real_cat / "my-symlinked-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill")

        # Create a skill in an external location
        alt_location = Path(tmp) / "alt-skills"
        alt_location.mkdir()
        symlinked_cat = alt_location / "symlinked-cat"
        symlinked_cat.mkdir()
        symlinked_skill = symlinked_cat / "my-symlinked-skill-2"
        symlinked_skill.mkdir()
        (symlinked_skill / "SKILL.md").write_text("# Symlinked Skill")

        # Symlink the external category into skills dir
        symlink = skills_dir / "linked-cat"
        os.symlink(symlinked_cat, symlink, target_is_directory=True)

        with patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]):
            # Direct (non-symlinked) skill — always worked
            result1 = _find_skill("my-symlinked-skill")
            assert result1 is not None, "Direct skill should be found"
            assert result1["path"] == skill_dir

            # Symlinked skill — was broken with rglob, fixed with iter_skill_index_files
            result2 = _find_skill("my-symlinked-skill-2")
            assert result2 is not None, (
                "Skill under symlinked category should be found "
                "(regression: rglob does not follow symlinks)"
            )
            assert result2["path"].resolve() == symlinked_skill.resolve(), (
                f"Expected {symlinked_skill.resolve()}, got {result2['path'].resolve()}"
            )
