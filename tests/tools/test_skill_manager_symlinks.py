"""Symlinked per-skill installs must be writable through skill_manage.

Regression: a dotfiles-managed library linked per skill into ~/.hermes/skills
was visible to skill_view/skills_list (os.walk followlinks=True) but every
skill_manage write failed with "not found in active profile" because
_find_skill walked with Path.rglob, which never descends into symlinked dirs.
"""
import json
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import (
    _find_skill,
    _iter_skill_dirs,
    _skill_not_found_error,
    skill_manage,
)

SKILL_MD = """\
---
name: linked-skill
description: A skill installed as a per-skill symlink.
---

# Linked Skill

Original body.
"""


@pytest.fixture()
def linked_library(tmp_path):
    """skills root holding one real dir and one per-skill symlink to an outside library."""
    library = tmp_path / "dotfiles-skills" / "devops" / "linked-skill"
    library.mkdir(parents=True)
    (library / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    root = tmp_path / "skills"
    (root / "devops").mkdir(parents=True)
    (root / "devops" / "linked-skill").symlink_to(library, target_is_directory=True)
    with patch("tools.skills_tool.SKILLS_DIR", root), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[root]):
        yield root, library


def test_iter_skill_dirs_follows_per_skill_symlinks(linked_library):
    root, _library = linked_library
    dirs = list(_iter_skill_dirs(root))
    assert any(d.name == "linked-skill" for d in dirs)


def test_find_skill_resolves_symlinked_skill(linked_library):
    hit = _find_skill("linked-skill")
    assert hit is not None
    assert hit["path"].name == "linked-skill"


def test_patch_writes_through_the_symlink(linked_library):
    _root, library = linked_library
    result = json.loads(skill_manage(action="", name="", operations=[{
        "name": "linked-skill", "action": "patch",
        "old_string": "Original body.", "new_string": "Patched body.",
    }]))
    assert result.get("success"), result
    # The write landed in the library (the symlink target), not a copy.
    assert "Patched body." in (library / "SKILL.md").read_text(encoding="utf-8")


def test_write_file_support_file_through_the_symlink(linked_library):
    _root, library = linked_library
    result = json.loads(skill_manage(action="", name="", operations=[{
        "name": "linked-skill", "action": "write_file",
        "file_path": "references/notes.md", "file_content": "# notes\n",
    }]))
    assert result.get("success"), result
    assert (library / "references" / "notes.md").exists()


def test_broken_symlink_gets_a_repair_hint(tmp_path):
    root = tmp_path / "skills"
    (root / "devops").mkdir(parents=True)
    (root / "devops" / "gone-skill").symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with patch("tools.skills_tool.SKILLS_DIR", root), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[root]):
        assert _find_skill("gone-skill") is None
        err = _skill_not_found_error("gone-skill")
        assert "broken symlink" in err and "gone-skill" in err