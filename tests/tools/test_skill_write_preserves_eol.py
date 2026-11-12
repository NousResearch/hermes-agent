"""A skill rewrite must not translate newlines.

On Windows the text handle inside ``utils._atomic_write`` runs the payload through the platform
newline translation, so a single ``skill_manage`` write turns a whole LF ``SKILL.md`` into CRLF.
``tools/skills_sync._dir_hash`` hashes the package byte-for-byte, so that alone flips a bundled
skill to ``user_modified`` and `hermes update` can never reconcile it again — even when the write
changed nothing else.

Windows-only by construction: the translation does not exist on other hosts, so there is nothing
to assert there (tests/AGENTS.md — host-specific behaviour is tested on that host).
"""

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import skill_manage
from tools.skills_sync import _dir_hash

SKILL_MD = """\
---
name: demo-skill
description: A demo skill used by the EOL-preservation test.
---

# Demo Skill

Step 1: do the thing.
"""


@contextmanager
def _skill_dir(tmp_path):
    """Search only *tmp_path* for skills, never the real ``~/.hermes/skills``."""
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


def _seed_lf_skill(tmp_path):
    """A skill package whose SKILL.md is LF, exactly like the copy the repo ships."""
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    # write_bytes, not write_text: the fixture must stay LF on every host — write_text would
    # translate on Windows and hide the very drift this file is about.
    (skill_dir / "SKILL.md").write_bytes(SKILL_MD.encode("utf-8"))
    return skill_dir


@pytest.mark.platforms("windows")
def test_rewriting_identical_content_leaves_the_skill_hash_unchanged(tmp_path):
    """The invariant `hermes update` depends on: a no-op rewrite is byte-neutral.

    With the platform translation in place the write alone moves the hash, so a bundled skill
    that the review fork merely re-saves is stranded as user-modified forever.
    """
    with _skill_dir(tmp_path):
        skill_dir = _seed_lf_skill(tmp_path)
        before = _dir_hash(skill_dir)

        result = json.loads(skill_manage(action="edit", name="demo-skill", content=SKILL_MD))

        assert result["success"] is True, result
        assert (skill_dir / "SKILL.md").read_bytes() == SKILL_MD.encode("utf-8")
        assert _dir_hash(skill_dir) == before


@pytest.mark.platforms("windows")
def test_patch_does_not_rewrite_the_untouched_lines_as_crlf(tmp_path):
    """A targeted patch edits its match; it must not re-line the whole file."""
    with _skill_dir(tmp_path):
        skill_dir = _seed_lf_skill(tmp_path)

        result = json.loads(skill_manage(
            action="patch", name="demo-skill",
            old_string="Step 1: do the thing.", new_string="Step 1: do the thing, twice."))

        assert result["success"] is True, result
        raw = (skill_dir / "SKILL.md").read_bytes()

    assert b"\r\n" not in raw
    assert b"Step 1: do the thing, twice.\n" in raw
    assert raw.startswith(b"---\nname: demo-skill\n")


@pytest.mark.platforms("windows")
def test_write_file_does_not_translate_newlines(tmp_path):
    """Supporting files go through the same writer, so they need the same guarantee."""
    with _skill_dir(tmp_path):
        skill_dir = _seed_lf_skill(tmp_path)

        result = json.loads(skill_manage(
            action="write_file", name="demo-skill",
            file_path="references/lesson.md", file_content="# Lesson\n\nOne rule.\n"))

        assert result["success"] is True, result
        assert (skill_dir / "references" / "lesson.md").read_bytes() == b"# Lesson\n\nOne rule.\n"
