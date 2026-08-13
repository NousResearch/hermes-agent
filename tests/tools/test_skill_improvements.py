"""Tests for skill fuzzy patching via tools.fuzzy_match."""

import json
import os
import stat

import pytest

from tools.skill_manager_tool import (
    _create_skill,
    _patch_skill,
    _write_file,
    skill_manage,
)


SKILL_CONTENT = """\
---
name: test-skill
description: A test skill for unit testing.
---

# Test Skill

Step 1: Do the thing.
Step 2: Do another thing.
Step 3: Final step.
"""


# ---------------------------------------------------------------------------
# Fuzzy patching
# ---------------------------------------------------------------------------


class TestFuzzyPatchSkill:
    @pytest.fixture(autouse=True)
    def setup_skills(self, tmp_path, monkeypatch):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        monkeypatch.setattr("tools.skill_manager_tool.SKILLS_DIR", skills_dir)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self.skills_dir = skills_dir

    def test_exact_match_still_works(self):
        _create_skill("test-skill", SKILL_CONTENT)
        result = _patch_skill("test-skill", "Step 1: Do the thing.", "Step 1: Done!")
        assert result["success"] is True
        content = (self.skills_dir / "test-skill" / "SKILL.md").read_text()
        assert "Step 1: Done!" in content

    def test_whitespace_trimmed_match(self):
        """Patch with extra leading whitespace should still find the target."""
        skill = """\
---
name: ws-skill
description: Whitespace test
---

# Commands

    def hello():
        print("hi")
"""
        _create_skill("ws-skill", skill)
        # Agent sends patch with no leading whitespace (common LLM behaviour)
        result = _patch_skill("ws-skill", "def hello():\n    print(\"hi\")", "def hello():\n    print(\"hello world\")")
        assert result["success"] is True
        content = (self.skills_dir / "ws-skill" / "SKILL.md").read_text()
        assert 'print("hello world")' in content


    def test_multiple_matches_blocked_without_replace_all(self):
        """Multiple fuzzy matches should return an error without replace_all."""
        skill = """\
---
name: dup-skill
description: Duplicate test
---

# Steps

word word word
"""
        _create_skill("dup-skill", skill)
        result = _patch_skill("dup-skill", "word", "replaced")
        assert result["success"] is False
        assert "match" in result["error"].lower()


    def test_skill_manage_patch_uses_fuzzy(self):
        """The dispatcher should route to the fuzzy-matching patch."""
        _create_skill("test-skill", SKILL_CONTENT)
        raw = skill_manage(
            action="patch",
            name="test-skill",
            old_string="  Step 1: Do the thing.",  # extra leading space
            new_string="Step 1: Updated.",
        )
        result = json.loads(raw)
        # Should succeed via line-trimmed or indentation-flexible matching
        assert result["success"] is True

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_created_skill_is_group_readable(self):
        """New instructional skills use the public-document mode 0644."""
        _create_skill("mode-skill", SKILL_CONTENT)
        mode = stat.S_IMODE((self.skills_dir / "mode-skill" / "SKILL.md").stat().st_mode)
        assert mode == 0o644

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_edit_preserves_group_readable_mode(self):
        """Full skill edits must preserve the existing document mode."""
        _create_skill("mode-skill", SKILL_CONTENT)
        skill_md = self.skills_dir / "mode-skill" / "SKILL.md"
        skill_md.chmod(0o644)
        replacement = SKILL_CONTENT.replace("Step 1: Do the thing.", "Step 1: Done!")

        from tools.skill_manager_tool import _edit_skill

        result = _edit_skill("mode-skill", replacement)

        assert result["success"] is True
        assert stat.S_IMODE(skill_md.stat().st_mode) == 0o644

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.parametrize("mode", [0o600, 0o660])
    def test_patched_skill_preserves_existing_mode(self, mode):
        """Atomic patching must preserve both private and shared modes."""
        _create_skill("mode-skill", SKILL_CONTENT)
        skill_md = self.skills_dir / "mode-skill" / "SKILL.md"
        skill_md.chmod(mode)

        result = _patch_skill("mode-skill", "Step 1: Do the thing.", "Step 1: Done!")

        assert result["success"] is True
        assert stat.S_IMODE(skill_md.stat().st_mode) == mode

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_supporting_file_write_uses_group_readable_mode(self):
        """New reference files should follow the same document mode."""
        _create_skill("mode-skill", SKILL_CONTENT)

        result = _write_file(
            "mode-skill",
            "references/example.md",
            "# Reference\n",
        )

        assert result["success"] is True
        reference = self.skills_dir / "mode-skill" / "references/example.md"
        assert stat.S_IMODE(reference.stat().st_mode) == 0o644

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_supporting_file_write_preserves_existing_mode(self):
        """Overwriting a reference must preserve its existing shared mode."""
        _create_skill("mode-skill", SKILL_CONTENT)
        reference = self.skills_dir / "mode-skill" / "references/example.md"
        reference.parent.mkdir()
        reference.write_text("old\n", encoding="utf-8")
        reference.chmod(0o660)

        result = _write_file("mode-skill", "references/example.md", "new\n")

        assert result["success"] is True
        assert stat.S_IMODE(reference.stat().st_mode) == 0o660
