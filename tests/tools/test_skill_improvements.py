"""Tests for skill fuzzy patching via tools.fuzzy_match."""

import json

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
