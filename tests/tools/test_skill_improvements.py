"""Tests for skill fuzzy patching via tools.fuzzy_match."""

import json
import os
from pathlib import Path
from unittest.mock import patch

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

    def test_indentation_flexible_match(self):
        """Patch where only indentation differs should succeed."""
        skill = """\
---
name: indent-skill
description: Indentation test
---

# Steps

  1. First step
  2. Second step
  3. Third step
"""
        _create_skill("indent-skill", skill)
        # Agent sends with different indentation
        result = _patch_skill(
            "indent-skill",
            "1. First step\n2. Second step",
            "1. Updated first\n2. Updated second"
        )
        assert result["success"] is True
        content = (self.skills_dir / "indent-skill" / "SKILL.md").read_text()
        assert "Updated first" in content

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

    def test_replace_all_with_fuzzy(self):
        skill = """\
---
name: dup-skill
description: Duplicate test
---

# Steps

word word word
"""
        _create_skill("dup-skill", skill)
        result = _patch_skill("dup-skill", "word", "replaced", replace_all=True)
        assert result["success"] is True
        content = (self.skills_dir / "dup-skill" / "SKILL.md").read_text()
        assert "word" not in content
        assert "replaced" in content

    def test_no_match_returns_preview(self):
        _create_skill("test-skill", SKILL_CONTENT)
        result = _patch_skill("test-skill", "this does not exist anywhere", "replacement")
        assert result["success"] is False
        assert "file_preview" in result

    def test_fuzzy_patch_on_supporting_file(self):
        """Fuzzy matching should also work on supporting files."""
        _create_skill("test-skill", SKILL_CONTENT)
        ref_content = "    function hello() {\n        console.log('hi');\n    }"
        _write_file("test-skill", "references/code.js", ref_content)
        # Patch with stripped indentation
        result = _patch_skill(
            "test-skill",
            "function hello() {\nconsole.log('hi');\n}",
            "function hello() {\nconsole.log('hello world');\n}",
            file_path="references/code.js"
        )
        assert result["success"] is True
        content = (self.skills_dir / "test-skill" / "references" / "code.js").read_text()
        assert "hello world" in content

    def test_patch_preserves_frontmatter_validation(self):
        """Fuzzy matching should still run frontmatter validation on SKILL.md."""
        _create_skill("test-skill", SKILL_CONTENT)
        # Try to destroy the frontmatter via patch
        result = _patch_skill("test-skill", "---\nname: test-skill", "BROKEN")
        assert result["success"] is False
        assert "structure" in result["error"].lower() or "frontmatter" in result["error"].lower()

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
