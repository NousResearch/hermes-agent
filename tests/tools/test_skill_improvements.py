"""Tests for skill fuzzy patching via tools.fuzzy_match."""

import json
import os
import stat

import pytest

from tools.skill_manager_tool import (
    _SecurityScanOutcome,
    _create_skill,
    _edit_skill,
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


def _blocked_scan(_skill_dir, **_kwargs):
    return _SecurityScanOutcome(error="blocked")


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

    def test_create_rollback_removes_skill_when_scan_blocks(self, monkeypatch):
        """Blocked skill creation removes the newly created skill directory."""
        monkeypatch.setattr(
            "tools.skill_manager_tool._security_scan_skill",
            _blocked_scan,
        )

        result = _create_skill("blocked-skill", SKILL_CONTENT)

        assert result["success"] is False
        assert not (self.skills_dir / "blocked-skill").exists()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_new_documents_are_exactly_0644_under_restrictive_umask(self):
        """New skill documents override a restrictive process umask."""
        old_umask = os.umask(0o077)
        try:
            create_result = _create_skill("umask-skill", SKILL_CONTENT)
            write_result = _write_file(
                "umask-skill", "references/example.md", "# Reference\n"
            )
        finally:
            os.umask(old_umask)

        assert create_result["success"] is True
        assert write_result["success"] is True
        skill_md = self.skills_dir / "umask-skill" / "SKILL.md"
        reference = self.skills_dir / "umask-skill" / "references/example.md"
        assert stat.S_IMODE(skill_md.stat().st_mode) == 0o644
        assert stat.S_IMODE(reference.stat().st_mode) == 0o644

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_explicit_skill_md_patch_preserves_existing_mode(self):
        """Explicit SKILL.md patch paths preserve the existing document mode."""
        _create_skill("explicit-skill", SKILL_CONTENT)
        skill_md = self.skills_dir / "explicit-skill" / "SKILL.md"
        skill_md.chmod(0o660)

        result = _patch_skill(
            "explicit-skill",
            "Step 1: Do the thing.",
            "Step 1: Done!",
            file_path="SKILL.md",
        )

        assert result["success"] is True
        assert "Step 1: Done!" in skill_md.read_text(encoding="utf-8")
        assert stat.S_IMODE(skill_md.stat().st_mode) == 0o660

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.parametrize("mode", [0o600, 0o660])
    def test_edit_preserves_existing_mode(self, mode):
        """Full skill edits must preserve private and shared document modes."""
        _create_skill("mode-skill", SKILL_CONTENT)
        skill_md = self.skills_dir / "mode-skill" / "SKILL.md"
        skill_md.chmod(mode)
        replacement = SKILL_CONTENT.replace("Step 1: Do the thing.", "Step 1: Done!")

        result = _edit_skill("mode-skill", replacement)

        assert result["success"] is True
        assert stat.S_IMODE(skill_md.stat().st_mode) == mode

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
    @pytest.mark.parametrize("mode", [0o600, 0o660])
    def test_supporting_file_write_preserves_existing_mode(self, mode):
        """Overwriting a reference preserves its existing private or shared mode."""
        _create_skill("mode-skill", SKILL_CONTENT)
        reference = self.skills_dir / "mode-skill" / "references/example.md"
        reference.parent.mkdir()
        reference.write_text("old\n", encoding="utf-8")
        reference.chmod(mode)

        result = _write_file("mode-skill", "references/example.md", "new\n")

        assert result["success"] is True
        assert stat.S_IMODE(reference.stat().st_mode) == mode

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.parametrize("mode", [0o600, 0o660])
    def test_supporting_file_patch_preserves_existing_mode(self, mode):
        """Patching a reference preserves its existing private or shared mode."""
        _create_skill("mode-skill", SKILL_CONTENT)
        reference = self.skills_dir / "mode-skill" / "references/example.md"
        reference.parent.mkdir()
        reference.write_text("old\n", encoding="utf-8")
        reference.chmod(mode)

        result = _patch_skill(
            "mode-skill",
            "old",
            "new",
            file_path="references/example.md",
        )

        assert result["success"] is True
        assert reference.read_text(encoding="utf-8") == "new\n"
        assert stat.S_IMODE(reference.stat().st_mode) == mode

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_supporting_file_patch_rollback_preserves_mode_when_scan_blocks(
        self, monkeypatch
    ):
        """Blocked reference patches restore content and the original mode."""
        _create_skill("rollback-skill", SKILL_CONTENT)
        reference = self.skills_dir / "rollback-skill" / "references/example.md"
        reference.parent.mkdir()
        reference.write_text("original\n", encoding="utf-8")
        reference.chmod(0o660)
        monkeypatch.setattr(
            "tools.skill_manager_tool._security_scan_skill",
            _blocked_scan,
        )

        result = _patch_skill(
            "rollback-skill",
            "original",
            "blocked",
            file_path="references/example.md",
        )

        assert result["success"] is False
        assert reference.read_text(encoding="utf-8") == "original\n"
        assert stat.S_IMODE(reference.stat().st_mode) == 0o660

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_edit_rollback_preserves_existing_mode_when_scan_blocks(self, monkeypatch):
        """Blocked full edits restore both content and the original mode."""
        _create_skill("rollback-skill", SKILL_CONTENT)
        skill_md = self.skills_dir / "rollback-skill" / "SKILL.md"
        skill_md.chmod(0o660)
        replacement = SKILL_CONTENT.replace("Step 1: Do the thing.", "blocked edit")
        monkeypatch.setattr(
            "tools.skill_manager_tool._security_scan_skill",
            _blocked_scan,
        )

        result = _edit_skill("rollback-skill", replacement)

        assert result["success"] is False
        assert skill_md.read_text(encoding="utf-8") == SKILL_CONTENT
        assert stat.S_IMODE(skill_md.stat().st_mode) == 0o660

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_patch_rollback_preserves_existing_mode_when_scan_blocks(self, monkeypatch):
        """Blocked patches restore both content and the original mode."""
        _create_skill("rollback-skill", SKILL_CONTENT)
        skill_md = self.skills_dir / "rollback-skill" / "SKILL.md"
        skill_md.chmod(0o600)
        monkeypatch.setattr(
            "tools.skill_manager_tool._security_scan_skill",
            _blocked_scan,
        )

        result = _patch_skill(
            "rollback-skill", "Step 1: Do the thing.", "blocked patch"
        )

        assert result["success"] is False
        assert skill_md.read_text(encoding="utf-8") == SKILL_CONTENT
        assert stat.S_IMODE(skill_md.stat().st_mode) == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_supporting_file_rollback_preserves_existing_mode_when_scan_blocks(
        self, monkeypatch
    ):
        """Blocked supporting-file overwrites restore content and mode."""
        _create_skill("rollback-skill", SKILL_CONTENT)
        reference = self.skills_dir / "rollback-skill" / "references/example.md"
        reference.parent.mkdir()
        reference.write_text("original\n", encoding="utf-8")
        reference.chmod(0o660)
        monkeypatch.setattr(
            "tools.skill_manager_tool._security_scan_skill",
            _blocked_scan,
        )

        result = _write_file("rollback-skill", "references/example.md", "blocked\n")

        assert result["success"] is False
        assert reference.read_text(encoding="utf-8") == "original\n"
        assert stat.S_IMODE(reference.stat().st_mode) == 0o660

    def test_new_supporting_file_rollback_removes_file_when_scan_blocks(self, monkeypatch):
        """Blocked supporting-file creates remove the newly written file."""
        _create_skill("rollback-skill", SKILL_CONTENT)
        reference = self.skills_dir / "rollback-skill" / "references/example.md"
        monkeypatch.setattr(
            "tools.skill_manager_tool._security_scan_skill",
            _blocked_scan,
        )

        result = _write_file("rollback-skill", "references/example.md", "blocked\n")

        assert result["success"] is False
        assert not reference.exists()
