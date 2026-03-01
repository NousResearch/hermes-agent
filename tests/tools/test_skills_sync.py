"""Tests for tools/skills_sync.py — manifest-based skill seeding."""

from pathlib import Path
from unittest.mock import patch

from tools.skills_sync import (
    _read_manifest,
    _write_manifest,
    _discover_bundled_skills,
    _compute_relative_dest,
    sync_skills,
    MANIFEST_FILE,
    SKILLS_DIR,
)


class TestReadWriteManifest:
    def test_read_missing_manifest(self, tmp_path):
        with patch.object(
            __import__("tools.skills_sync", fromlist=["MANIFEST_FILE"]),
            "MANIFEST_FILE",
            tmp_path / "nonexistent",
        ):
            result = _read_manifest()
        assert result == set()

    def test_write_and_read_roundtrip(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        names = {"skill-a", "skill-b", "skill-c"}

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest(names)
            result = _read_manifest()

        assert result == names

    def test_write_manifest_sorted(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        names = {"zebra", "alpha", "middle"}

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest(names)

        lines = manifest_file.read_text().strip().splitlines()
        assert lines == ["alpha", "middle", "zebra"]

    def test_read_manifest_ignores_blank_lines(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        manifest_file.write_text("skill-a\n\n  \nskill-b\n")

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = _read_manifest()

        assert result == {"skill-a", "skill-b"}


class TestDiscoverBundledSkills:
    def test_finds_skills_with_skill_md(self, tmp_path):
        # Create two skills
        (tmp_path / "category" / "skill-a").mkdir(parents=True)
        (tmp_path / "category" / "skill-a" / "SKILL.md").write_text("# Skill A")
        (tmp_path / "skill-b").mkdir()
        (tmp_path / "skill-b" / "SKILL.md").write_text("# Skill B")

        # A directory without SKILL.md — should NOT be found
        (tmp_path / "not-a-skill").mkdir()
        (tmp_path / "not-a-skill" / "README.md").write_text("Not a skill")

        skills = _discover_bundled_skills(tmp_path)
        skill_names = {name for name, _ in skills}
        assert "skill-a" in skill_names
        assert "skill-b" in skill_names
        assert "not-a-skill" not in skill_names

    def test_ignores_git_directories(self, tmp_path):
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        (tmp_path / ".git" / "hooks" / "SKILL.md").write_text("# Fake")
        skills = _discover_bundled_skills(tmp_path)
        assert len(skills) == 0

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        skills = _discover_bundled_skills(tmp_path / "nonexistent")
        assert skills == []


class TestComputeRelativeDest:
    def test_preserves_category_structure(self):
        bundled = Path("/repo/skills")
        skill_dir = Path("/repo/skills/mlops/axolotl")
        dest = _compute_relative_dest(skill_dir, bundled)
        assert str(dest).endswith("mlops/axolotl")

    def test_flat_skill(self):
        bundled = Path("/repo/skills")
        skill_dir = Path("/repo/skills/simple")
        dest = _compute_relative_dest(skill_dir, bundled)
        assert dest.name == "simple"


class TestSyncSkills:
    def _setup_bundled(self, tmp_path):
        """Create a fake bundled skills directory."""
        bundled = tmp_path / "bundled_skills"
        (bundled / "category" / "new-skill").mkdir(parents=True)
        (bundled / "category" / "new-skill" / "SKILL.md").write_text("# New")
        (bundled / "category" / "new-skill" / "main.py").write_text("print(1)")
        (bundled / "category" / "DESCRIPTION.md").write_text("Category desc")
        (bundled / "old-skill").mkdir()
        (bundled / "old-skill" / "SKILL.md").write_text("# Old")
        return bundled

    def test_fresh_install_copies_all(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = sync_skills(quiet=True)

        assert len(result["copied"]) == 2
        assert result["total_bundled"] == 2
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()
        assert (skills_dir / "old-skill" / "SKILL.md").exists()
        # DESCRIPTION.md should also be copied
        assert (skills_dir / "category" / "DESCRIPTION.md").exists()

    def test_update_skips_known_skills(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        # Pre-populate manifest with old-skill
        manifest_file.write_text("old-skill\n")

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = sync_skills(quiet=True)

        # Only new-skill should be copied, old-skill skipped
        assert "new-skill" in result["copied"]
        assert "old-skill" not in result["copied"]
        assert result["skipped"] >= 1

    def test_does_not_overwrite_existing_skill_dir(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Pre-create the skill dir with user content
        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# User modified")

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = sync_skills(quiet=True)

        # Should not overwrite user's version
        assert (user_skill / "SKILL.md").read_text() == "# User modified"

    def test_nonexistent_bundled_dir(self, tmp_path):
        with patch("tools.skills_sync._get_bundled_dir", return_value=tmp_path / "nope"):
            result = sync_skills(quiet=True)
        assert result == {"copied": [], "skipped": 0, "total_bundled": 0}
