"""Tests for tools/skills_sync.py — manifest-based skill seeding and updating."""

from pathlib import Path
from unittest.mock import patch

from tools.skills_sync import (
    _read_manifest,
    _write_manifest,
    _discover_bundled_skills,
    _compute_relative_dest,
    _dir_hash,
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


class TestDirHash:
    def test_same_content_same_hash(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        for d in (dir_a, dir_b):
            d.mkdir()
            (d / "SKILL.md").write_text("# Test")
            (d / "main.py").write_text("print(1)")
        assert _dir_hash(dir_a) == _dir_hash(dir_b)

    def test_different_content_different_hash(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "SKILL.md").write_text("# Version 1")
        (dir_b / "SKILL.md").write_text("# Version 2")
        assert _dir_hash(dir_a) != _dir_hash(dir_b)

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        h = _dir_hash(d)
        assert isinstance(h, str) and len(h) == 32

    def test_nonexistent_dir(self, tmp_path):
        h = _dir_hash(tmp_path / "nope")
        assert isinstance(h, str)  # returns hash of empty content


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
        assert result["updated"] == []
        assert result["cleaned"] == []
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()
        assert (skills_dir / "old-skill" / "SKILL.md").exists()
        # DESCRIPTION.md should also be copied
        assert (skills_dir / "category" / "DESCRIPTION.md").exists()

    def test_user_deleted_skill_not_re_added(self, tmp_path):
        """Skill in manifest but not on disk = user deleted it. Don't re-add."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        # old-skill is in manifest but NOT on disk (user deleted it)
        manifest_file.write_text("old-skill\n")

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = sync_skills(quiet=True)

        # new-skill should be copied, old-skill should be skipped
        assert "new-skill" in result["copied"]
        assert "old-skill" not in result["copied"]
        assert "old-skill" not in result.get("updated", [])
        assert not (skills_dir / "old-skill").exists()

    def test_existing_skill_gets_updated(self, tmp_path):
        """Skill in manifest AND on disk with changed content = updated."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Pre-create old-skill on disk with DIFFERENT content
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old version from last sync")
        # Mark it in the manifest
        manifest_file.write_text("old-skill\n")

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = sync_skills(quiet=True)

        # old-skill should be updated
        assert "old-skill" in result["updated"]
        assert (user_skill / "SKILL.md").read_text() == "# Old"

    def test_unchanged_skill_not_updated(self, tmp_path):
        """Skill in manifest AND on disk with same content = skipped."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Pre-create old-skill on disk with SAME content
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old")
        # Mark it in the manifest
        manifest_file.write_text("old-skill\n")

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = sync_skills(quiet=True)

        # Should be skipped, not updated
        assert "old-skill" not in result.get("updated", [])
        assert result["skipped"] >= 1

    def test_stale_manifest_entries_cleaned(self, tmp_path):
        """Skills in manifest that no longer exist in bundled dir get cleaned."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        # Add a stale entry that doesn't exist in bundled
        manifest_file.write_text("old-skill\nremoved-skill\n")

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = sync_skills(quiet=True)

        assert "removed-skill" in result["cleaned"]
        # Verify manifest no longer has removed-skill
        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            manifest = _read_manifest()
        assert "removed-skill" not in manifest

    def test_does_not_overwrite_existing_unmanifested_skill(self, tmp_path):
        """New skill whose name collides with user-created skill = skipped."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Pre-create the skill dir with user content (not in manifest)
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
        assert result == {"copied": [], "updated": [], "skipped": 0, "cleaned": [], "total_bundled": 0}
