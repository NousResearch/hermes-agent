"""Tests for tools/skills_sync.py — manifest-based skill seeding and updating."""

from pathlib import Path
from unittest.mock import patch

from tools.skills_sync import (
    _get_bundled_dir,
    _read_manifest,
    _read_skill_name,
    _write_manifest,
    _discover_bundled_skills,
    _compute_relative_dest,
    _dir_hash,
    sync_skills,
    reset_bundled_skill,
    MANIFEST_FILE,
    SKILLS_DIR,
)


class TestReadWriteManifest:
    def test_read_missing_manifest(self, tmp_path):
        with patch(
            "tools.skills_sync.MANIFEST_FILE",
            tmp_path / "nonexistent",
        ):
            result = _read_manifest()
        assert result == {}

    def test_write_and_read_roundtrip_v2(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        entries = {"skill-a": "abc123", "skill-b": "def456", "skill-c": "789012"}

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest(entries)
            result = _read_manifest()

        assert result == entries

    def test_write_manifest_sorted(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        entries = {"zebra": "hash1", "alpha": "hash2", "middle": "hash3"}

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest(entries)

        lines = manifest_file.read_text().strip().splitlines()
        names = [line.split(":")[0] for line in lines]
        assert names == ["alpha", "middle", "zebra"]

    def test_read_v1_manifest_migration(self, tmp_path):
        """v1 format (plain names, no hashes) should be read with empty hashes."""
        manifest_file = tmp_path / ".bundled_manifest"
        manifest_file.write_text("skill-a\nskill-b\n")

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = _read_manifest()

        assert result == {"skill-a": "", "skill-b": ""}

    def test_read_manifest_ignores_blank_lines(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        manifest_file.write_text("skill-a:hash1\n\n  \nskill-b:hash2\n")

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = _read_manifest()

        assert result == {"skill-a": "hash1", "skill-b": "hash2"}

    def test_read_manifest_mixed_v1_v2(self, tmp_path):
        """Manifest with both v1 and v2 lines (shouldn't happen but handle gracefully)."""
        manifest_file = tmp_path / ".bundled_manifest"
        manifest_file.write_text("old-skill\nnew-skill:abc123\n")

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = _read_manifest()

        assert result == {"old-skill": "", "new-skill": "abc123"}


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
        (tmp_path / "category" / "skill-a").mkdir(parents=True)
        (tmp_path / "category" / "skill-a" / "SKILL.md").write_text("# Skill A")
        (tmp_path / "skill-b").mkdir()
        (tmp_path / "skill-b" / "SKILL.md").write_text("# Skill B")
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


class TestReadSkillName:
    def test_reads_name_from_frontmatter(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: audiocraft-audio-generation\n---\n# Skill")
        assert _read_skill_name(skill_md, "audiocraft") == "audiocraft-audio-generation"

    def test_falls_back_to_dir_name_without_frontmatter(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("# Just a heading\nNo frontmatter here")
        assert _read_skill_name(skill_md, "my-skill") == "my-skill"

    def test_falls_back_when_name_field_empty(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname:\n---\n")
        assert _read_skill_name(skill_md, "fallback") == "fallback"

    def test_handles_quoted_name(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text('---\nname: "serving-llms-vllm"\n---\n')
        assert _read_skill_name(skill_md, "vllm") == "serving-llms-vllm"

    def test_discover_uses_frontmatter_name(self, tmp_path):
        skill_dir = tmp_path / "category" / "audiocraft"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: audiocraft-audio-generation\n---\n# Skill"
        )
        skills = _discover_bundled_skills(tmp_path)
        assert skills[0][0] == "audiocraft-audio-generation"


class TestComputeRelativeDest:
    def test_preserves_category_structure(self):
        bundled = Path("/repo/skills")
        skill_dir = Path("/repo/skills/mlops/axolotl")
        dest = _compute_relative_dest(skill_dir, bundled)
        assert dest.as_posix().endswith("mlops/axolotl")

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

    def _patches(self, bundled, skills_dir, manifest_file):
        """Return context manager stack for patching sync globals."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        return stack

    def test_fresh_install_copies_all(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert len(result["copied"]) == 2
        assert result["total_bundled"] == 2
        assert result["updated"] == []
        assert result["user_modified"] == []
        assert result["cleaned"] == []
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()
        assert (skills_dir / "old-skill" / "SKILL.md").exists()
        assert (skills_dir / "category" / "DESCRIPTION.md").exists()

    def test_fresh_install_records_origin_hashes(self, tmp_path):
        """After fresh install, manifest should have v2 format with hashes."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)
            manifest = _read_manifest()

        assert "new-skill" in manifest
        assert "old-skill" in manifest
        # Hashes should be non-empty MD5 strings
        assert len(manifest["new-skill"]) == 32
        assert len(manifest["old-skill"]) == 32

    def test_user_deleted_skill_not_re_added(self, tmp_path):
        """Skill in manifest but not on disk = user deleted it. Don't re-add."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        # old-skill is in manifest (v2 format) but NOT on disk
        old_hash = _dir_hash(bundled / "old-skill")
        manifest_file.write_text(f"old-skill:{old_hash}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert "new-skill" in result["copied"]
        assert "old-skill" not in result["copied"]
        assert "old-skill" not in result.get("updated", [])
        assert not (skills_dir / "old-skill").exists()

    def test_unmodified_skill_gets_updated(self, tmp_path):
        """Skill in manifest + on disk + user hasn't modified = update from bundled."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Simulate: user has old version that was synced from an older bundled
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        old_origin_hash = _dir_hash(user_skill)

        # Record origin hash = hash of what was synced (the old version)
        manifest_file.write_text(f"old-skill:{old_origin_hash}\n")

        # Now bundled has a newer version ("# Old" != "# Old v1")
        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        # Should be updated because user copy matches origin (unmodified)
        assert "old-skill" in result["updated"]
        assert (user_skill / "SKILL.md").read_text() == "# Old"

    def test_user_modified_skill_not_overwritten(self, tmp_path):
        """Skill modified by user should NOT be overwritten even if bundled changed."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Simulate: user had the old version synced, then modified it
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        old_origin_hash = _dir_hash(user_skill)

        # Record origin hash from what was originally synced
        manifest_file.write_text(f"old-skill:{old_origin_hash}\n")

        # User modifies their copy
        (user_skill / "SKILL.md").write_text("# My custom version")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        # Should NOT update — user modified it
        assert "old-skill" in result["user_modified"]
        assert "old-skill" not in result.get("updated", [])
        assert (user_skill / "SKILL.md").read_text() == "# My custom version"

    def test_unchanged_skill_not_updated(self, tmp_path):
        """Skill in sync (user == bundled == origin) = no action needed."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Copy bundled to user dir (simulating perfect sync state)
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old")
        origin_hash = _dir_hash(user_skill)
        manifest_file.write_text(f"old-skill:{origin_hash}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert "old-skill" not in result.get("updated", [])
        assert "old-skill" not in result.get("user_modified", [])
        assert result["skipped"] >= 1

    def test_v1_manifest_migration_sets_baseline(self, tmp_path):
        """v1 manifest entries (no hash) should set baseline from user's current copy."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Pre-create skill on disk
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old modified by user")

        # v1 manifest (no hashes)
        manifest_file.write_text("old-skill\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)
            # Should skip (migration baseline set), NOT update
            assert "old-skill" not in result.get("updated", [])
            assert "old-skill" not in result.get("user_modified", [])

            # Now check manifest was upgraded to v2 with user's hash as baseline
            manifest = _read_manifest()
            assert len(manifest["old-skill"]) == 32  # MD5 hash

    def test_v1_migration_then_bundled_update_detected(self, tmp_path):
        """After v1 migration, a subsequent sync should detect bundled updates."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # User has the SAME content as bundled (in sync)
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old")

        # v1 manifest
        manifest_file.write_text("old-skill\n")

        with self._patches(bundled, skills_dir, manifest_file):
            # First sync: migration — sets baseline
            sync_skills(quiet=True)

            # Now change bundled content
            (bundled / "old-skill" / "SKILL.md").write_text("# Old v2 — improved")

            # Second sync: should detect bundled changed + user unmodified → update
            result = sync_skills(quiet=True)

        assert "old-skill" in result["updated"]
        assert (user_skill / "SKILL.md").read_text() == "# Old v2 — improved"

    def test_stale_manifest_entries_cleaned(self, tmp_path):
        """Skills in manifest that no longer exist in bundled dir get cleaned."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("old-skill:abc123\nremoved-skill:def456\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert "removed-skill" in result["cleaned"]
        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            manifest = _read_manifest()
        assert "removed-skill" not in manifest

    def test_does_not_overwrite_existing_unmanifested_skill(self, tmp_path):
        """New skill whose name collides with user-created skill = skipped."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# User modified")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert (user_skill / "SKILL.md").read_text() == "# User modified"

    def test_collision_does_not_poison_manifest(self, tmp_path):
        """Collision with an unmanifested user skill must NOT record bundled_hash.

        Otherwise the next sync compares user_hash against the recorded
        bundled_hash, finds a mismatch, and permanently flags the skill as
        'user-modified' — even though the user never touched a bundled copy.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Pre-existing user skill (e.g. from hub, custom, or leftover) that
        # happens to share a name with a newly bundled skill.
        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# From hub — unrelated to bundled")

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)

        # User file must survive (existing invariant).
        assert (user_skill / "SKILL.md").read_text() == (
            "# From hub — unrelated to bundled"
        )

        # Manifest must NOT contain the skill — it was never synced from bundled.
        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            manifest = _read_manifest()
        assert "new-skill" not in manifest, (
            "Collision path wrote bundled_hash to the manifest even though "
            "the on-disk copy is unrelated to bundled. This poisons update "
            "detection: the next sync will mark the skill as 'user-modified'."
        )

    def test_collision_does_not_trigger_false_user_modified_on_resync(self, tmp_path):
        """End-to-end: after a collision, a second sync must not flag user_modified.

        Pre-fix bug: first sync wrote bundled_hash to the manifest; second
        sync then diffed user_hash vs bundled_hash, mismatched, and shoved
        the skill into the user_modified bucket forever.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# From hub — unrelated to bundled")

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)  # first sync: collision path
            result2 = sync_skills(quiet=True)  # second sync: must not flag

        assert "new-skill" not in result2["user_modified"], (
            "Second sync after a collision falsely flagged the user's skill "
            "as 'user-modified' — the manifest was poisoned on the first sync."
        )

    def test_collision_prints_reset_hint(self, tmp_path, capsys):
        """Non-quiet sync must print a reset hint when a collision is skipped.

        Silent skip hides the fact that a bundled skill shipped but was
        shadowed by the user's local copy. The hint tells the user the
        exact command to take the bundled version instead.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# From hub — unrelated to bundled")

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=False)

        captured = capsys.readouterr().out
        assert "new-skill" in captured
        assert "hermes skills reset new-skill" in captured

    def test_nonexistent_bundled_dir(self, tmp_path):
        with patch("tools.skills_sync._get_bundled_dir", return_value=tmp_path / "nope"):
            result = sync_skills(quiet=True)
        assert result == {
            "copied": [], "updated": [], "skipped": 0,
            "user_modified": [], "cleaned": [], "total_bundled": 0,
        }

    def test_failed_copy_does_not_poison_manifest(self, tmp_path):
        """If copytree fails, the skill must NOT be added to the manifest.

        Otherwise the next sync treats it as 'user deleted' and never retries.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file):
            # Patch copytree to fail for new-skill
            original_copytree = __import__("shutil").copytree

            def failing_copytree(src, dst, *a, **kw):
                if "new-skill" in str(src):
                    raise OSError("Simulated disk full")
                return original_copytree(src, dst, *a, **kw)

            with patch("shutil.copytree", side_effect=failing_copytree):
                result = sync_skills(quiet=True)

            # new-skill should NOT be in copied (it failed)
            assert "new-skill" not in result["copied"]

            # Critical: new-skill must NOT be in the manifest
            manifest = _read_manifest()
            assert "new-skill" not in manifest, (
                "Failed copy was recorded in manifest — next sync will "
                "treat it as 'user deleted' and never retry"
            )

            # Now run sync again (copytree works this time) — it should retry
            result2 = sync_skills(quiet=True)
            assert "new-skill" in result2["copied"]
            assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()

    def test_failed_update_does_not_destroy_user_copy(self, tmp_path):
        """If copytree fails during update, the user's existing copy must survive."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Start with old synced version
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        old_hash = _dir_hash(user_skill)
        manifest_file.write_text(f"old-skill:{old_hash}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            # Patch copytree to fail (rmtree succeeds, copytree fails)
            original_copytree = __import__("shutil").copytree

            def failing_copytree(src, dst, *a, **kw):
                if "old-skill" in str(src):
                    raise OSError("Simulated write failure")
                return original_copytree(src, dst, *a, **kw)

            with patch("shutil.copytree", side_effect=failing_copytree):
                result = sync_skills(quiet=True)

            # old-skill should NOT be in updated (it failed)
            assert "old-skill" not in result.get("updated", [])

            # The skill directory should still exist (rmtree destroyed it
            # but copytree failed to replace it — this is data loss)
            assert user_skill.exists(), (
                "Update failure destroyed user's skill copy without replacing it"
            )

    def test_update_records_new_origin_hash(self, tmp_path):
        """After updating a skill, the manifest should record the new bundled hash."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Start with old synced version
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        old_hash = _dir_hash(user_skill)
        manifest_file.write_text(f"old-skill:{old_hash}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)  # updates to "# Old"
            manifest = _read_manifest()

        # New origin hash should match the bundled version
        new_bundled_hash = _dir_hash(bundled / "old-skill")
        assert manifest["old-skill"] == new_bundled_hash
        assert manifest["old-skill"] != old_hash


class TestGetBundledDir:
    def test_env_var_override(self, tmp_path, monkeypatch):
        """HERMES_BUNDLED_SKILLS env var overrides the default path resolution."""
        custom_dir = tmp_path / "custom_skills"
        custom_dir.mkdir()
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(custom_dir))
        assert _get_bundled_dir() == custom_dir

    def test_default_without_env_var(self, monkeypatch):
        """Without the env var, falls back to relative path from __file__."""
        monkeypatch.delenv("HERMES_BUNDLED_SKILLS", raising=False)
        result = _get_bundled_dir()
        assert result.name == "skills"

    def test_env_var_empty_string_ignored(self, monkeypatch):
        """Empty HERMES_BUNDLED_SKILLS should fall back to default."""
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", "")
        result = _get_bundled_dir()
        assert result.name == "skills"


class TestResetBundledSkill:
    """Covers reset_bundled_skill() — the escape hatch for the 'user-modified' trap."""

    def _setup_bundled(self, tmp_path):
        """Create a minimal bundled skills tree with a single 'google-workspace' skill."""
        bundled = tmp_path / "bundled_skills"
        (bundled / "productivity" / "google-workspace").mkdir(parents=True)
        (bundled / "productivity" / "google-workspace" / "SKILL.md").write_text(
            "---\nname: google-workspace\n---\n# GW v2 (upstream)\n"
        )
        return bundled

    def _patches(self, bundled, skills_dir, manifest_file):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        return stack

    def test_reset_clears_stuck_user_modified_flag(self, tmp_path):
        """The core bug repro: copy-pasted bundled restore doesn't un-stick the flag; reset does."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Simulate the stuck state: user edited the skill on an older bundled version,
        # so manifest has an old origin hash that no longer matches anything on disk.
        dest = skills_dir / "productivity" / "google-workspace"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("---\nname: google-workspace\n---\n# GW v2 (upstream)\n")
        # Stale origin_hash — from some prior bundled version. User "restored" by pasting
        # the current bundled contents, so user_hash == current bundled_hash, but manifest
        # still points at the stale hash → treated as user_modified forever.
        manifest_file.write_text("google-workspace:STALEHASH000000000000000000000000\n")

        with self._patches(bundled, skills_dir, manifest_file):
            # Sanity check: without reset, sync would flag it user_modified
            pre = sync_skills(quiet=True)
            assert "google-workspace" in pre["user_modified"]

            # Reset (no --restore) should clear the manifest entry and re-baseline
            result = reset_bundled_skill("google-workspace", restore=False)

            assert result["ok"] is True
            assert result["action"] == "manifest_cleared"

            # After reset, the manifest should hold the *current* bundled hash
            manifest_after = _read_manifest()
            expected = _dir_hash(bundled / "productivity" / "google-workspace")
            assert manifest_after["google-workspace"] == expected
        # User's copy was preserved (we didn't delete)
        assert dest.exists()
        assert "GW v2" in (dest / "SKILL.md").read_text()

    def test_reset_restore_replaces_user_copy(self, tmp_path):
        """--restore nukes the user's copy and re-copies the bundled version."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        dest = skills_dir / "productivity" / "google-workspace"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("# heavily edited by user\n")
        (dest / "my_custom_file.py").write_text("print('user-added')\n")
        manifest_file.write_text("google-workspace:STALEHASH000000000000000000000000\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = reset_bundled_skill("google-workspace", restore=True)

        assert result["ok"] is True
        assert result["action"] == "restored"
        # User's custom file should be gone
        assert not (dest / "my_custom_file.py").exists()
        # SKILL.md should be the bundled content
        assert "GW v2 (upstream)" in (dest / "SKILL.md").read_text()

    def test_reset_nonexistent_skill_errors_gracefully(self, tmp_path):
        """Resetting a skill that's neither bundled nor in the manifest returns a clear error."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("")

        with self._patches(bundled, skills_dir, manifest_file):
            result = reset_bundled_skill("some-hub-skill", restore=False)

        assert result["ok"] is False
        assert result["action"] == "not_in_manifest"
        assert "not a tracked bundled skill" in result["message"]

    def test_reset_restore_when_bundled_removed_upstream(self, tmp_path):
        """If a skill was removed upstream, --restore should fail with a clear message."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        dest = skills_dir / "productivity" / "ghost-skill"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("---\nname: ghost-skill\n---\n# Ghost\n")
        manifest_file.write_text("ghost-skill:OLDHASH00000000000000000000000000\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = reset_bundled_skill("ghost-skill", restore=True)

        assert result["ok"] is False
        assert result["action"] == "bundled_missing"

    def test_reset_no_op_when_already_clean(self, tmp_path):
        """If manifest has skill but user copy is in-sync, reset still safely clears + re-baselines."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Simulate a clean state — do a fresh sync first
        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)
            pre_manifest = _read_manifest()
            assert "google-workspace" in pre_manifest

            result = reset_bundled_skill("google-workspace", restore=False)

            assert result["ok"] is True
            assert result["action"] == "manifest_cleared"
            # Manifest entry still present (re-baselined), user copy still present
            post_manifest = _read_manifest()
            assert "google-workspace" in post_manifest
        assert (skills_dir / "productivity" / "google-workspace" / "SKILL.md").exists()
