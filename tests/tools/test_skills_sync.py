"""Tests for tools/skills_sync.py — manifest-based skill seeding and updating."""

import shutil
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from tools.skills_sync import (
    _get_bundled_dir,
    _read_manifest,
    _read_skill_name,
    _write_manifest,
    _discover_bundled_skills,
    _compute_relative_dest,
    _content_hash,
    _dir_hash,
    sync_skills,
    reset_bundled_skill,
    restore_official_optional_skill,
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
        assert str(dest).endswith("mlops/axolotl")

    def test_flat_skill(self):
        bundled = Path("/repo/skills")
        skill_dir = Path("/repo/skills/simple")
        dest = _compute_relative_dest(skill_dir, bundled)
        assert dest.name == "simple"


class TestRmtreeWritableScopeGuard:
    """``_rmtree_writable`` must refuse to remove anything outside
    ``HERMES_HOME/skills/``.

    The previous implementation called ``shutil.rmtree(path)`` on whatever
    argument the caller passed. If any of the five call sites in
    ``tools/skills_sync.py`` ever computes a path outside the skills
    root — through a bad join, a missing default, a malicious
    bundled-manifest entry, or a stale path in scope after an
    exception — the result is a silent ``shutil.rmtree(~/.hermes/)``
    that destroys the user's ``.env``, ``MEMORY.md``, ``kanban.db``,
    custom skills, scripts, and the rest of the install in one go
    (#48200).

    The scope guard turns that into a loud ``ValueError`` so the
    failure is observable, reproducible, and recoverable rather than
    a data-loss incident.
    """

    def test_refuses_root_path(self, tmp_path):
        """``Path('/')`` is the entire filesystem — must always be rejected."""
        from tools.skills_sync import _rmtree_writable, SKILLS_DIR

        skills = tmp_path / "skills"
        skills.mkdir()
        with patch("tools.skills_sync.SKILLS_DIR", skills):
            with pytest.raises(ValueError, match="refusing to rmtree"):
                _rmtree_writable(Path("/"))

    def test_refuses_hermes_home_itself(self, tmp_path):
        """``~/.hermes/`` itself is what the #48200 wipe destroyed."""
        from tools.skills_sync import _rmtree_writable

        hermes = tmp_path / "home"
        hermes.mkdir()
        (hermes / "skills").mkdir()
        with patch("tools.skills_sync.SKILLS_DIR", hermes / "skills"):
            with pytest.raises(ValueError, match="refusing to rmtree"):
                _rmtree_writable(hermes)

    def test_refuses_sibling_directory(self, tmp_path):
        """A directory that is a sibling of SKILLS_DIR (e.g. a wrong
        ``bundled_dir`` computation) must be rejected, not silently rmtree'd.
        """
        from tools.skills_sync import _rmtree_writable

        hermes = tmp_path / "home"
        hermes.mkdir()
        skills = hermes / "skills"
        skills.mkdir()
        not_skills = hermes / "kanban.db"  # any non-skills path
        not_skills.mkdir()
        with patch("tools.skills_sync.SKILLS_DIR", skills):
            with pytest.raises(ValueError, match="refusing to rmtree"):
                _rmtree_writable(not_skills)

    def test_refuses_skills_root_itself(self, tmp_path):
        """The skills root directory itself must be refused.

        No caller in skills_sync.py ever passes SKILLS_DIR directly — every
        site passes a skill subdirectory or its ``.bak`` sibling. Removing
        the root would wipe every installed skill, and a ``dest`` that
        collapses to the root is exactly the degenerate path #48200 guards
        against. Require a strict-child relationship.
        """
        from tools.skills_sync import _rmtree_writable

        skills = tmp_path / "skills"
        (skills / "keep").mkdir(parents=True)
        with patch("tools.skills_sync.SKILLS_DIR", skills):
            with pytest.raises(ValueError, match="refusing to rmtree"):
                _rmtree_writable(skills)
        assert (skills / "keep").exists()  # nothing was wiped

    def test_allows_subdirectory_of_skills(self, tmp_path):
        """Any directory strictly under SKILLS_DIR is allowed."""
        from tools.skills_sync import _rmtree_writable

        skills = tmp_path / "skills"
        skills.mkdir()
        sub = skills / "category" / "old-skill"
        sub.mkdir(parents=True)
        (sub / "SKILL.md").write_text("# old")

        with patch("tools.skills_sync.SKILLS_DIR", skills):
            _rmtree_writable(sub)

        assert skills.exists()
        assert not sub.exists()


class TestExternalDirsIndexing:
    """Tests for external_dirs awareness in sync_skills (#28126)."""

    def _setup_bundled(self, tmp_path):
        """Create a fake bundled skills directory."""
        bundled = tmp_path / "bundled_skills"
        (bundled / "devops" / "clair-qa").mkdir(parents=True)
        (bundled / "devops" / "clair-qa" / "SKILL.md").write_text("# bundled clair")
        (bundled / "creative" / "ascii-art").mkdir(parents=True)
        (bundled / "creative" / "ascii-art" / "SKILL.md").write_text("# bundled ascii")
        return bundled

    def _setup_external(self, tmp_path):
        """Create a fake external skills directory."""
        ext_dir = tmp_path / "external_skills"
        (ext_dir / "devops" / "clair-qa").mkdir(parents=True)
        (ext_dir / "devops" / "clair-qa" / "SKILL.md").write_text("# external clair")
        (ext_dir / "devops" / "clair-qa" / "main.py").write_text("print('ext')")
        return ext_dir

    def _patches(self, bundled, skills_dir, manifest_file):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        return stack

    def test_shadowed_skill_skipped_and_deferred(self, tmp_path):
        """When external dir provides the skill, sync_skills should not write it locally."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        ext_dir = self._setup_external(tmp_path)

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("agent.skill_utils.get_external_skills_dirs", return_value=[ext_dir]):
                result = sync_skills(quiet=True)

        assert "clair-qa" in result["shadowed_by_external"]
        assert "clair-qa" not in result["copied"]
        assert "ascii-art" in result["copied"]
        assert not (skills_dir / "devops" / "clair-qa").exists()

    def test_shadowed_skill_not_recorded_in_manifest(self, tmp_path):
        """A skill we never wrote locally must NOT be baselined in the manifest.

        Recording bundled_hash for a deferred skill would later make the
        loader misclassify the external copy as a user-deleted bundled skill
        and poison update detection. The shadowed name stays out of the
        manifest entirely.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        ext_dir = self._setup_external(tmp_path)

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("agent.skill_utils.get_external_skills_dirs", return_value=[ext_dir]):
                sync_skills(quiet=True)
                manifest = _read_manifest()

        assert "clair-qa" not in manifest
        # The non-shadowed skill is still synced and baselined normally.
        assert "ascii-art" in manifest

    def test_stale_shadow_self_healed(self, tmp_path):
        """A byte-identical-to-bundled local shadow is removed when the same
        skill is now provided by external_dirs (heals profiles broken by an
        earlier sync that ran before external_dirs was configured)."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        ext_dir = self._setup_external(tmp_path)

        # Pre-seed a shadow identical to the bundled source.
        shadow = skills_dir / "devops" / "clair-qa"
        shadow.mkdir(parents=True)
        (shadow / "SKILL.md").write_text("# bundled clair")

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("agent.skill_utils.get_external_skills_dirs", return_value=[ext_dir]):
                result = sync_skills(quiet=True)

        assert "clair-qa" in result["shadowed_by_external"]
        assert not shadow.exists()

    def test_user_customized_shadow_preserved(self, tmp_path):
        """A local skill that DIFFERS from bundled is the user's own — it must
        never be deleted even when external_dirs provides the same name."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        ext_dir = self._setup_external(tmp_path)

        custom = skills_dir / "devops" / "clair-qa"
        custom.mkdir(parents=True)
        (custom / "SKILL.md").write_text("# my own customized clair")

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("agent.skill_utils.get_external_skills_dirs", return_value=[ext_dir]):
                result = sync_skills(quiet=True)

        assert "clair-qa" in result["shadowed_by_external"]
        assert custom.exists()
        assert (custom / "SKILL.md").read_text() == "# my own customized clair"

    def test_no_external_dirs_unchanged(self, tmp_path):
        """Without external_dirs, all bundled skills should be copied normally."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("agent.skill_utils.get_external_skills_dirs", return_value=[]):
                result = sync_skills(quiet=True)

        assert "clair-qa" in result["copied"]
        assert "ascii-art" in result["copied"]
        assert result["shadowed_by_external"] == []


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
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        return stack

    def test_suppressed_builtin_not_reseeded(self, tmp_path):
        """A curator-pruned built-in in the suppression list must NOT be
        re-copied on sync — that's what makes the prune durable across updates.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file), \
                patch("tools.skills_sync._read_suppressed_names", return_value={"old-skill"}):
            result = sync_skills(quiet=True)

        # old-skill is suppressed → skipped, not copied.
        assert "old-skill" in result["suppressed"]
        assert "old-skill" not in result["copied"]
        assert not (skills_dir / "old-skill").exists()
        # The non-suppressed bundled skill is still copied normally.
        assert "new-skill" in result["copied"]
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()

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

    # ── Optional → bundled migration ───────────────────────────────────────
    def _seed_optional_install(
        self, skills_dir, install_path, on_disk_text, *, source="official",
        recorded_from_text=None,
    ):
        """Create an on-disk skill copy plus a matching hub lock entry.

        ``recorded_from_text`` seeds the lock's ``content_hash`` from a
        *different* directory content than what lands on disk, so tests can
        model a user who edited the installed copy after installing it.
        """
        dest = skills_dir / Path(*install_path.split("/"))
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text(on_disk_text)

        if recorded_from_text is None:
            recorded_hash = _content_hash(dest)
        else:
            tmp = skills_dir.parent / "_recorded_probe"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            (tmp / "SKILL.md").write_text(recorded_from_text)
            recorded_hash = _content_hash(tmp)
            shutil.rmtree(tmp)

        lock_name = Path(install_path).name
        lock = skills_dir / ".hub" / "lock.json"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({
            "version": 1,
            "installed": {
                lock_name: {
                    "source": source,
                    "identifier": f"{source}/{install_path}",
                    "trust_level": "builtin" if source == "official" else "community",
                    "scan_verdict": "clean",
                    "content_hash": recorded_hash,
                    "install_path": install_path,
                    "files": ["SKILL.md"],
                    "metadata": {},
                    "installed_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            },
        }, indent=2))
        return dest

    def test_migrates_unmodified_official_optional_install(self, tmp_path):
        """An unmodified official optional install is replaced by the bundled
        version and its stale hub-lock entry is dropped."""
        bundled = self._setup_bundled(tmp_path)
        (bundled / "email" / "agentmail").mkdir(parents=True)
        (bundled / "email" / "agentmail" / "SKILL.md").write_text(
            "---\nname: agentmail\n---\n# CLI-first (bundled)\n"
        )

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        dest = self._seed_optional_install(
            skills_dir, "email/agentmail",
            "---\nname: agentmail\n---\n# MCP-first (optional install)\n",
        )

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert result["migrated"] == ["agentmail"]
        assert "# CLI-first (bundled)" in (dest / "SKILL.md").read_text()

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            manifest = _read_manifest()
        assert manifest["agentmail"] == _dir_hash(bundled / "email" / "agentmail")

        lock = json.loads((skills_dir / ".hub" / "lock.json").read_text())
        assert "agentmail" not in lock["installed"], (
            "migrated skill's stale official hub-lock entry must be removed"
        )

    def test_migration_is_idempotent(self, tmp_path):
        """A second sync after migration must not re-flag or re-migrate."""
        bundled = self._setup_bundled(tmp_path)
        (bundled / "email" / "agentmail").mkdir(parents=True)
        (bundled / "email" / "agentmail" / "SKILL.md").write_text(
            "---\nname: agentmail\n---\n# CLI-first (bundled)\n"
        )
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        self._seed_optional_install(
            skills_dir, "email/agentmail",
            "---\nname: agentmail\n---\n# MCP-first\n",
        )

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)
            result2 = sync_skills(quiet=True)

        assert result2["migrated"] == []
        assert result2["user_modified"] == []
        assert "agentmail" not in result2["copied"]

    def test_does_not_migrate_user_modified_official_optional_install(self, tmp_path):
        """A user-edited official optional copy is preserved, not migrated."""
        bundled = self._setup_bundled(tmp_path)
        (bundled / "email" / "agentmail").mkdir(parents=True)
        (bundled / "email" / "agentmail" / "SKILL.md").write_text(
            "---\nname: agentmail\n---\n# CLI-first (bundled)\n"
        )
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        # On disk differs from what the lock recorded at install time → edited.
        dest = self._seed_optional_install(
            skills_dir, "email/agentmail",
            "---\nname: agentmail\n---\n# MY LOCAL EDITS\n",
            recorded_from_text="---\nname: agentmail\n---\n# MCP-first (pristine)\n",
        )

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert result["migrated"] == []
        assert "# MY LOCAL EDITS" in (dest / "SKILL.md").read_text()
        lock = json.loads((skills_dir / ".hub" / "lock.json").read_text())
        assert "agentmail" in lock["installed"], (
            "user-modified skill keeps its hub-lock ownership"
        )

    def test_does_not_migrate_non_official_collision(self, tmp_path):
        """A community/custom hub install at the collision path is not migrated."""
        bundled = self._setup_bundled(tmp_path)
        (bundled / "email" / "agentmail").mkdir(parents=True)
        (bundled / "email" / "agentmail" / "SKILL.md").write_text(
            "---\nname: agentmail\n---\n# CLI-first (bundled)\n"
        )
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        dest = self._seed_optional_install(
            skills_dir, "email/agentmail",
            "---\nname: agentmail\n---\n# community fork\n",
            source="community",
        )

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert result["migrated"] == []
        assert "# community fork" in (dest / "SKILL.md").read_text()

    def test_failed_migration_preserves_original_and_lock(self, tmp_path):
        """If replacing the copy fails, the original copy, lock entry, and an
        un-poisoned manifest all survive for the next sync to retry."""
        bundled = self._setup_bundled(tmp_path)
        (bundled / "email" / "agentmail").mkdir(parents=True)
        (bundled / "email" / "agentmail" / "SKILL.md").write_text(
            "---\nname: agentmail\n---\n# CLI-first (bundled)\n"
        )
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        dest = self._seed_optional_install(
            skills_dir, "email/agentmail",
            "---\nname: agentmail\n---\n# MCP-first\n",
        )

        original_copytree = shutil.copytree

        def failing_copytree(src, dst, *a, **kw):
            if "agentmail" in str(src):
                raise OSError("Simulated disk full")
            return original_copytree(src, dst, *a, **kw)

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("shutil.copytree", side_effect=failing_copytree):
                result = sync_skills(quiet=True)
            manifest = _read_manifest()

        assert result["migrated"] == []
        assert "# MCP-first" in (dest / "SKILL.md").read_text()
        assert "agentmail" not in manifest, "failed migration must not poison manifest"
        lock = json.loads((skills_dir / ".hub" / "lock.json").read_text())
        assert "agentmail" in lock["installed"]

    def test_migration_prints_notice(self, tmp_path, capsys):
        """Non-quiet sync announces the migration instead of doing it silently."""
        bundled = self._setup_bundled(tmp_path)
        (bundled / "email" / "agentmail").mkdir(parents=True)
        (bundled / "email" / "agentmail" / "SKILL.md").write_text(
            "---\nname: agentmail\n---\n# CLI-first (bundled)\n"
        )
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        self._seed_optional_install(
            skills_dir, "email/agentmail",
            "---\nname: agentmail\n---\n# MCP-first\n",
        )

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=False)

        out = capsys.readouterr().out
        assert "agentmail" in out
        assert "migrated" in out.lower()

    def test_backfills_official_optional_provenance_for_existing_identical_skill(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        optional = tmp_path / "optional-skills"
        optional_skill = optional / "mlops" / "training" / "trl-fine-tuning"
        optional_skill.mkdir(parents=True)
        (optional_skill / "SKILL.md").write_text(
            "---\nname: fine-tuning-with-trl\n---\n# TRL\n"
        )
        (optional_skill / "references").mkdir()
        (optional_skill / "references" / "api.md").write_text("api\n")

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        active = skills_dir / "mlops" / "training" / "trl-fine-tuning"
        active.mkdir(parents=True)
        (active / "SKILL.md").write_text(
            "---\nname: fine-tuning-with-trl\n---\n# TRL\n"
        )
        (active / "references").mkdir()
        (active / "references" / "api.md").write_text("api\n")

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._get_optional_dir", return_value=optional):
                result = sync_skills(quiet=True)

        assert result["optional_provenance_backfilled"] == ["trl-fine-tuning"]
        lock_path = skills_dir / ".hub" / "lock.json"
        data = json.loads(lock_path.read_text())
        entry = data["installed"]["trl-fine-tuning"]
        assert entry["source"] == "official"
        assert entry["identifier"] == "official/mlops/training/trl-fine-tuning"
        assert entry["trust_level"] == "builtin"
        assert entry["install_path"] == "mlops/training/trl-fine-tuning"

    def test_does_not_backfill_optional_provenance_for_modified_skill(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        optional = tmp_path / "optional-skills"
        optional_skill = optional / "mlops" / "training" / "trl-fine-tuning"
        optional_skill.mkdir(parents=True)
        (optional_skill / "SKILL.md").write_text("# upstream optional\n")

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        active = skills_dir / "mlops" / "training" / "trl-fine-tuning"
        active.mkdir(parents=True)
        (active / "SKILL.md").write_text("# user modified\n")

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._get_optional_dir", return_value=optional):
                result = sync_skills(quiet=True)

        assert result["optional_provenance_backfilled"] == []
        assert not (skills_dir / ".hub" / "lock.json").exists()

    def test_repair_official_optional_restores_reorganized_skill_with_backup(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        optional = tmp_path / "optional-skills"
        optional_skill = optional / "mlops" / "training" / "trl-fine-tuning"
        optional_skill.mkdir(parents=True)
        (optional_skill / "SKILL.md").write_text(
            "---\nname: fine-tuning-with-trl\n---\n# Official TRL\n"
        )

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        wrong = skills_dir / "mlops" / "trl-fine-tuning"
        wrong.mkdir(parents=True)
        (wrong / "SKILL.md").write_text(
            "---\nname: fine-tuning-with-trl\n---\n# Curator mangled\n"
        )

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._get_optional_dir", return_value=optional):
                result = restore_official_optional_skill("fine-tuning-with-trl", restore=True)

        canonical = skills_dir / "mlops" / "training" / "trl-fine-tuning"
        assert result["ok"] is True
        assert result["restored"] == ["trl-fine-tuning"]
        assert result["backed_up"] == ["mlops/trl-fine-tuning"]
        assert "Official TRL" in (canonical / "SKILL.md").read_text()
        assert not wrong.exists()
        assert (Path(result["backup_dir"]) / "mlops" / "trl-fine-tuning" / "SKILL.md").exists()

        data = json.loads((skills_dir / ".hub" / "lock.json").read_text())
        assert data["installed"]["trl-fine-tuning"]["source"] == "official"
        assert data["installed"]["trl-fine-tuning"]["install_path"] == "mlops/training/trl-fine-tuning"

    def test_repair_official_optional_without_restore_does_not_replace_modified_copy(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        optional = tmp_path / "optional-skills"
        optional_skill = optional / "mlops" / "training" / "trl-fine-tuning"
        optional_skill.mkdir(parents=True)
        (optional_skill / "SKILL.md").write_text("# official\n")

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        canonical = skills_dir / "mlops" / "training" / "trl-fine-tuning"
        canonical.mkdir(parents=True)
        (canonical / "SKILL.md").write_text("# modified\n")

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._get_optional_dir", return_value=optional):
                result = restore_official_optional_skill("trl-fine-tuning", restore=False)

        assert result["ok"] is True
        assert result["restored"] == []
        assert result["backfilled"] == []
        assert (canonical / "SKILL.md").read_text() == "# modified\n"
        assert not (skills_dir / ".hub" / "lock.json").exists()

    def test_nonexistent_bundled_dir(self, tmp_path):
        with patch("tools.skills_sync._get_bundled_dir", return_value=tmp_path / "nope"):
            result = sync_skills(quiet=True)
        assert result == {
            "copied": [], "updated": [], "skipped": 0,
            "user_modified": [], "migrated": [], "cleaned": [], "suppressed": [],
            "total_bundled": 0, "optional_provenance_backfilled": [],
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
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
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

    def test_reset_restore_succeeds_on_readonly_nix_tree(self, tmp_path):
        """#34972: --restore must succeed even when the user copy is a fully
        read-only tree (r-xr-xr-x dirs + files), as produced by copying a
        Nix-store source. The manifest is re-baselined and bundled re-copied."""
        import os
        import stat

        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        dest = skills_dir / "productivity" / "google-workspace"
        sub = dest / "references"
        sub.mkdir(parents=True)
        (dest / "SKILL.md").write_text("# user version\n")
        (sub / "ref.md").write_text("# nested ref\n")
        manifest_file.write_text(
            "google-workspace:STALEHASH000000000000000000000000\n"
        )

        # Read-only files AND directories — the real Nix-store case.
        ro_dir = (
            stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
            | stat.S_IROTH | stat.S_IXOTH
        )
        os.chmod(sub / "ref.md", stat.S_IREAD)
        os.chmod(dest / "SKILL.md", stat.S_IREAD)
        os.chmod(sub, ro_dir)
        os.chmod(dest, ro_dir)

        try:
            with self._patches(bundled, skills_dir, manifest_file):
                result = reset_bundled_skill("google-workspace", restore=True)

            assert result["ok"] is True
            assert result["action"] == "restored"
            # Bundled version was re-copied over the (deleted) user copy.
            assert "upstream" in (dest / "SKILL.md").read_text()
            # The read-only nested user dir/file was fully removed, not left behind.
            assert not (sub / "ref.md").exists()
            # sync ran and re-copied the skill (not stuck in limbo).
            assert "google-workspace" in result["synced"]["copied"]
        finally:
            # Restore perms so tmp_path teardown can remove anything left.
            for p in (sub, dest):
                if p.exists():
                    os.chmod(p, stat.S_IRWXU)

    def test_reset_restore_preserves_manifest_on_rmtree_failure(self, tmp_path):
        """#34972: when the user copy genuinely cannot be removed, the manifest
        entry must NOT be deleted — otherwise the skill enters a limbo state
        where future syncs silently skip it forever."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        dest = skills_dir / "productivity" / "google-workspace"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("# user version\n")
        manifest_file.write_text(
            "google-workspace:STALEHASH000000000000000000000000\n"
        )

        # Simulate an unremovable tree (e.g. a busy mountpoint or a path even
        # chmod can't rescue) by making the removal helper raise.
        def _boom(_path):
            raise PermissionError(13, "Permission denied")

        with self._patches(bundled, skills_dir, manifest_file), patch(
            "tools.skills_sync._rmtree_writable", side_effect=_boom
        ):
            result = reset_bundled_skill("google-workspace", restore=True)

        # Restore failed, and the manifest must be left untouched.
        assert result["ok"] is False
        assert result["action"] == "not_reset"
        assert "Manifest entry preserved" in result["message"]
        manifest_after = manifest_file.read_text()
        assert "google-workspace" in manifest_after
        # User copy is still on disk (we changed nothing).
        assert (dest / "SKILL.md").exists()


class TestNoBundledSkillsOptOut:
    """The .no-bundled-skills marker makes sync_skills() a no-op.

    This is what `hermes profile create --no-skills` (named profiles) and the
    installer's `--no-skills` flag (default ~/.hermes) rely on so bundled
    skills are never seeded at install time NOR re-injected by `hermes update`.
    """

    def _setup_bundled(self, tmp_path):
        bundled = tmp_path / "bundled"
        skill = bundled / "category" / "new-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: new-skill\n---\nbody\n")
        return bundled

    def test_marker_skips_sync(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        (hermes_home / ".no-bundled-skills").write_text("opted out\n")

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file), \
             patch("tools.skills_sync.HERMES_HOME", hermes_home):
            result = sync_skills(quiet=True)

        # Opt-out signalled, nothing copied, nothing written to disk.
        assert result["skipped_opt_out"] is True
        assert result["copied"] == []
        assert result["total_bundled"] == 0
        assert not (skills_dir / "category" / "new-skill" / "SKILL.md").exists()

    def test_no_marker_seeds_normally(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        # No marker written.

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file), \
             patch("tools.skills_sync.HERMES_HOME", hermes_home):
            result = sync_skills(quiet=True)

        assert result.get("skipped_opt_out") is not True
        assert "new-skill" in result["copied"]
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()


class TestOptOutToggleAndRemove:
    """`hermes skills opt-out/opt-in` core: marker toggle + safe removal."""

    def _setup_bundled(self, tmp_path):
        bundled = tmp_path / "bundled"
        for n in ("alpha", "beta"):
            d = bundled / n
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {n}\n---\nbody {n}\n")
        return bundled

    def test_marker_toggle(self, tmp_path):
        from tools.skills_sync import (
            set_bundled_skills_opt_out, is_bundled_skills_opt_out,
        )
        home = tmp_path / "home"
        home.mkdir()
        with patch("tools.skills_sync.HERMES_HOME", home):
            assert is_bundled_skills_opt_out() is False
            r = set_bundled_skills_opt_out(True)
            assert r["ok"] and r["changed"]
            assert is_bundled_skills_opt_out() is True
            # idempotent
            r2 = set_bundled_skills_opt_out(True)
            assert r2["ok"] and r2["changed"] is False
            # opt back in
            r3 = set_bundled_skills_opt_out(False)
            assert r3["ok"] and r3["changed"]
            assert is_bundled_skills_opt_out() is False

    def test_remove_keeps_user_modified(self, tmp_path):
        from tools.skills_sync import (
            sync_skills, remove_pristine_bundled_skills,
        )
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        home = tmp_path / "home"
        home.mkdir()
        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file), \
             patch("tools.skills_sync.HERMES_HOME", home):
            sync_skills(quiet=True)
            # User edits 'beta'
            (skills_dir / "beta" / "SKILL.md").write_text("---\nname: beta\n---\nEDITED\n")
            # A hand-written, non-bundled skill must also survive.
            (skills_dir / "mine").mkdir()
            (skills_dir / "mine" / "SKILL.md").write_text("---\nname: mine\n---\nlocal\n")

            preview = remove_pristine_bundled_skills(dry_run=True)
            assert "alpha" in preview["removed"]
            assert "beta" not in preview["removed"]

            result = remove_pristine_bundled_skills(dry_run=False)
            assert "alpha" in result["removed"]
            assert not (skills_dir / "alpha").exists()
            # user-modified bundled skill kept
            assert (skills_dir / "beta" / "SKILL.md").exists()
            assert "EDITED" in (skills_dir / "beta" / "SKILL.md").read_text()
            # non-bundled local skill never considered
            assert (skills_dir / "mine" / "SKILL.md").exists()


class TestUpdateBackupRecovery:
    """Regression tests for backup handling in the bundled-update path.

    Covers three failure modes around ``dest.with_suffix(".bak")``:
    a stale backup poisoning the next update's move/restore, an orphaned
    backup (crash between move and copytree) being misread as a user
    deletion, and a partially-written dest blocking restore-on-failure.
    """

    def _setup(self, tmp_path, bundled_text="# Old v2 (updated)"):
        """Bundled dir with one flat skill, plus user dirs."""
        bundled = tmp_path / "bundled_skills"
        (bundled / "old-skill").mkdir(parents=True)
        (bundled / "old-skill" / "SKILL.md").write_text(bundled_text)
        skills_dir = tmp_path / "user_skills"
        skills_dir.mkdir()
        manifest_file = skills_dir / ".bundled_manifest"
        return bundled, skills_dir, manifest_file

    def _patches(self, bundled, skills_dir, manifest_file):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        return stack

    def _seed_synced_copy(self, skills_dir, manifest_file, text="# Old v1"):
        """User copy of old-skill whose hash matches the manifest origin."""
        dest = skills_dir / "old-skill"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text(text)
        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest({"old-skill": _dir_hash(dest)})
        return dest

    def test_stale_backup_does_not_poison_failed_update(self, tmp_path):
        """A leftover .bak must not nest the live copy or corrupt restore.

        With a stale ``old-skill.bak`` present, ``shutil.move(dest, backup)``
        moves the live copy *inside* the stale dir. If copytree then fails,
        restore drags the stale junk (with the live copy nested in it) back
        to dest — corrupting the skill and wedging it as "user-modified".
        """
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        dest = self._seed_synced_copy(skills_dir, manifest_file)

        stale = skills_dir / "old-skill.bak"
        stale.mkdir()
        (stale / "SKILL.md").write_text("# stale junk from an earlier failure")

        def _boom(src, dst, **kwargs):
            raise OSError("simulated copy failure")

        with self._patches(bundled, skills_dir, manifest_file), \
                patch("tools.skills_sync.shutil.copytree", side_effect=_boom):
            sync_skills(quiet=True)

        # The live copy must survive the failed update untouched...
        assert (dest / "SKILL.md").read_text() == "# Old v1"
        # ...not be nested inside recycled stale-backup content.
        assert not (dest / "old-skill").exists()
        # And no backup directory may linger.
        assert not (skills_dir / "old-skill.bak").exists()

    def test_orphaned_backup_is_recovered_not_treated_as_deleted(self, tmp_path):
        """Crash between move and copytree must not lose the skill.

        After such a crash, dest is gone and the user's only copy sits in
        ``old-skill.bak``. The sync loop's "in manifest but not on disk"
        branch reads that as a deliberate user deletion and skips — the
        skill silently vanishes. It must instead be recovered and updated.
        """
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        dest = self._seed_synced_copy(skills_dir, manifest_file)
        # Simulate the crash: dest was moved aside, new copy never arrived.
        shutil.move(str(dest), str(skills_dir / "old-skill.bak"))

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        # Recovered and then updated to the new bundled version in one run.
        assert (dest / "SKILL.md").exists()
        assert (dest / "SKILL.md").read_text() == "# Old v2 (updated)"
        assert "old-skill" in result["updated"]
        assert not (skills_dir / "old-skill.bak").exists()

    def test_partial_copy_failure_restores_original(self, tmp_path):
        """A half-written dest must not block restore-on-failure.

        If copytree dies after creating dest, the ``not dest.exists()``
        guard skips the restore: the user keeps a broken partial skill,
        the .bak lingers, and the partial hash wedges the skill as
        "user-modified" on every later sync.
        """
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        dest = self._seed_synced_copy(skills_dir, manifest_file)

        def _partial_then_fail(src, dst, **kwargs):
            Path(dst).mkdir(parents=True, exist_ok=True)
            (Path(dst) / "PARTIAL").write_text("half-written")
            raise OSError("simulated failure mid-copy")

        with self._patches(bundled, skills_dir, manifest_file), \
                patch("tools.skills_sync.shutil.copytree", side_effect=_partial_then_fail):
            sync_skills(quiet=True)

        # Original content restored, partial debris and backup gone.
        assert (dest / "SKILL.md").read_text() == "# Old v1"
        assert not (dest / "PARTIAL").exists()
        assert not (skills_dir / "old-skill.bak").exists()

        # And the skill is not wedged: a later normal sync updates cleanly.
        with self._patches(bundled, skills_dir, manifest_file):
            result2 = sync_skills(quiet=True)
        assert "old-skill" in result2["updated"]
        assert result2["user_modified"] == []
