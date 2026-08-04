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
    _dir_hash,
    sync_skills,
    reset_bundled_skill,
    restore_official_optional_skill,
)


class TestReadWriteManifest:
    def test_write_and_read_roundtrip_v2(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        entries = {"zebra": "hash1", "alpha": "hash2", "middle": "hash3"}

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest(entries)
            result = _read_manifest()

        assert result == entries
        # Entries are written sorted for stable diffs.
        names = [line.split(":")[0] for line in manifest_file.read_text().strip().splitlines()]
        assert names == ["alpha", "middle", "zebra"]

        # A missing manifest reads as empty, not an error.
        with patch("tools.skills_sync.MANIFEST_FILE", tmp_path / "nonexistent"):
            assert _read_manifest() == {}

    def test_reads_v1_lines_blanks_and_mixed_formats(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        # v1 format (plain names, no hashes) reads with empty hashes; blank
        # lines are ignored; mixed v1/v2 lines are handled gracefully.
        manifest_file.write_text("old-skill\n\n  \nnew-skill:abc123\n")

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = _read_manifest()

        assert result == {"old-skill": "", "new-skill": "abc123"}


class TestDirHash:
    def test_hash_reflects_content_only(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        for d in (dir_a, dir_b):
            d.mkdir()
            (d / "SKILL.md").write_text("# Test")
            (d / "main.py").write_text("print(1)")
        assert _dir_hash(dir_a) == _dir_hash(dir_b)

        (dir_b / "SKILL.md").write_text("# Version 2")
        assert _dir_hash(dir_a) != _dir_hash(dir_b)

        empty = tmp_path / "empty"
        empty.mkdir()
        assert isinstance(_dir_hash(empty), str) and len(_dir_hash(empty)) == 32
        # A nonexistent dir hashes as empty content rather than raising.
        assert isinstance(_dir_hash(tmp_path / "nope"), str)


class TestDiscoverBundledSkills:
    def test_finds_skill_dirs_and_ignores_non_skills(self, tmp_path):
        (tmp_path / "category" / "skill-a").mkdir(parents=True)
        (tmp_path / "category" / "skill-a" / "SKILL.md").write_text("# Skill A")
        (tmp_path / "skill-b").mkdir()
        (tmp_path / "skill-b" / "SKILL.md").write_text("# Skill B")
        (tmp_path / "not-a-skill").mkdir()
        (tmp_path / "not-a-skill" / "README.md").write_text("Not a skill")
        # .git internals never count as skills.
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        (tmp_path / ".git" / "hooks" / "SKILL.md").write_text("# Fake")

        skills = _discover_bundled_skills(tmp_path)
        assert {name for name, _ in skills} == {"skill-a", "skill-b"}

        assert _discover_bundled_skills(tmp_path / "nonexistent") == []

    def test_ignores_nested_skill_packages_in_support_dirs(self, tmp_path):
        real = tmp_path / "category" / "umbrella"
        nested = real / "references" / "archived-skill"
        nested.mkdir(parents=True)
        (real / "SKILL.md").write_text("---\nname: umbrella\n---\n")
        (nested / "SKILL.md").write_text("---\nname: archived-skill\n---\n")

        assert [name for name, _ in _discover_bundled_skills(tmp_path)] == ["umbrella"]


class TestReadSkillName:
    def test_name_from_frontmatter_with_dir_name_fallbacks(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"

        skill_md.write_text("---\nname: audiocraft-audio-generation\n---\n# Skill")
        assert _read_skill_name(skill_md, "audiocraft") == "audiocraft-audio-generation"

        skill_md.write_text('---\nname: "serving-llms-vllm"\n---\n')
        assert _read_skill_name(skill_md, "vllm") == "serving-llms-vllm"

        skill_md.write_text("# Just a heading\nNo frontmatter here")
        assert _read_skill_name(skill_md, "my-skill") == "my-skill"

        skill_md.write_text("---\nname:\n---\n")
        assert _read_skill_name(skill_md, "fallback") == "fallback"

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
        dest = _compute_relative_dest(Path("/repo/skills/mlops/axolotl"), bundled)
        assert str(dest).endswith("mlops/axolotl")
        # Flat (uncategorized) skills keep their own name.
        assert _compute_relative_dest(Path("/repo/skills/simple"), bundled).name == "simple"


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

    def test_refuses_anything_that_is_not_a_strict_child_of_skills(self, tmp_path):
        """``/``, ``~/.hermes`` itself, a sibling dir, and the skills root
        are all rejected — the root because a ``dest`` that collapses to it
        would wipe every installed skill (the degenerate #48200 path)."""
        from tools.skills_sync import _rmtree_writable

        hermes = tmp_path / "home"
        hermes.mkdir()
        skills = hermes / "skills"
        (skills / "keep").mkdir(parents=True)
        sibling = hermes / "kanban.db"  # any non-skills path
        sibling.mkdir()

        with patch("tools.skills_sync.SKILLS_DIR", skills):
            for target in (Path("/"), hermes, sibling, skills):
                with pytest.raises(ValueError, match="refusing to rmtree"):
                    _rmtree_writable(target)

        assert (skills / "keep").exists()  # nothing was wiped
        assert sibling.exists()

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

    def test_shadowed_skill_skipped_and_not_manifested(self, tmp_path):
        """When an external dir provides the skill, sync must not write it
        locally — nor baseline it in the manifest.

        Recording bundled_hash for a deferred skill would later make the
        loader misclassify the external copy as a user-deleted bundled skill
        and poison update detection.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        ext_dir = self._setup_external(tmp_path)

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("agent.skill_utils.get_external_skills_dirs", return_value=[ext_dir]):
                result = sync_skills(quiet=True)
                manifest = _read_manifest()

        assert "clair-qa" in result["shadowed_by_external"]
        assert "clair-qa" not in result["copied"]
        assert "ascii-art" in result["copied"]
        assert not (skills_dir / "devops" / "clair-qa").exists()
        assert "clair-qa" not in manifest
        # The non-shadowed skill is still synced and baselined normally.
        assert "ascii-art" in manifest


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


class TestRenamedBundledSkillRecovery:
    """Upstream renames/recategorizations must not strand the user's copy.

    ``sync_skills()`` keys the manifest by frontmatter *name*, but computes the
    destination from the bundled *path*. When upstream moves a skill, the name
    still matches while the new dest does not exist yet — the pre-fix code fell
    into its "in manifest but not on disk" branch, misread the skill as
    user-deleted, and left the old directory stranded at the stale path forever.
    """

    def _patches(self, bundled, skills_dir, manifest_file):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(
            patch(
                "tools.skills_sync._get_optional_dir",
                return_value=bundled.parent / "optional-skills",
            )
        )
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        return stack

    def _skill(self, root, rel, body="# Body\n", name="moved-skill"):
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n{body}")
        return d

    def test_rename_relocates_unmodified_copy(self, tmp_path):
        """The stale copy is moved to the new path and updated, not stranded."""
        bundled = tmp_path / "bundled"
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # User's copy sits at the OLD path, byte-identical to what sync wrote.
        old = self._skill(skills_dir, "oldcat/moved-skill")
        origin_hash = _dir_hash(old)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(f"moved-skill:{origin_hash}\n")

        # Upstream moved it to a NEW category and changed the content.
        self._skill(bundled, "newcat/moved-skill", body="# Updated upstream\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)
            # Manifest now tracks the current bundled hash (read inside the
            # patch context — MANIFEST_FILE is a module global).
            recorded = _read_manifest()["moved-skill"]

        new = skills_dir / "newcat" / "moved-skill"
        assert new.exists(), "renamed skill was not relocated to the new path"
        assert not old.exists(), "stale copy left behind — would shadow forever"
        assert "moved-skill" in result["relocated"]
        # Having been relocated, it then takes the normal update path.
        assert "moved-skill" in result["updated"]
        assert "Updated upstream" in (new / "SKILL.md").read_text()
        # Future syncs can now detect further upstream changes.
        assert recorded == _dir_hash(bundled / "newcat" / "moved-skill")

    def test_rename_leaves_user_modified_and_hub_owned_copies_alone(self, tmp_path):
        """A user-edited copy is never moved or overwritten, and a hub-owned
        path is never relocated — the hub lock owns it."""
        bundled = tmp_path / "bundled"
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        edited = self._skill(skills_dir, "oldcat/moved-skill")
        edited_hash = _dir_hash(edited)
        # User then edits their copy, so it no longer matches the origin hash.
        (edited / "SKILL.md").write_text("---\nname: moved-skill\n---\n# MY EDITS\n")

        hub = self._skill(skills_dir, "oldcat/hub-skill", name="hub-skill")
        hub_hash = _dir_hash(hub)

        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(
            f"moved-skill:{edited_hash}\nhub-skill:{hub_hash}\n"
        )
        lock = skills_dir / ".hub" / "lock.json"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(
            json.dumps(
                {
                    "version": 1,
                    "installed": {
                        "hub-skill": {"install_path": "oldcat/hub-skill"}
                    },
                }
            )
        )

        # Upstream moved both into a new category.
        self._skill(bundled, "newcat/moved-skill", body="# Updated upstream\n")
        self._skill(
            bundled, "newcat/hub-skill", body="# Updated upstream\n", name="hub-skill"
        )

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert edited.exists(), "user's modified copy must not be moved"
        assert "MY EDITS" in (edited / "SKILL.md").read_text()
        assert hub.exists(), "hub-installed skill must not be relocated"
        assert "moved-skill" not in result.get("relocated", [])
        assert "hub-skill" not in result.get("relocated", [])

    def test_genuine_user_deletion_still_respected(self, tmp_path):
        """No copy anywhere on disk = a real deletion; must not be resurrected."""
        bundled = tmp_path / "bundled"
        skills_dir = tmp_path / "user_skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = skills_dir / ".bundled_manifest"
        manifest_file.write_text("moved-skill:deadbeef\n")

        self._skill(bundled, "newcat/moved-skill")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert not (skills_dir / "newcat" / "moved-skill").exists()
        assert "moved-skill" not in result["copied"]
        assert "moved-skill" not in result.get("relocated", [])


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

    def test_fresh_install_copies_all_and_records_origin_hashes(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)
            manifest = _read_manifest()

        assert len(result["copied"]) == 2
        assert result["total_bundled"] == 2
        assert result["updated"] == []
        assert result["user_modified"] == []
        assert result["cleaned"] == []
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()
        assert (skills_dir / "old-skill" / "SKILL.md").exists()
        assert (skills_dir / "category" / "DESCRIPTION.md").exists()
        # v2 manifest: non-empty MD5 hashes for both skills.
        assert len(manifest["new-skill"]) == 32
        assert len(manifest["old-skill"]) == 32

    def test_user_deleted_skill_not_re_added_and_stale_entries_cleaned(self, tmp_path):
        """In manifest but not on disk = user deleted it; don't re-add. And a
        manifest entry no longer present in bundled gets cleaned out."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        old_hash = _dir_hash(bundled / "old-skill")
        manifest_file.write_text(f"old-skill:{old_hash}\nremoved-skill:def456\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)
            manifest = _read_manifest()

        assert "new-skill" in result["copied"]
        assert "old-skill" not in result["copied"]
        assert "old-skill" not in result.get("updated", [])
        assert not (skills_dir / "old-skill").exists()
        assert "removed-skill" in result["cleaned"]
        assert "removed-skill" not in manifest


    def test_copy_failure_does_not_poison_manifest_or_destroy_user_copy(self, tmp_path):
        """A failed copytree must leave nothing in the manifest (otherwise the
        next sync treats it as 'user deleted' and never retries) and must not
        destroy the user's existing copy on the update path."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # An already-synced, unmodified copy so the update path runs too.
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        manifest_file.write_text(f"old-skill:{_dir_hash(user_skill)}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            def failing_copytree(src, dst, *a, **kw):
                raise OSError("Simulated disk full")

            with patch("shutil.copytree", side_effect=failing_copytree):
                result = sync_skills(quiet=True)

            assert "new-skill" not in result["copied"]
            assert "old-skill" not in result.get("updated", [])
            assert user_skill.exists(), (
                "Update failure destroyed user's skill copy without replacing it"
            )
            assert "new-skill" not in _read_manifest(), (
                "Failed copy was recorded in manifest — next sync will "
                "treat it as 'user deleted' and never retry"
            )

            # Now run sync again (copytree works this time) — it should retry.
            result2 = sync_skills(quiet=True)
            assert "new-skill" in result2["copied"]
            assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()


class TestGetBundledDir:
    def test_env_var_override_with_default_fallback(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom_skills"
        custom_dir.mkdir()
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(custom_dir))
        assert _get_bundled_dir() == custom_dir

        # Empty or unset falls back to the relative path from __file__.
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", "")
        assert _get_bundled_dir().name == "skills"
        monkeypatch.delenv("HERMES_BUNDLED_SKILLS", raising=False)
        assert _get_bundled_dir().name == "skills"


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


    def test_reset_errors_when_untracked_or_removed_upstream(self, tmp_path):
        """Untracked skills and skills removed upstream both fail clearly."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("")

        with self._patches(bundled, skills_dir, manifest_file):
            untracked = reset_bundled_skill("some-hub-skill", restore=False)

        assert untracked["ok"] is False
        assert untracked["action"] == "not_in_manifest"
        assert "not a tracked bundled skill" in untracked["message"]

        # Tracked in the manifest, but no longer shipped upstream.
        ghost = skills_dir / "productivity" / "ghost-skill"
        ghost.mkdir(parents=True)
        (ghost / "SKILL.md").write_text("---\nname: ghost-skill\n---\n# Ghost\n")
        manifest_file.write_text("ghost-skill:OLDHASH00000000000000000000000000\n")

        with self._patches(bundled, skills_dir, manifest_file):
            removed = reset_bundled_skill("ghost-skill", restore=True)

        assert removed["ok"] is False
        assert removed["action"] == "bundled_missing"

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

    def test_marker_skips_sync_and_removal_seeds_normally(self, tmp_path):
        bundled = tmp_path / "bundled"
        skill = bundled / "category" / "new-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: new-skill\n---\nbody\n")

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        marker = hermes_home / ".no-bundled-skills"
        marker.write_text("opted out\n")

        from contextlib import ExitStack

        def _patches():
            stack = ExitStack()
            stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
            stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
            stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
            stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
            stack.enter_context(patch("tools.skills_sync.HERMES_HOME", hermes_home))
            return stack

        with _patches():
            opted_out = sync_skills(quiet=True)

        # Opt-out signalled, nothing copied, nothing written to disk.
        assert opted_out["skipped_opt_out"] is True
        assert opted_out["copied"] == []
        assert opted_out["total_bundled"] == 0
        assert not (skills_dir / "category" / "new-skill" / "SKILL.md").exists()

        marker.unlink()
        with _patches():
            seeded = sync_skills(quiet=True)

        assert seeded.get("skipped_opt_out") is not True
        assert "new-skill" in seeded["copied"]
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
