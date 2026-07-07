"""Tests for discovering and diffing user-modified bundled skills.

`hermes update` keeps (does not overwrite) bundled skills the user edited
locally, but historically only printed a *count* — there was no way to find
which skills, or see what changed. These tests cover the two helpers that close
that gap, exercising the real sync pipeline (no mocks of the comparison logic):

* ``list_user_modified_bundled_skills()`` — the discovery half of the exact
  test the sync loop uses to decide what to skip.
* ``diff_bundled_skill()`` — a unified diff of the user copy vs the stock copy.

Revert already exists (``reset_bundled_skill``); the last test confirms it
clears the modified state so the two stay consistent.
"""

from contextlib import ExitStack
from unittest.mock import patch

from tools.skills_sync import (
    sync_skills,
    reset_bundled_skill,
    list_user_modified_bundled_skills,
    diff_bundled_skill,
)


def _make_bundled(tmp_path):
    """A fake bundled skills tree with one skill: category/foo."""
    bundled = tmp_path / "bundled_skills"
    foo = bundled / "category" / "foo"
    foo.mkdir(parents=True)
    (foo / "SKILL.md").write_text("---\nname: foo\n---\n# Foo Skill\n")
    (foo / "helper.py").write_text("print('stock')\n")
    return bundled


def _patches(bundled, skills_dir, manifest_file):
    stack = ExitStack()
    stack.enter_context(
        patch("tools.skills_sync._get_bundled_dir", return_value=bundled)
    )
    stack.enter_context(
        patch(
            "tools.skills_sync._get_optional_dir",
            return_value=bundled.parent / "optional-skills",
        )
    )
    stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
    stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
    return stack


def _env(tmp_path):
    bundled = _make_bundled(tmp_path)
    skills_dir = tmp_path / "user_skills"
    manifest_file = skills_dir / ".bundled_manifest"
    return bundled, skills_dir, manifest_file


def test_pristine_skill_is_not_listed_as_modified(tmp_path):
    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        assert list_user_modified_bundled_skills() == []


def test_edited_skill_is_listed_as_modified(tmp_path):
    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        (skills_dir / "category" / "foo" / "helper.py").write_text("print('mine')\n")

        modified = list_user_modified_bundled_skills()
        names = [m["name"] for m in modified]
        assert names == ["foo"]
        entry = modified[0]
        assert entry["dest"] == skills_dir / "category" / "foo"
        assert entry["bundled_src"] == bundled / "category" / "foo"


def test_diff_reports_no_changes_when_pristine(tmp_path):
    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        result = diff_bundled_skill("foo")
        assert result["ok"] is True
        assert result["modified"] is False
        assert result["diffs"] == []


def test_diff_shows_modified_and_added_files(tmp_path):
    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        user_foo = skills_dir / "category" / "foo"
        (user_foo / "helper.py").write_text("print('mine')\n")
        (user_foo / "extra.txt").write_text("local note\n")

        result = diff_bundled_skill("foo")
        assert result["ok"] is True
        assert result["modified"] is True

        by_path = {d["path"]: d for d in result["diffs"]}
        assert by_path["helper.py"]["status"] == "modified"
        # The unified diff shows the user's line replacing the stock line.
        assert "print('mine')" in by_path["helper.py"]["diff"]
        assert "print('stock')" in by_path["helper.py"]["diff"]
        # A file only in the user copy is reported as added.
        assert by_path["extra.txt"]["status"] == "added"


def test_diff_unknown_skill_is_not_ok(tmp_path):
    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        result = diff_bundled_skill("does-not-exist")
        assert result["ok"] is False
        assert result["found"] is False


def test_reset_clears_modified_state(tmp_path):
    """Revert (existing) and discovery (new) must agree: after reset, not modified."""
    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        (skills_dir / "category" / "foo" / "helper.py").write_text("print('mine')\n")
        assert [m["name"] for m in list_user_modified_bundled_skills()] == ["foo"]

        # Restore from the stock source, then it must no longer be flagged.
        result = reset_bundled_skill("foo", restore=True)
        assert result["ok"] is True
        assert list_user_modified_bundled_skills() == []


# ---------------------------------------------------------------------------
# skill_diff  (agent-callable wrapper around diff_bundled_skill)
# ---------------------------------------------------------------------------


def test_skill_diff_pristine_skill(tmp_path):
    """skill_diff reports no changes for a pristine bundled skill."""
    import json

    from tools.skills_tool import skill_diff

    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        raw = skill_diff("foo")
    result = json.loads(raw)
    assert result["ok"] is True
    assert result["found"] is True
    assert result["modified"] is False
    assert result["diffs"] == []


def test_skill_diff_modified_skill(tmp_path):
    """skill_diff reports diffs when a bundled skill has been edited."""
    import json

    from tools.skills_tool import skill_diff

    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        (skills_dir / "category" / "foo" / "helper.py").write_text("print('mine')\n")
        (skills_dir / "category" / "foo" / "extra.txt").write_text("local note\n")

        raw = skill_diff("foo")
    result = json.loads(raw)
    assert result["ok"] is True
    assert result["found"] is True
    assert result["modified"] is True

    by_path = {d["path"]: d for d in result["diffs"]}
    assert by_path["helper.py"]["status"] == "modified"
    assert by_path["extra.txt"]["status"] == "added"


def test_skill_diff_unknown_skill(tmp_path):
    """skill_diff returns found=False for a non-existent skill name."""
    import json

    from tools.skills_tool import skill_diff

    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        raw = skill_diff("does-not-exist")
    result = json.loads(raw)
    assert result["ok"] is False
    assert result["found"] is False


def test_skill_diff_non_bundled_skill(tmp_path):
    """skill_diff returns found=False for non-bundled (custom) skills."""
    import json

    from tools.skills_tool import skill_diff

    bundled, skills_dir, manifest_file = _env(tmp_path)
    with _patches(bundled, skills_dir, manifest_file):
        sync_skills(quiet=True)
        # Non-bundled skills won't be in the bundled manifest
        raw = skill_diff("custom-skill")
    result = json.loads(raw)
    assert result["ok"] is False
    assert result["found"] is False
