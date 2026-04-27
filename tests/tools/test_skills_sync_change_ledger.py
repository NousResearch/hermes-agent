"""Tests for skills_sync integration with the skill change ledger."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from tools.skill_change_ledger import get_skill_change, list_skill_changes
from tools.skills_sync import _dir_hash, sync_skills


def _setup_bundled(tmp_path: Path) -> Path:
    bundled = tmp_path / "bundled_skills"
    (bundled / "github" / "github-code-review").mkdir(parents=True)
    (bundled / "github" / "github-code-review" / "SKILL.md").write_text("# GitHub review\n", encoding="utf-8")
    return bundled


def _patch_sync(bundled: Path, skills_dir: Path, manifest_file: Path):
    stack = ExitStack()
    stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
    stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
    stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
    return stack


def test_fresh_bundled_copy_records_system_change_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    bundled = _setup_bundled(tmp_path)
    skills_dir = tmp_path / "user_skills"
    manifest_file = skills_dir / ".bundled_manifest"

    with _patch_sync(bundled, skills_dir, manifest_file):
        result = sync_skills(quiet=True)

    assert result["copied"] == ["github-code-review"]

    events = list_skill_changes(skill="github-code-review")
    assert len(events) == 1
    event = events[0]
    assert event["action"] == "bundled_copy"
    assert event["actor"] == "hermes-system"
    assert event["source"] == "skills_sync"
    assert event["category"] == "github"
    assert event["reason_kind"] == "system"
    assert event["before_hash"] is None
    assert event["after_hash"].startswith("sha256:")
    assert event["changed_files"] == ["SKILL.md"]

    detail = get_skill_change(event["event_id"])
    assert detail is not None
    assert "+# GitHub review" in detail["diff_text"]


def test_bundled_update_records_diff_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    bundled = _setup_bundled(tmp_path)
    skills_dir = tmp_path / "user_skills"
    manifest_file = skills_dir / ".bundled_manifest"

    user_skill = skills_dir / "github" / "github-code-review"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("# GitHub review v1\n", encoding="utf-8")
    old_origin_hash = _dir_hash(user_skill)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(f"github-code-review:{old_origin_hash}\n", encoding="utf-8")

    with _patch_sync(bundled, skills_dir, manifest_file):
        result = sync_skills(quiet=True)

    assert result["updated"] == ["github-code-review"]

    event = list_skill_changes(skill="github-code-review")[0]
    assert event["action"] == "bundled_update"
    assert event["source"] == "skills_sync"
    assert event["reason_kind"] == "system"
    assert event["before_hash"] != event["after_hash"]
    assert event["changed_files"] == ["SKILL.md"]

    detail = get_skill_change(event["event_id"])
    assert detail is not None
    assert "-# GitHub review v1" in detail["diff_text"]
    assert "+# GitHub review" in detail["diff_text"]
