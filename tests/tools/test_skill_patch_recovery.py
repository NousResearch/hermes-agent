"""Patch misses must offer a usable fresh read, including after batch rollback."""

import json

import pytest

from tools import skill_manager_tool, skills_tool
from tools.registry import registry


@pytest.fixture
def skill_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", root)
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", root)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [root])
    skill = root / "recovery-probe"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: recovery-probe\ndescription: Use when checking patch recovery.\n---\n"
        "# Instructions\n" + "Preserve this unrelated context.\n" * 30
        + "Read the current document before changing it.\n", encoding="utf-8")
    (skill / "references").mkdir()
    (skill / "references" / "guide.md").write_text(
        "Read the current document before changing it.\n", encoding="utf-8")
    return skill


@pytest.mark.parametrize("file_path", [None, "references/guide.md"])
def test_no_match_recovery_returns_fresh_text_and_retries(skill_target, file_path):
    task = "patch-recovery"
    args = {"name": skill_target.name, **({"file_path": file_path} if file_path else {})}
    target = skill_target / (file_path or "SKILL.md")
    before = target.read_bytes()
    # An unchanged view normally deduplicates. A miss must make only this file readable again.
    first = json.loads(registry.dispatch("skill_view", args, task_id=task))
    assert first["success"]
    assert json.loads(registry.dispatch("skill_view", args, task_id=task))["dedup"]
    miss = json.loads(registry.dispatch("skill_manage", {"operations": [{
        "action": "patch", **args, "old_string": "Never found anywhere in this skill",
        "new_string": "Read the revised document before changing it.",
    }]}, task_id=task))
    assert not miss["success"]
    assert target.read_bytes() == before
    recovery = miss["recovery"]
    fresh = json.loads(registry.dispatch(recovery["tool"], recovery["arguments"], task_id=task))
    assert fresh["success"] and not fresh.get("dedup")
    old = "Read the current document before changing it."
    assert old in fresh["content"]
    retry = json.loads(registry.dispatch("skill_manage", {"operations": [{
        "action": "patch", **args, "old_string": old,
        "new_string": "Read the revised document before changing it.",
    }]}, task_id=task))
    assert retry["success"]
    assert target.read_bytes() == before.replace(b"current", b"revised")


def test_batch_recovery_reads_restored_file_without_evicting_other_views(skill_target):
    task = "batch-recovery"
    args = {"name": skill_target.name}
    supporting = {**args, "file_path": "references/guide.md"}
    for view_args in (args, supporting):
        assert json.loads(registry.dispatch("skill_view", view_args, task_id=task))["success"]
    target = skill_target / "SKILL.md"
    before = target.read_bytes()
    miss = json.loads(registry.dispatch("skill_manage", {"operations": [
        {"action": "patch", **args, "old_string": "current", "new_string": "transient"},
        {"action": "patch", **args, "old_string": "Never found anywhere in this skill",
         "new_string": "replacement"},
    ]}, task_id=task))
    assert not miss["success"] and miss["completed_before_failure"] == 1
    assert target.read_bytes() == before
    fresh = json.loads(registry.dispatch(miss["recovery"]["tool"],
                                       miss["recovery"]["arguments"], task_id=task))
    assert not fresh.get("dedup")
    assert "current document" in fresh["content"] and "transient" not in fresh["content"]
    assert json.loads(registry.dispatch("skill_view", supporting, task_id=task))["dedup"]
