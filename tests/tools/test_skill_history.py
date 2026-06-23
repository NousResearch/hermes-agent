import json
from pathlib import Path


VALID_SKILL_CONTENT = """---
name: my-skill
description: test skill
---

# My Skill

Original body.
"""


def _configure_skill_dirs(tmp_path, monkeypatch):
    from agent import skill_utils
    from tools import skill_history, skill_manager_tool

    home = tmp_path / "home"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)

    monkeypatch.setattr(skill_history, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [skills_dir])
    return skills_dir


def test_background_review_patch_snapshots_and_rolls_back(tmp_path, monkeypatch):
    from tools.skill_history import list_skill_history, rollback_skill
    from tools.skill_manager_tool import skill_manage
    from tools.skill_provenance import (
        BACKGROUND_REVIEW,
        reset_current_write_origin,
        set_current_write_origin,
    )

    skills_dir = _configure_skill_dirs(tmp_path, monkeypatch)

    created = json.loads(
        skill_manage(action="create", name="my-skill", content=VALID_SKILL_CONTENT)
    )
    assert created["success"] is True

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        patched = json.loads(
            skill_manage(
                action="patch",
                name="my-skill",
                old_string="Original body.",
                new_string="Updated body.",
            )
        )
    finally:
        reset_current_write_origin(token)

    assert patched["success"] is True
    snapshot_id = patched["history_snapshot"]
    skill_md = skills_dir / "my-skill" / "SKILL.md"
    assert "Updated body." in skill_md.read_text(encoding="utf-8")

    rows = list_skill_history("my-skill")
    assert rows[0]["id"] == snapshot_id
    assert rows[0]["action"] == "patch"
    assert (Path(rows[0]["path"]) / "files" / "SKILL.md").exists()

    ok, msg, restored = rollback_skill("my-skill", snapshot_id=snapshot_id)

    assert ok is True, msg
    assert restored == skills_dir / "my-skill"
    assert "Original body." in skill_md.read_text(encoding="utf-8")


def test_foreground_patch_does_not_create_autonomous_history(tmp_path, monkeypatch):
    from tools.skill_history import list_skill_history
    from tools.skill_manager_tool import skill_manage

    _configure_skill_dirs(tmp_path, monkeypatch)

    created = json.loads(
        skill_manage(action="create", name="my-skill", content=VALID_SKILL_CONTENT)
    )
    patched = json.loads(
        skill_manage(
            action="patch",
            name="my-skill",
            old_string="Original body.",
            new_string="Updated body.",
        )
    )

    assert created["success"] is True
    assert patched["success"] is True
    assert "history_snapshot" not in patched
    assert list_skill_history("my-skill") == []


def test_invalid_skill_name_does_not_write_history_outside_root(tmp_path, monkeypatch):
    from tools.skill_history import snapshot_autonomous_edit

    _configure_skill_dirs(tmp_path, monkeypatch)

    snapshot_id = snapshot_autonomous_edit(
        name="../../escape",
        action="patch",
        skill_dir=None,
    )

    assert snapshot_id is None
    assert not (tmp_path / "home" / "escape").exists()


def test_known_root_check_normalizes_dotdot_paths(tmp_path, monkeypatch):
    from tools.skill_history import _is_under_known_skills_root

    skills_dir = _configure_skill_dirs(tmp_path, monkeypatch)

    assert _is_under_known_skills_root(skills_dir / "my-skill")
    assert not _is_under_known_skills_root(skills_dir / ".." / "outside")
