"""Tests for skill_manage integration with the skill change ledger."""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from tools.skill_change_ledger import get_skill_change, list_skill_changes
from tools.skill_manager_tool import skill_manage


VALID_SKILL_CONTENT = """\
---
name: ledger-skill
description: A test skill for ledger integration.
---

# Ledger Skill

Step 1: Do the thing.
"""

VALID_SKILL_CONTENT_2 = """\
---
name: ledger-skill
description: Updated description.
---

# Ledger Skill v2

Step 1: Do the new thing.
"""


@contextmanager
def isolated_skill_environment(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    with patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]):
        yield skills_dir, hermes_home


def test_successful_create_records_skill_change_with_explicit_reason(tmp_path, monkeypatch):
    with isolated_skill_environment(tmp_path, monkeypatch):
        raw = skill_manage(
            action="create",
            name="ledger-skill",
            content=VALID_SKILL_CONTENT,
            reason="User requested a reusable ledger test skill.",
        )

    result = json.loads(raw)
    assert result["success"] is True
    assert result["skill_change_event_id"]

    events = list_skill_changes(skill="ledger-skill")
    assert len(events) == 1
    event = events[0]
    assert event["event_id"] == result["skill_change_event_id"]
    assert event["action"] == "create"
    assert event["source"] == "skill_manage"
    assert event["reason"] == "User requested a reusable ledger test skill."
    assert event["reason_kind"] == "explicit"
    assert event["before_hash"] is None
    assert event["after_hash"].startswith("sha256:")
    assert "SKILL.md" in event["changed_files"]

    detail = get_skill_change(event["event_id"])
    assert detail is not None
    assert "+# Ledger Skill" in detail["diff_text"]


def test_successful_patch_records_diff_and_missing_reason_as_unattributed(tmp_path, monkeypatch):
    with isolated_skill_environment(tmp_path, monkeypatch):
        json.loads(skill_manage(action="create", name="ledger-skill", content=VALID_SKILL_CONTENT))
        raw = skill_manage(
            action="patch",
            name="ledger-skill",
            old_string="Do the thing.",
            new_string="Do the observed thing.",
        )

    result = json.loads(raw)
    assert result["success"] is True

    events = list_skill_changes(skill="ledger-skill")
    patch_event = events[0]
    assert patch_event["event_id"] == result["skill_change_event_id"]
    assert patch_event["action"] == "patch"
    assert patch_event["reason"] is None
    assert patch_event["reason_kind"] == "unattributed"
    assert patch_event["before_hash"] != patch_event["after_hash"]
    assert patch_event["changed_files"] == ["SKILL.md"]

    detail = get_skill_change(patch_event["event_id"])
    assert detail is not None
    assert "-Step 1: Do the thing." in detail["diff_text"]
    assert "+Step 1: Do the observed thing." in detail["diff_text"]


def test_failed_patch_does_not_record_skill_change(tmp_path, monkeypatch):
    with isolated_skill_environment(tmp_path, monkeypatch):
        json.loads(skill_manage(action="create", name="ledger-skill", content=VALID_SKILL_CONTENT))
        before = len(list_skill_changes(skill="ledger-skill"))
        raw = skill_manage(
            action="patch",
            name="ledger-skill",
            old_string="missing text",
            new_string="replacement",
            reason="This should not be recorded because the patch fails.",
        )

    result = json.loads(raw)
    assert result["success"] is False
    assert len(list_skill_changes(skill="ledger-skill")) == before


def test_delete_records_before_hash_and_delete_diff(tmp_path, monkeypatch):
    with isolated_skill_environment(tmp_path, monkeypatch):
        json.loads(skill_manage(action="create", name="ledger-skill", content=VALID_SKILL_CONTENT))
        raw = skill_manage(
            action="delete",
            name="ledger-skill",
            reason="Remove obsolete skill after review.",
        )

    result = json.loads(raw)
    assert result["success"] is True

    event = list_skill_changes(skill="ledger-skill")[0]
    assert event["event_id"] == result["skill_change_event_id"]
    assert event["action"] == "delete"
    assert event["before_hash"].startswith("sha256:")
    assert event["after_hash"] is None
    assert event["changed_files"] == ["SKILL.md"]

    detail = get_skill_change(event["event_id"])
    assert detail is not None
    assert "-# Ledger Skill" in detail["diff_text"]
