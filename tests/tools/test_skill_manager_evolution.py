import json
from contextlib import contextmanager
from unittest.mock import patch

from agent.evolution_log import read_events
from tools.skill_manager_tool import _create_skill, skill_manage


VALID_SKILL_CONTENT = """---
name: test-skill
description: A test skill for evolution logging.
---

# Test Skill

Follow the tested process.
"""


@contextmanager
def _isolated_skills(skills_dir):
    with (
        patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir),
        patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]),
    ):
        yield


def _enable_evolution(home):
    (home / "config.yaml").write_text(
        "evolution:\n"
        "  enabled: true\n"
        "  record_diff: true\n"
        "  redact: true\n"
        "  max_diff_chars: 20000\n",
        encoding="utf-8",
    )


def test_skill_create_records_evolution_event_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    skills_dir = tmp_path / "skills"

    with _isolated_skills(skills_dir):
        result = json.loads(
            skill_manage(
                action="create",
                name="test-skill",
                content=VALID_SKILL_CONTENT,
                summary="Created test skill",
                reason="A reusable tested process was discovered.",
            )
        )

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "skill.create"
    assert event["source_tool"] == "skill_manage"
    assert event["target"] == "skills/test-skill/SKILL.md"
    assert event["target_kind"] == "skill"
    assert event["target_name"] == "test-skill"
    assert event["summary"] == "Created test skill"
    assert event["reason"] == "A reusable tested process was discovered."
    assert "--- before" in event["diff"]
    assert "+++ after" in event["diff"]
    assert "+# Test Skill" in event["diff"]


UPDATED_SKILL_CONTENT = """---
name: test-skill
description: Updated evolution logging skill.
---

# Test Skill

Follow the improved process.
"""


def _seed_skill():
    result = _create_skill("test-skill", VALID_SKILL_CONTENT)
    assert result["success"] is True


def test_skill_patch_and_edit_record_evolution_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    skills_dir = tmp_path / "skills"

    with _isolated_skills(skills_dir):
        _seed_skill()
        patch_result = json.loads(
            skill_manage(
                action="patch",
                name="test-skill",
                old_string="Follow the tested process.",
                new_string="Follow the improved process.",
                summary="Patched test skill",
            )
        )
        edit_result = json.loads(
            skill_manage(
                action="edit",
                name="test-skill",
                content=UPDATED_SKILL_CONTENT,
                summary="Edited test skill",
            )
        )

    assert patch_result["success"] is True
    assert edit_result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert [event["type"] for event in events] == ["skill.patch", "skill.edit"]
    assert events[0]["target"] == "skills/test-skill/SKILL.md"
    assert "-Follow the tested process." in events[0]["diff"]
    assert "+Follow the improved process." in events[0]["diff"]
    assert events[1]["target"] == "skills/test-skill/SKILL.md"
    assert "Updated evolution logging skill" in events[1]["diff"]


def test_skill_delete_records_placeholder_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    skills_dir = tmp_path / "skills"

    with _isolated_skills(skills_dir):
        _seed_skill()
        result = json.loads(
            skill_manage(
                action="delete",
                name="test-skill",
                absorbed_into="",
                summary="Deleted test skill",
            )
        )

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert len(events) == 1
    assert events[0]["type"] == "skill.delete"
    assert events[0]["target"] == "skills/test-skill/SKILL.md"
    assert events[0]["diff"] == "[skill deleted: content omitted]"


def test_skill_write_file_and_remove_file_record_evolution_events(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    skills_dir = tmp_path / "skills"

    with _isolated_skills(skills_dir):
        _seed_skill()
        write_result = json.loads(
            skill_manage(
                action="write_file",
                name="test-skill",
                file_path="references/api.md",
                file_content="Use the stable API.",
                summary="Wrote skill reference",
            )
        )
        remove_result = json.loads(
            skill_manage(
                action="remove_file",
                name="test-skill",
                file_path="references/api.md",
                summary="Removed skill reference",
            )
        )

    assert write_result["success"] is True
    assert remove_result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert [event["type"] for event in events] == [
        "skill.write_file",
        "skill.remove_file",
    ]
    assert events[0]["target"] == "skills/test-skill/references/api.md"
    assert "+Use the stable API." in events[0]["diff"]
    assert events[1]["target"] == "skills/test-skill/references/api.md"
    assert "-Use the stable API." in events[1]["diff"]


def test_failed_skill_mutation_records_no_evolution_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    skills_dir = tmp_path / "skills"

    with _isolated_skills(skills_dir):
        result = json.loads(
            skill_manage(
                action="patch",
                name="missing-skill",
                old_string="old",
                new_string="new",
            )
        )

    assert result["success"] is False
    events, warnings = read_events()
    assert warnings == []
    assert events == []
