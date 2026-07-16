"""Tests for the skill_manage tool schema shape.

Guards the description text that spells out per-action required params
for `skill_manage` — the load-bearing fix for description-driven models
(e.g. Grok/DeepSeek) that omit required params when the schema only lists
`action`/`name` in `required[]` without stating what each action itself
needs. Mirrors tests/cron/test_cronjob_schema.py for the analogous
cronjob_tools.py fix.
"""

from __future__ import annotations

import pytest


def test_skill_manage_schema_action_description_flags_required_params():
    """`action` description must state required params for every enum action."""
    from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA

    action_desc = SKILL_MANAGE_SCHEMA["parameters"]["properties"]["action"]["description"]
    assert "Required params per action" in action_desc
    for action in ["create", "patch", "edit", "delete", "write_file", "remove_file"]:
        assert action in action_desc


@pytest.mark.parametrize(
    "action,expected_substring",
    [
        ("create", "requires: name, content"),
        ("patch", "requires: name, old_string, new_string"),
        ("edit", "requires: name, content"),
        ("delete", "requires: name"),
        ("write_file", "requires: name, file_path"),
        ("remove_file", "requires: name, file_path"),
    ],
)
def test_skill_manage_schema_per_action_requirements(action, expected_substring):
    """Each action's exact required-param list must appear in the description."""
    from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA

    action_desc = SKILL_MANAGE_SCHEMA["parameters"]["properties"]["action"]["description"]
    assert f"{action} ({expected_substring}" in action_desc


def test_skill_manage_schema_required_array_stays_minimal():
    """`required[]` stays minimal — `action` and `name` only.

    The schema intentionally does NOT promote action-specific fields (content,
    old_string, new_string, file_path, file_content) into the top-level
    required array because they're only mandatory for specific actions, not
    universally. The description text carries the conditional requirement
    instead, same pattern as CRONJOB_SCHEMA.
    """
    from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA

    assert SKILL_MANAGE_SCHEMA["parameters"]["required"] == ["action", "name"]
