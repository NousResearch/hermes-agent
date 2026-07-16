"""Tests for the skill_manage tool schema shape.

Guards the description text that spells out per-action required params
for `skill_manage` — the load-bearing fix for description-driven models
(e.g. Grok/DeepSeek) that omit required params when the schema only lists
`action`/`name` in `required[]` without stating what each action itself
needs. Mirrors tests/cron/test_cronjob_schema.py for the analogous
cronjob_tools.py fix.
"""

import pytest

from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA

# Ground truth for what each action actually requires beyond the globally
# required `action`/`name`. Kept here (not duplicated across tests) so both
# the "does the action summary say this" and "does the field's own
# description agree" checks share one source of intent.
REQUIRED_EXTRA_FIELDS_BY_ACTION = {
    "create": ["content"],
    "patch": ["old_string", "new_string"],
    "edit": ["content"],
    "delete": [],
    "write_file": ["file_path", "file_content"],
    "remove_file": ["file_path"],
}

ACTION_SPECIFIC_FIELDS = {"content", "old_string", "new_string", "file_path", "file_content"}


def _action_description() -> str:
    return SKILL_MANAGE_SCHEMA["parameters"]["properties"]["action"]["description"]


def test_skill_manage_schema_action_description_has_required_params_header():
    """`action` description must lead with the per-action requirements callout."""
    assert "Required params per action" in _action_description()


def test_skill_manage_schema_action_enum_matches_documented_actions():
    """Every enum value must be documented — catches a new action added to
    `enum` without updating the description (drift the previous version of
    this test, which hardcoded the action list instead of reading it from
    the schema, would not have caught)."""
    enum_actions = SKILL_MANAGE_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert set(enum_actions) == set(REQUIRED_EXTRA_FIELDS_BY_ACTION)
    action_desc = _action_description()
    for action in enum_actions:
        assert action in action_desc


def _clause_for(action: str) -> str:
    """Slice out the description text covering just this action's clause.

    Actions are documented in enum order, each as "<action> (...)"; slicing
    between one action's name and the next (rather than paren-matching) is
    robust to clauses containing their own parens, e.g. edit's "read skill
    first with skill_view()".
    """
    desc = _action_description()
    actions = list(REQUIRED_EXTRA_FIELDS_BY_ACTION)
    start = desc.index(f"{action} (")
    idx = actions.index(action)
    if idx + 1 < len(actions):
        end = desc.index(f"{actions[idx + 1]} (", start)
    else:
        end = len(desc)
    return desc[start:end]


@pytest.mark.parametrize("action,extra_fields", REQUIRED_EXTRA_FIELDS_BY_ACTION.items())
def test_skill_manage_schema_per_action_requirements(action, extra_fields):
    """Each action's clause must open with "requires: name" and mention
    every extra field it needs somewhere in its clause (not necessarily
    contiguous — e.g. write_file's clause interleaves an inline example
    between file_path and file_content)."""
    clause = _clause_for(action)
    assert clause.startswith(f"{action} (requires: name")
    for field in extra_fields:
        assert field in clause, f"{action}'s clause doesn't mention required field {field!r}: {clause!r}"


@pytest.mark.parametrize(
    "action,extra_fields",
    [(a, f) for a, f in REQUIRED_EXTRA_FIELDS_BY_ACTION.items() if f],
)
def test_skill_manage_schema_field_descriptions_agree_with_action_summary(action, extra_fields):
    """The `action` description and each individual field's own description
    are two independent sources of the same requirement — nothing enforces
    they stay in sync. A field description could drop its "required for X"
    callout (or the action summary could drift) without either test above
    noticing, since each only checks one side. This checks both agree: for
    every field the action summary claims `action` requires, that field's
    own description must also name `action` as requiring it.
    """
    properties = SKILL_MANAGE_SCHEMA["parameters"]["properties"]
    for field in extra_fields:
        field_desc = properties[field]["description"]
        assert f"'{action}'" in field_desc, (
            f"action description says {action!r} requires {field!r}, "
            f"but {field!r}'s own description doesn't mention {action!r}"
        )


def test_skill_manage_schema_required_array_stays_minimal():
    """`required[]` stays minimal — only the fields required by every
    action, i.e. `action` and `name`.

    The schema intentionally does NOT promote action-specific fields into
    the top-level required array because they're only mandatory for
    specific actions, not universally — the description text carries the
    conditional requirement instead, same pattern as CRONJOB_SCHEMA. This
    guards against the naive "fix" of a model-omission bug by just adding
    a field to `required[]`, which would break every action that doesn't
    need that field.
    """
    required = set(SKILL_MANAGE_SCHEMA["parameters"]["required"])
    assert required == {"action", "name"}
    assert not required & ACTION_SPECIFIC_FIELDS
