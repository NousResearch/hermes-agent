"""Regression coverage for delegation.max_spawn_depth=0."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from tools.delegate_tool import (
    _build_dynamic_schema_overrides,
    _get_max_spawn_depth,
    delegate_task,
)


def test_zero_max_spawn_depth_disables_delegation():
    with patch("tools.delegate_tool._load_config", return_value={"max_spawn_depth": 0}):
        assert _get_max_spawn_depth() == 0


def test_depth_zero_parent_is_rejected_before_child_is_built():
    parent = SimpleNamespace(_delegate_depth=0)
    with patch("tools.delegate_tool._load_config", return_value={"max_spawn_depth": 0}), \
         patch("tools.delegate_tool._build_child_agent", side_effect=AssertionError("must not spawn")):
        result = json.loads(delegate_task(goal="do not spawn", parent_agent=parent))

    assert result["error"].startswith("Delegation depth limit reached")
    assert "max_spawn_depth=0" in result["error"]


def test_dynamic_schema_explains_when_delegation_is_disabled():
    with patch("tools.delegate_tool._load_config", return_value={"max_spawn_depth": 0}):
        schema = _build_dynamic_schema_overrides()

    assert "Delegation is DISABLED" in schema["description"]
    assert "no child can be spawned" in schema["parameters"]["properties"]["role"]["description"]
