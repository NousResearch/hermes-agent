#!/usr/bin/env python3
"""Tests for per-subagent toolset scoping (tasks[].enabled_toolsets on delegate_task).

Metadata-only pivot per #56386: no model-facing top-level `toolsets` argument;
scoping rides inside the per-task dict and is resolved in the spawn loop.
"""

import json

from tools.delegate_tool import DELEGATE_TASK_SCHEMA


def _items_props():
    return (
        DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    )


def test_schema_accepts_enabled_toolsets():
    """The per-task items schema must include the enabled_toolsets field."""
    items_props = _items_props()
    assert "enabled_toolsets" in items_props
    field = items_props["enabled_toolsets"]
    assert field["type"] == "array"
    assert field["items"]["type"] == "string"


def test_toplevel_has_no_enabled_toolsets():
    """The top-level schema must NOT expose enabled_toolsets (metadata-only pivot)."""
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "enabled_toolsets" not in props


def test_enabled_toolsets_not_required():
    """enabled_toolsets must NOT be in the required list (fully optional)."""
    items_schema = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert "enabled_toolsets" not in items_schema.get("required", [])


def test_schema_validates_structure():
    """The DELEGATE_TASK_SCHEMA must be valid JSON Schema (no syntax errors)."""
    # Serialise/deserialise round-trip to catch structural issues
    rt = json.loads(json.dumps(DELEGATE_TASK_SCHEMA))
    items_props = (
        rt["parameters"]["properties"]["tasks"]["items"]["properties"]
    )
    assert "enabled_toolsets" in items_props
    assert items_props["enabled_toolsets"]["type"] == "array"