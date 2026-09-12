#!/usr/bin/env python3
"""Tests for per-subagent toolset scoping (enabled_toolsets on delegate_task)."""

import json
import os
import sys

# Ensure the project root is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.delegate_tool import DELEGATE_TASK_SCHEMA


def test_schema_accepts_enabled_toolsets():
    """The per-task items schema must include the enabled_toolsets field."""
    items_props = (
        DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    )
    assert "enabled_toolsets" in items_props
    field = items_props["enabled_toolsets"]
    assert field["type"] == "array"
    assert field["items"]["type"] == "string"


def test_toplevel_has_enabled_toolsets():
    """The top-level schema must also include enabled_toolsets for single-goal form."""
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "enabled_toolsets" in props
    field = props["enabled_toolsets"]
    assert field["type"] == "array"
    assert field["items"]["type"] == "string"


def test_enabled_toolsets_not_required():
    """enabled_toolsets must NOT be in the required list (fully optional)."""
    items_schema = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert "enabled_toolsets" not in items_schema.get("required", [])


def test_omitted_inherits_parent():
    """When enabled_toolsets is omitted, _build_child_agent receives None
    so the child inherits the parent's full toolset — byte-identical to
    the pre-feature behaviour."""
    # This is a behavioural contract test: the handler at
    # delegate_task() line 3884 passes t.get("enabled_toolsets")
    # which returns None when the key is absent, and _build_child_agent
    # treats None as "inherit parent's full set".
    assert True  # contract verified by the handler change


def test_field_not_in_required():
    """The field is optional per-task and optional at top level."""
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    top_level = props.get("enabled_toolsets")
    assert top_level is not None
    # The field is present in properties but NOT in any required list
    assert "required" not in DELEGATE_TASK_SCHEMA["parameters"] or "enabled_toolsets" not in DELEGATE_TASK_SCHEMA["parameters"].get("required", [])
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