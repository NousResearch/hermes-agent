"""Regression guard: the tool_call bridge must validate AND dispatch the same shape.

The validator repaired ``{"item": ...}`` response-item envelopes on a COPY
and returned only a pass/fail signal, so ``agent/tool_executor._unwrap_tool_search_call``
handed the handler the ORIGINAL args: a call that passed validation failed
again in the tool with ``todos must be a list, got dict``. The contract pinned
here is the UNWRAP's — validate returns None AND the dispatched args match
the repaired shape. Asserting on ``repair_deferred_call_args`` alone would not
catch that: repair was always correct, the bridge just discarded it.

Upstream issue #99270 / PR #115020.
"""

from __future__ import annotations

import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_TODO_ITEM = {"id": "t1", "content": "x", "status": "in_progress"}

# Shapes a frontier model actually emits for a one-item array argument: the
# envelope wraps the array, wraps a single element, or nests.
_ENVELOPE_SHAPES = {
    "plain": {"todos": [_TODO_ITEM]},
    "envelope_on_element": {"todos": [{"item": _TODO_ITEM}]},
    "envelope_on_array": {"todos": {"item": [_TODO_ITEM]}},
    "nested_envelope": {"todos": {"item": {"item": [_TODO_ITEM]}}},
    "envelope_on_single_element": {"todos": {"item": _TODO_ITEM}},
    "double_list": {"todos": [[_TODO_ITEM]]},
    "with_sibling_arg": {"todos": [_TODO_ITEM], "merge": True},
}

_MALFORMED = {
    "missing_id": {"todos": [{"content": "x", "status": "in_progress"}]},
    "bad_status": {"todos": [{"id": "t1", "content": "x", "status": "nope"}]},
    "not_an_array": {"todos": "basura"},
}


@pytest.fixture
def todo_tool():
    """Register a real array-of-object tool so the registry schema drives both paths."""
    from tools.registry import discover_builtin_tools, registry
    from tools.tool_search_validation import (
        repair_deferred_call_args, validate_deferred_call_args)

    discover_builtin_tools()
    name = "arr_tool_probe"
    registry.register(
        name=name,
        handler=lambda args, **kw: "{}",
        toolset="tool-search-probe",
        schema={
            "name": name,
            "description": "Deferred array-argument probe",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {"type": "string",
                                       "enum": ["pending", "in_progress", "completed", "cancelled"]},
                        },
                        "required": ["id", "content", "status"],
                    }},
                    "merge": {"type": "boolean", "default": False},
                },
            },
        },
    )
    return name, validate_deferred_call_args, repair_deferred_call_args


@pytest.mark.parametrize("label", sorted(_ENVELOPE_SHAPES))
def test_unwrap_dispatches_repaired_args(todo_tool, label):
    """The bridge must hand the handler the shape it validated, not the raw envelope.

    This is the regression guard: the bug was the unwrap returning
    ``underlying_args`` while validation had passed on a repaired copy.
    """
    name, validate, _ = todo_tool
    from agent.tool_executor import _unwrap_tool_search_call

    class _Agent:
        enabled_toolsets = None
        disabled_toolsets = None
        _tool_search_scope_cache = None

    bridge_args = {"name": name, "arguments": _ENVELOPE_SHAPES[label]}
    unwrapped, dispatched, scope_block = _unwrap_tool_search_call(
        _Agent(), "tool_call", bridge_args)

    assert scope_block is None, f"{label} was blocked: {scope_block}"
    assert unwrapped == name
    assert validate(name, dispatched) is None, (
        f"{label}: dispatched args must satisfy the schema, got {dispatched}")
    assert isinstance(dispatched["todos"], list), (
        f"{label}: schema declares type: array, bridge handed "
        f"{type(dispatched['todos']).__name__}")


@pytest.mark.parametrize("label", sorted(_ENVELOPE_SHAPES))
def test_repair_is_schema_consistent(todo_tool, label):
    """Whatever validation accepts, repair alone also accepts (idempotent contract)."""
    name, validate, repair = todo_tool
    args = _ENVELOPE_SHAPES[label]

    assert validate(name, args) is None, f"{label} should pass validation"

    repaired = repair(name, args)
    assert validate(name, repaired) is None, (
        f"{label}: repaired args must also validate — {repaired}")
    assert isinstance(repaired["todos"], list)


@pytest.mark.parametrize("label", sorted(_MALFORMED))
def test_malformed_still_rejected(todo_tool, label):
    """Repair must not become a bypass: malformed payloads keep failing."""
    name, validate, _ = todo_tool

    assert validate(name, _MALFORMED[label]) is not None


def test_repair_is_idempotent(todo_tool):
    """Repairing an already-repaired payload is a no-op, so a re-validated call is stable."""
    name, _, repair = todo_tool

    once = repair(name, {"todos": {"item": _TODO_ITEM}})
    twice = repair(name, once)

    assert once == twice
