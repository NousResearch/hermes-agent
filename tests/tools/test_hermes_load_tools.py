"""Unit tests for tools.hermes_load_tools.load_tools."""

from __future__ import annotations

import json
from typing import Set

import pytest

from tools.hermes_load_tools import load_tools


def _parse(s: str) -> dict:
    return json.loads(s)


def test_load_single_deferred_tool():
    promoted: Set[str] = set()
    out = _parse(load_tools(
        names=["slack_send"],
        promoted=promoted,
        available_names={"slack_send", "core_tool"},
        deferred_names={"slack_send"},
    ))
    assert out["loaded"] == ["slack_send"]
    assert out["already_loaded"] == []
    assert out["already_eager"] == []
    assert out["unknown"] == []
    assert out["total_promoted"] == 1
    assert "Schemas are now available" in out["hint"]
    assert "slack_send" in promoted


def test_load_batched():
    promoted: Set[str] = set()
    out = _parse(load_tools(
        names=["a", "b", "c"],
        promoted=promoted,
        available_names={"a", "b", "c"},
        deferred_names={"a", "b", "c"},
    ))
    assert out["loaded"] == ["a", "b", "c"]
    assert promoted == {"a", "b", "c"}


def test_already_loaded_bucket():
    promoted: Set[str] = {"slack_send"}
    out = _parse(load_tools(
        names=["slack_send", "slack_search"],
        promoted=promoted,
        available_names={"slack_send", "slack_search"},
        deferred_names={"slack_send", "slack_search"},
    ))
    assert out["loaded"] == ["slack_search"]
    assert out["already_loaded"] == ["slack_send"]
    assert promoted == {"slack_send", "slack_search"}


def test_already_eager_bucket():
    """Names not in deferred_names should land in already_eager, not loaded."""
    promoted: Set[str] = set()
    out = _parse(load_tools(
        names=["core_tool"],
        promoted=promoted,
        available_names={"core_tool", "slack_x"},
        deferred_names={"slack_x"},  # core_tool is eager
    ))
    assert out["loaded"] == []
    assert out["already_eager"] == ["core_tool"]
    assert "core_tool" not in promoted, "eager tools shouldn't be promoted"


def test_unknown_name():
    promoted: Set[str] = set()
    out = _parse(load_tools(
        names=["typo_tool"],
        promoted=promoted,
        available_names={"slack_send"},
        deferred_names={"slack_send"},
    ))
    assert out["loaded"] == []
    assert out["unknown"] == ["typo_tool"]
    assert "None of the requested names" in out["hint"]


def test_mixed_buckets():
    promoted: Set[str] = {"slack_send"}
    out = _parse(load_tools(
        names=["slack_send", "slack_search", "core_x", "typo"],
        promoted=promoted,
        available_names={"slack_send", "slack_search", "core_x"},
        deferred_names={"slack_send", "slack_search"},
    ))
    assert out["loaded"] == ["slack_search"]
    assert out["already_loaded"] == ["slack_send"]
    assert out["already_eager"] == ["core_x"]
    assert out["unknown"] == ["typo"]


def test_empty_names_no_op():
    promoted: Set[str] = set()
    out = _parse(load_tools(
        names=[],
        promoted=promoted,
        available_names={"a"},
        deferred_names={"a"},
    ))
    assert out["loaded"] == []
    assert out["total_promoted"] == 0
    assert promoted == set()


def test_whitespace_and_empty_strings_ignored():
    promoted: Set[str] = set()
    out = _parse(load_tools(
        names=["", "  ", "  slack_send  ", None],  # type: ignore[list-item]
        promoted=promoted,
        available_names={"slack_send"},
        deferred_names={"slack_send"},
    ))
    assert out["loaded"] == ["slack_send"]


def test_deferred_names_none_means_classify_all_as_known():
    """When deferred_names is None (tool_search off), names just classify by availability."""
    promoted: Set[str] = set()
    out = _parse(load_tools(
        names=["slack_send", "typo"],
        promoted=promoted,
        available_names={"slack_send"},
        deferred_names=None,
    ))
    # Without deferred_names, every available name lands in "loaded".
    assert out["loaded"] == ["slack_send"]
    assert out["unknown"] == ["typo"]


def test_result_is_valid_json():
    promoted: Set[str] = set()
    raw = load_tools(
        names=["a"],
        promoted=promoted,
        available_names={"a"},
        deferred_names={"a"},
    )
    out = json.loads(raw)
    assert "loaded" in out


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


def test_module_registers_tool():
    """Importing the module should register the tool with the global registry."""
    import tools.hermes_load_tools  # noqa: F401  side-effect import
    from tools.registry import registry

    defs = registry.get_definitions({"hermes_load_tools"})
    assert len(defs) == 1, "hermes_load_tools should be registered"
    schema = defs[0]["function"]
    assert schema["name"] == "hermes_load_tools"
    assert "names" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["names"]


def test_safety_net_handler_returns_error():
    """If the agent-loop interception fails to fire, the registry handler must
    return a structured error rather than silently mutating nothing."""
    import tools.hermes_load_tools  # noqa: F401
    from tools.registry import registry

    result = registry.dispatch("hermes_load_tools", {"names": ["x"]})
    out = json.loads(result)
    assert "error" in out
    assert "agent loop" in out["error"]
