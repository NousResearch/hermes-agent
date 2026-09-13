from __future__ import annotations

from tools.delegate_tool import DEFAULT_MAX_ITERATIONS, DELEGATE_TASK_SCHEMA
from tools.delegate_tool_config import (
    DEFAULT_CHILD_TIMEOUT,
    MAX_DEPTH,
    _get_child_timeout,
    _get_max_concurrent_children,
    _get_max_spawn_depth,
)
from tools.delegate_tool_progress import _build_child_system_prompt
from tools.delegate_tool_tasks import _normalize_task_list


def test_advertised_task_schema_requires_a_closed_purpose():
    item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert "purpose" in item["required"]
    assert item["properties"]["purpose"]["enum"] == [
        "research_evidence", "bounded_implementation",
    ]


def test_legacy_missing_purpose_defaults_to_research_evidence():
    normalized, error = _normalize_task_list(
        None, None, [{"goal": "Review the implementation"}], None, "leaf", 1,
    )
    assert error is None
    assert normalized[0]["purpose"] == "research_evidence"


def test_invalid_purpose_is_rejected_before_child_construction():
    normalized, error = _normalize_task_list(
        None, None, [{"goal": "Review the implementation", "purpose": "unbounded"}], None, "leaf", 1,
    )
    assert normalized is None
    assert "Task 0" in error
    assert "purpose" in error.lower()


def test_bounded_implementation_purpose_is_preserved():
    normalized, error = _normalize_task_list(
        None, None,
        [{"goal": "Implement the bounded fix", "purpose": "bounded_implementation"}],
        None, "leaf", 1,
    )
    assert error is None
    assert normalized[0]["purpose"] == "bounded_implementation"


def test_child_prompt_carries_the_selected_purpose():
    research = _build_child_system_prompt("Review code", purpose="research_evidence")
    implementation = _build_child_system_prompt("Patch code", purpose="bounded_implementation")
    assert "DELEGATION PURPOSE: research_evidence" in research
    assert "DELEGATION PURPOSE: bounded_implementation" in implementation


def test_phase3_runtime_defaults_are_bounded(monkeypatch):
    from tools import delegate_tool_config

    monkeypatch.setattr(delegate_tool_config, "_cfg", lambda: {})
    monkeypatch.delenv("DELEGATION_MAX_CONCURRENT_CHILDREN", raising=False)
    monkeypatch.delenv("DELEGATION_CHILD_TIMEOUT_SECONDS", raising=False)
    assert DEFAULT_CHILD_TIMEOUT == 900.0
    assert DEFAULT_MAX_ITERATIONS == 10
    assert _get_max_concurrent_children() == 1
    assert _get_child_timeout() == 900.0
    assert MAX_DEPTH == 1
    assert _get_max_spawn_depth() == 1
