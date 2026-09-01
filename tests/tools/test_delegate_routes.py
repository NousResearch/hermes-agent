"""Regression coverage for operator-defined delegate_task model routes."""

from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    _build_dynamic_schema_overrides,
    _resolve_delegation_route_config,
    _resolve_route_limit,
)


def _route_config():
    return {
        "model": "",
        "provider": "",
        "reasoning_effort": "",
        "max_iterations": 250,
        "child_timeout_seconds": 1800,
        "default_route": "inherit",
        "routes": {
            "scout": {
                "provider": "cliproxyapi",
                "model": "gemini-3.7-flash-high",
                "reasoning_effort": "medium",
                "max_iterations": 40,
                "child_timeout_seconds": 600,
            },
            "reviewer": {
                "provider": "cliproxyapi",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "max_iterations": 100,
                "child_timeout_seconds": 1200,
            },
        },
    }


def test_dynamic_schema_advertises_only_configured_routes(monkeypatch):
    monkeypatch.setattr("tools.delegate_tool._load_config", _route_config)

    schema = _build_dynamic_schema_overrides()["parameters"]
    task_props = schema["properties"]["tasks"]["items"]["properties"]

    assert task_props["route"]["enum"] == ["inherit", "reviewer", "scout"]
    assert "route" not in schema["properties"]
    assert "model" not in task_props
    assert "provider" not in task_props
    assert "output_schema" not in task_props


def test_named_route_is_self_contained_and_carries_operator_budgets():
    routed, route_name = _resolve_delegation_route_config(_route_config(), "scout")

    assert route_name == "scout"
    assert routed["provider"] == "cliproxyapi"
    assert routed["model"] == "gemini-3.7-flash-high"
    assert routed["reasoning_effort"] == "medium"
    assert routed["max_iterations"] == 40
    assert routed["child_timeout_seconds"] == 600


def test_inherit_route_clears_global_model_pin_and_keeps_parent_reasoning():
    cfg = _route_config()
    cfg.update({"provider": "openrouter", "model": "cheap", "reasoning_effort": "low"})

    routed, route_name = _resolve_delegation_route_config(cfg, "inherit")

    assert route_name == "inherit"
    assert routed["provider"] == ""
    assert routed["model"] == ""
    assert routed["reasoning_effort"] == ""
    assert _resolve_route_limit(routed, "max_iterations", 250, minimum=1) == 250


def test_unknown_route_fails_with_available_names():
    with pytest.raises(ValueError, match="reviewer.*scout"):
        _resolve_delegation_route_config(_route_config(), "expensive")


def test_mixed_routes_reach_child_construction_with_per_route_budgets(monkeypatch):
    import json
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "parent"
    parent.model = "parent-model"
    parent.provider = "cliproxyapi"
    parent.base_url = "http://127.0.0.1:8317/v1"
    parent.api_key = "key"
    parent._active_children = []
    parent._active_children_lock = None

    built = []

    def fake_resolve(cfg, _parent):
        return {
            "model": cfg.get("model") or None,
            "provider": cfg.get("provider") or None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
            "max_output_tokens": None,
            "command": None,
            "args": None,
        }

    def fake_build(**kwargs):
        built.append(kwargs)
        child = MagicMock()
        child._delegate_role = "leaf"
        return child

    monkeypatch.setattr(dt, "_load_config", _route_config)
    monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 2)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", fake_resolve)
    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_build)
    monkeypatch.setattr(
        dt,
        "_run_single_child",
        lambda task_index, goal, child, parent_agent: {
            "task_index": task_index,
            "status": "completed",
            "summary": "ok",
            "api_calls": 1,
            "duration_seconds": 0.1,
        },
    )

    result = json.loads(
        dt.delegate_task(
            tasks=[
                {"goal": "Research independent sources", "route": "scout"},
                {"goal": "Review the stable candidate", "route": "reviewer"},
            ],
            parent_agent=parent,
        )
    )

    assert len(result["results"]) == 2
    assert [entry["model"] for entry in built] == [
        "gemini-3.7-flash-high",
        "gpt-5.6-sol",
    ]
    assert [entry["max_iterations"] for entry in built] == [40, 100]
    assert [entry["override_reasoning_effort"] for entry in built] == [
        "medium",
        "high",
    ]


def test_per_child_timeout_override_beats_global_config():
    child = MagicMock()
    child._delegate_timeout_seconds = 600

    assert _resolve_route_limit(
        {"child_timeout_seconds": child._delegate_timeout_seconds},
        "child_timeout_seconds",
        1800,
        minimum=30,
        zero_disables=True,
    ) == 600
