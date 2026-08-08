from __future__ import annotations

import json

from model_tools import get_all_tool_names
from toolsets import resolve_toolset
from tools.finish_tool import finish_tool
from tools.registry import registry


def test_finish_tool_is_registered_and_in_core_toolset() -> None:
    assert "finish" in get_all_tool_names()
    assert "finish" in resolve_toolset("finish")
    assert "finish" in resolve_toolset("hermes-cli")


def test_finish_tool_returns_structured_completion_signal() -> None:
    result = json.loads(finish_tool("done", "Complete", ["tests passed"]))

    assert result == {
        "success": True,
        "status": "done",
        "summary": "Complete",
        "evidence": ["tests passed"],
    }


def test_finish_tool_rejects_invalid_status() -> None:
    result = json.loads(finish_tool("continue", "Not done", []))

    assert result["success"] is False
    assert "status" in result["error"]


def test_finish_tool_dispatch_normalizes_evidence_to_list() -> None:
    result = json.loads(
        registry.dispatch(
            "finish",
            {"status": "blocked", "summary": "Needs input", "evidence": "missing key"},
        )
    )

    assert result == {
        "success": True,
        "status": "blocked",
        "summary": "Needs input",
        "evidence": ["missing key"],
    }
