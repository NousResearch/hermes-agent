"""Nested progress must preserve the worker identity for ACP and other consumers."""

from types import SimpleNamespace

import pytest

from tools.delegate_tool_progress import _build_child_progress_callback


@pytest.mark.parametrize("event", ["subagent.tool", "subagent.progress"])
def test_nested_progress_preserves_identity_without_counting_parent_tools(event):
    delivered = []
    parent = SimpleNamespace(tool_progress_callback=lambda *args, **kwargs: delivered.append((args, kwargs)))
    relay = _build_child_progress_callback(0, "Parent", parent, subagent_id="orchestrator")
    relay(event, "terminal", "work", {"command": "pwd"}, subagent_id="nested",
          parent_id="orchestrator", tool_count=3)
    assert delivered[0][0] == (event, "terminal", "work", {"command": "pwd"})
    assert delivered[0][1]["subagent_id"] == "nested"
    assert delivered[0][1]["parent_id"] == "orchestrator"
    assert delivered[0][1]["tool_count"] == 3
    assert relay.tool_count == 0
