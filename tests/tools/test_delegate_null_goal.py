"""delegate_task batch goals must tolerate JSON-null / non-string values."""

from __future__ import annotations

import json

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import delegate_task


def test_batch_null_goal_returns_tool_error_not_attribute_error():
    """``{"goal": null}`` must not AttributeError on ``.strip()``."""
    result = json.loads(
        delegate_task(
            tasks=[{"goal": None}],
            parent_agent=_make_mock_parent(),
        )
    )
    assert "error" in result
    assert "missing a 'goal'" in result["error"]


def test_batch_non_string_goal_returns_tool_error():
    result = json.loads(
        delegate_task(
            tasks=[{"goal": 42}],
            parent_agent=_make_mock_parent(),
        )
    )
    assert "error" in result
    assert "missing a 'goal'" in result["error"]


def test_batch_empty_string_goal_returns_tool_error():
    result = json.loads(
        delegate_task(
            tasks=[{"goal": "   "}],
            parent_agent=_make_mock_parent(),
        )
    )
    assert "error" in result
    assert "missing a 'goal'" in result["error"]
