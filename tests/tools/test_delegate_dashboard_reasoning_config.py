"""Delegation child reasoning resolution uses the parent's active config."""
from types import SimpleNamespace

import pytest

from tools.delegate_tool_config import _resolve_child_runtime


@pytest.mark.parametrize(("override", "expected"), [("", "xhigh"), ("low", "low")])
def test_child_runtime_uses_parent_reasoning_or_explicit_override(override, expected):
    parent = SimpleNamespace(
        model="parent-model",
        provider="openai",
        base_url=None,
        api_mode="chat_completions",
        acp_args=[],
        reasoning_config={"enabled": True, "effort": "xhigh"},
    )

    result = _resolve_child_runtime(
        parent,
        {"reasoning_effort": override},
        "parent-api-key",
        model=None,
        override_provider=None,
        override_base_url=None,
        override_api_key=None,
        override_api_mode=None,
        override_acp_command=None,
        override_acp_args=None,
    )

    assert result["reasoning_config"]["effort"] == expected
