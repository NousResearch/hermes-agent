"""Trusted, code-only mode routing into delegation."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.mode_router import UnknownModeError
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_system_prompt,
    route_trusted_mode,
)


def _parent(*, enabled=True):
    return SimpleNamespace(_mode_router_enabled=enabled)


def test_thinking_expansion_stays_with_parent_and_never_delegates():
    with patch("tools.delegate_tool.delegate_task") as delegate:
        decision = route_trusted_mode(
            mode="thinking-expansion",
            goal="Compare release approaches",
            context="Prefer reversible options",
            parent_agent=_parent(),
        )

    assert decision.route == "direct-parent"
    assert decision.mode == "thinking-expansion"
    assert decision.goal == "Compare release approaches"
    delegate.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "expected_hints"),
    [
        ("research-analysis", ("web", "browser")),
        ("execution-development", ("terminal", "file")),
    ],
)
def test_child_modes_dispatch_synchronously_with_internal_policy_hints(mode, expected_hints):
    with patch("tools.delegate_tool.delegate_task", return_value='{"results": []}') as delegate:
        decision = route_trusted_mode(
            mode=mode,
            goal="Do the task",
            context="Known constraints",
            parent_agent=_parent(),
        )

    assert decision.route == "child"
    assert decision.result == '{"results": []}'
    delegate.assert_called_once_with(
        goal="Do the task",
        context="Known constraints",
        background=False,
        parent_agent=delegate.call_args.kwargs["parent_agent"],
        _trusted_toolsets=expected_hints,
        _trusted_mode=mode,
    )


def test_unknown_mode_fails_closed_before_delegation():
    with patch("tools.delegate_tool.delegate_task") as delegate:
        with pytest.raises(UnknownModeError):
            route_trusted_mode(
                mode="untrusted-custom-mode",
                goal="Do something",
                parent_agent=_parent(),
            )
    delegate.assert_not_called()


def test_disabled_router_preserves_no_routing_behavior():
    with patch("tools.delegate_tool.delegate_task") as delegate:
        decision = route_trusted_mode(
            mode="research-analysis",
            goal="Research",
            parent_agent=_parent(enabled=False),
        )
    assert decision.route == "direct-parent"
    assert decision.reason == "mode-router-disabled"
    delegate.assert_not_called()


def test_mode_contract_is_appended_without_replacing_existing_child_rules():
    prompt = _build_child_system_prompt(
        "Implement it",
        "Constraints",
        workspace_path="/tmp/project",
        mode="execution-development",
    )
    assert "You are a focused subagent" in prompt
    assert "WORKSPACE PATH" in prompt
    assert "Important workspace rule" in prompt
    assert "Mode contract: execution-development" in prompt
    assert "inspect -> implement -> test -> deliver" in prompt
    assert "Verify material claims and validate outputs" in prompt


def test_trusted_seam_does_not_change_model_facing_schema():
    before = deepcopy(DELEGATE_TASK_SCHEMA)
    route_trusted_mode(
        mode="thinking-expansion",
        goal="Think",
        parent_agent=_parent(),
    )
    assert DELEGATE_TASK_SCHEMA == before
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "mode" not in props
    assert "toolsets" not in props
