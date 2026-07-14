"""Tests for the explicit multi-select user interaction tool."""

import json
from types import SimpleNamespace

from tools.select_many_tool import (
    MAX_SELECT_MANY_CHOICES,
    SELECT_MANY_SCHEMA,
    select_many_tool,
)


def test_returns_selected_choices_as_json_array():
    seen = {}

    def callback(question, choices):
        seen["question"] = question
        seen["choices"] = choices
        return [choices[0], choices[2]]

    result = json.loads(
        select_many_tool(
            "Which directories should be removed?",
            ["cache", "dist", "node_modules"],
            callback=callback,
        )
    )

    assert seen == {
        "question": "Which directories should be removed?",
        "choices": ["cache", "dist", "node_modules"],
    }
    assert result["selected_choices"] == ["cache", "node_modules"]
    assert result["cancelled"] is False


def test_empty_selection_is_explicit_cancel():
    result = json.loads(
        select_many_tool("Pick directories", ["cache"], callback=lambda *_: [])
    )

    assert result["selected_choices"] == []
    assert result["cancelled"] is True


def test_rejects_callback_values_not_offered():
    result = json.loads(
        select_many_tool(
            "Pick directories",
            ["cache", "dist"],
            callback=lambda *_: [".git"],
        )
    )

    assert "error" in result
    assert ".git" not in result.get("selected_choices", [])


def test_rejects_too_many_choices_without_silent_truncation():
    choices = [f"dir-{index}" for index in range(MAX_SELECT_MANY_CHOICES + 1)]
    called = False

    def callback(*_args):
        nonlocal called
        called = True
        return []

    result = json.loads(select_many_tool("Pick directories", choices, callback=callback))

    assert "error" in result
    assert called is False


def test_schema_is_explicitly_multi_select():
    assert SELECT_MANY_SCHEMA["name"] == "select_many"
    assert SELECT_MANY_SCHEMA["parameters"]["required"] == ["question", "choices"]
    choices = SELECT_MANY_SCHEMA["parameters"]["properties"]["choices"]
    assert choices["maxItems"] == MAX_SELECT_MANY_CHOICES


def test_agent_runtime_dispatches_to_multi_select_callback():
    from agent.agent_runtime_helpers import invoke_tool

    agent = SimpleNamespace(
        _memory_manager=None,
        select_many_callback=lambda _question, choices: [choices[1]],
        session_id="session-many",
        valid_tool_names={"select_many"},
        enabled_toolsets=["select_many"],
        disabled_toolsets=None,
        _current_turn_id="turn-many",
        _current_api_request_id="request-many",
    )

    result = json.loads(
        invoke_tool(
            agent,
            "select_many",
            {"question": "Pick directories", "choices": ["cache", "dist"]},
            "task-many",
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
        )
    )

    assert result["selected_choices"] == ["dist"]
    assert result["cancelled"] is False


def test_feishu_bundle_exposes_select_many_without_growing_core():
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import _HERMES_CORE_TOOLS, resolve_toolset
    from tools.delegate_tool import DELEGATE_BLOCKED_TOOLS

    assert "select_many" not in _HERMES_CORE_TOOLS
    assert "select_many" in resolve_toolset("hermes-feishu")
    assert "select_many" not in resolve_toolset("hermes-cli")
    assert "select_many" in _get_platform_tools({}, "feishu")
    assert "select_many" not in _get_platform_tools({}, "cli")

    explicit_config = {
        "platform_toolsets": {
            "feishu": ["terminal", "file", "clarify"],
        }
    }
    assert "select_many" in _get_platform_tools(explicit_config, "feishu")
    assert "select_many" in DELEGATE_BLOCKED_TOOLS
