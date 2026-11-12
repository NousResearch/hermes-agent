"""ask_parent is offered only to delegated children, and only while delegation.ask_parent.enabled is on:
nobody else's tool list changes (prompt-cache safe, no schema change for users who don't opt in)."""
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def enabled(monkeypatch):
    from tools import ask_parent, delegate_tool_config

    settings = {"enabled": True}
    monkeypatch.setattr(delegate_tool_config, "_load_config", lambda: {"ask_parent": settings})
    monkeypatch.setattr(ask_parent, "_questions", {})
    monkeypatch.setattr(ask_parent, "_asked", {})
    return settings


def _agent(depth=1):
    return SimpleNamespace(_delegate_depth=depth, _subagent_id="sa-1" if depth else None, tools=[{
        "type": "function", "function": {"name": "terminal", "parameters": {}}}], valid_tool_names={"terminal"})


def _names(agent):
    return [t["function"]["name"] for t in agent.tools]


def test_a_delegated_child_gets_the_tool_once(enabled):
    from tools.ask_parent import ensure_ask_parent_tool

    child = _agent()

    assert ensure_ask_parent_tool(child) is True
    assert ensure_ask_parent_tool(child) is True
    assert _names(child) == ["terminal", "ask_parent"]
    assert "ask_parent" in child.valid_tool_names


def test_the_top_level_agent_never_gets_it(enabled):
    from tools.ask_parent import ensure_ask_parent_tool

    master = _agent(depth=0)

    assert ensure_ask_parent_tool(master) is False
    assert _names(master) == ["terminal"]


def test_nothing_changes_while_the_feature_is_off(enabled):
    from tools.ask_parent import ensure_ask_parent_tool

    enabled["enabled"] = False
    child = _agent()

    assert ensure_ask_parent_tool(child) is False
    assert _names(child) == ["terminal"] and child.valid_tool_names == {"terminal"}


def test_the_schema_tells_the_child_when_to_ask_and_what_comes_back():
    from tools.ask_parent import ask_parent_tool_schema

    function = ask_parent_tool_schema()["function"]
    props = function["parameters"]["properties"]

    assert function["name"] == "ask_parent"
    assert set(props) == {"question", "state", "options"}
    assert function["parameters"]["required"] == ["question"]
    for outcome in ("answered", "timed_out", "abandoned", "parent_unavailable"):
        assert outcome in function["description"]


def test_the_agent_dispatches_the_call_with_its_live_self(enabled):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext

    child = _agent()
    child._async_unit_id = None  # joined: answered at once, without a parent

    result = json.loads(INLINE_TOOL_EXECUTORS["ask_parent"](child, {"question": "Which branch?"},
                                                            InlineToolContext(effective_task_id="t")))

    assert result["outcome"] == "parent_unavailable"


def test_the_defaults_keep_it_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["delegation"]["ask_parent"] == {"enabled": False, "max_questions": 3, "timeout_seconds": 1800}
