"""Tests for deferred tool loading and the tool_search tool."""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from model_tools import get_tool_definitions
import run_agent
from run_agent import AIAgent
from tools.registry import ToolRegistry, registry


def _dummy_handler(args, **kwargs):
    return json.dumps({"ok": True})


def _schema(name: str, description: str = "") -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


def _tool_call(name: str, arguments: str = "{}", call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _assistant_message(*tool_calls: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=list(tool_calls))


def _chat_response(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls or []),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="test-model",
    )


def _make_agent(*, enabled_toolsets: list[str], hermes_home) -> AIAgent:
    with (
        patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        patch.object(run_agent, "_hermes_home", hermes_home),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=enabled_toolsets,
        )
    agent.client = MagicMock()
    agent.tool_progress_callback = lambda *args, **kwargs: None
    return agent


@pytest.fixture
def register_tools():
    registered = []

    def _register(
        name: str,
        *,
        toolset: str = "c2_tool_search_suite",
        description: str = "",
        search_hint: str = "",
        deferred: bool = False,
        always_load: bool = False,
    ) -> str:
        registry.register(
            name=name,
            toolset=toolset,
            schema=_schema(name, description),
            handler=_dummy_handler,
            description=description,
            search_hint=search_hint,
            deferred=deferred,
            always_load=always_load,
        )
        registered.append(name)
        return name

    yield _register

    for name in reversed(registered):
        registry.deregister(name)


def test_tool_search_finds_tools_by_name_substring(register_tools):
    name = register_tools(
        "c2_name_match_alpha_tool",
        description="Useful for exact substring matching tests.",
    )

    payload = json.loads(registry.dispatch("tool_search", {"query": "name_match_alpha"}))
    names = {item["name"] for item in payload["results"]}
    assert name in names


def test_tool_search_finds_tools_by_search_hint_keyword(register_tools):
    name = register_tools(
        "c2_hint_lookup_tool",
        description="Hidden behind a search hint.",
        search_hint="quasar orchestration pipeline",
    )

    payload = json.loads(registry.dispatch("tool_search", {"query": "quasar"}))
    names = {item["name"] for item in payload["results"]}
    assert name in names


def test_tool_search_returns_deferred_tools(register_tools):
    name = register_tools(
        "c2_deferred_lookup_tool",
        description="Deferred search target.",
        search_hint="nebula deferred capability",
        deferred=True,
    )

    payload = json.loads(registry.dispatch("tool_search", {"query": "nebula"}))
    deferred_matches = {
        item["name"]: item["deferred"]
        for item in payload["results"]
    }
    assert deferred_matches[name] is True


def test_tool_search_result_includes_expected_fields(register_tools):
    name = register_tools(
        "c2_field_probe_tool",
        toolset="c2_field_probe_toolset",
        description="Field coverage target.",
        search_hint="field probe discovery",
        deferred=True,
    )

    payload = json.loads(registry.dispatch("tool_search", {"query": "field probe"}))
    result = next(item for item in payload["results"] if item["name"] == name)
    assert set(result.keys()) == {"name", "description", "toolset", "search_hint", "deferred"}


def test_deferred_tool_not_in_get_tool_definitions_default_output(register_tools):
    toolset = "c2_deferred_toolset"
    name = register_tools(
        "c2_deferred_not_loaded_tool",
        toolset=toolset,
        description="Deferred by default.",
        search_hint="manual activation required",
        deferred=True,
    )

    default_defs = get_tool_definitions(enabled_toolsets=[toolset], quiet_mode=True)
    default_names = {tool["function"]["name"] for tool in default_defs}
    assert name not in default_names

    activated_defs = get_tool_definitions(
        enabled_toolsets=[toolset],
        quiet_mode=True,
        activated_tools=[name],
    )
    activated_names = {tool["function"]["name"] for tool in activated_defs}
    assert name in activated_names


def test_tool_search_activates_deferred_tools_in_agent_runtime(register_tools, tmp_path):
    toolset = "c2_runtime_activation_toolset"
    deferred_name = register_tools(
        "c2_runtime_deferred_tool",
        toolset=toolset,
        description="Deferred runtime activation target.",
        search_hint="orion activation",
        deferred=True,
    )
    hidden_name = register_tools(
        "c2_still_hidden_tool",
        toolset=toolset,
        description="Should remain hidden until searched.",
        search_hint="andromeda hidden",
        deferred=True,
    )
    always_name = register_tools(
        "c2_runtime_always_tool",
        toolset="c2_runtime_always_toolset",
        description="Always visible regardless of activation.",
        always_load=True,
    )

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    agent = _make_agent(enabled_toolsets=[toolset], hermes_home=hermes_home)
    assert deferred_name not in agent.valid_tool_names
    assert hidden_name not in agent.valid_tool_names
    assert always_name in agent.valid_tool_names
    assert "tool_search" in agent.valid_tool_names

    tool_call = _tool_call("tool_search", arguments='{"query":"orion"}', call_id="search-1")
    assistant_message = _assistant_message(tool_call)
    messages = []

    with patch(
        "run_agent.handle_function_call",
        side_effect=lambda function_name, function_args, *_args, **_kwargs: registry.dispatch(function_name, function_args),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    assert deferred_name in agent._activated_deferred_tools
    assert hidden_name not in agent._activated_deferred_tools
    assert deferred_name in agent.valid_tool_names
    assert hidden_name not in agent.valid_tool_names
    assert always_name in agent.valid_tool_names

    next_defs = get_tool_definitions(
        enabled_toolsets=[toolset],
        quiet_mode=True,
        activated_tools=sorted(agent._activated_deferred_tools),
    )
    next_names = {tool["function"]["name"] for tool in next_defs}
    assert deferred_name in next_names
    assert hidden_name not in next_names
    assert always_name in next_names


def test_tool_search_activation_updates_real_run_conversation_loop(register_tools, tmp_path):
    toolset = "c2_conversation_activation_toolset"
    deferred_name = register_tools(
        "c2_conversation_deferred_tool",
        toolset=toolset,
        description="Deferred tool activated during the real agent loop.",
        search_hint="orion conversation activation",
        deferred=True,
    )

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    agent = _make_agent(enabled_toolsets=[toolset], hermes_home=hermes_home)
    assert isinstance(agent._activated_deferred_tools, set)
    assert deferred_name not in agent._activated_deferred_tools
    assert deferred_name not in agent.valid_tool_names

    create_tool_names = []
    search_call = _tool_call("tool_search", arguments='{"query":"orion conversation"}', call_id="search-1")
    responses = [
        _chat_response(tool_calls=[search_call], finish_reason="tool_calls"),
        _chat_response(content="done"),
    ]

    def _create(**kwargs):
        tool_names = {
            tool["function"]["name"]
            for tool in (kwargs.get("tools") or [])
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        create_tool_names.append(tool_names)
        return responses.pop(0)

    agent.client.chat.completions.create.side_effect = _create

    result = agent.run_conversation("Find the right tool")

    assert result["final_response"] == "done"
    assert deferred_name in agent._activated_deferred_tools
    assert len(create_tool_names) >= 2
    assert deferred_name not in create_tool_names[0]
    assert deferred_name in create_tool_names[1]

    next_defs = get_tool_definitions(
        enabled_toolsets=[toolset],
        quiet_mode=True,
        activated_tools=sorted(agent._activated_deferred_tools),
    )
    next_names = {tool["function"]["name"] for tool in next_defs}
    assert deferred_name in next_names


def test_tool_search_activation_hydrates_from_history(register_tools, tmp_path):
    toolset = "c2_history_activation_toolset"
    deferred_name = register_tools(
        "c2_history_deferred_tool",
        toolset=toolset,
        description="Recovered from prior search history.",
        search_hint="perseus restore",
        deferred=True,
    )

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    agent = _make_agent(enabled_toolsets=[toolset], hermes_home=hermes_home)
    search_result = registry.dispatch("tool_search", {"query": "perseus"})
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-restore",
                    "call_id": "call-restore",
                    "type": "function",
                    "function": {"name": "tool_search", "arguments": '{"query":"perseus"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-restore",
            "content": search_result,
        },
    ]

    agent._hydrate_activated_deferred_tools_from_history(history)
    agent._refresh_tool_definitions()

    assert deferred_name in agent._activated_deferred_tools
    assert deferred_name in agent.valid_tool_names


def test_always_load_tool_always_in_get_tool_definitions_output(register_tools):
    name = register_tools(
        "c2_always_loaded_tool",
        toolset="c2_always_loaded_toolset",
        description="Must always be present.",
        always_load=True,
    )

    defs = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
    names = {tool["function"]["name"] for tool in defs}
    assert name in names


def test_normal_tool_in_get_tool_definitions_output(register_tools):
    toolset = "c2_normal_toolset"
    name = register_tools(
        "c2_normal_loaded_tool",
        toolset=toolset,
        description="Normal tool inclusion path.",
    )

    defs = get_tool_definitions(enabled_toolsets=[toolset], quiet_mode=True)
    names = {tool["function"]["name"] for tool in defs}
    assert name in names


def test_invalid_deferred_tool_name_is_ignored_during_activation(register_tools, tmp_path):
    toolset = "c2_invalid_activation_toolset"
    valid_name = register_tools(
        "c2_valid_activation_tool",
        toolset=toolset,
        description="Valid deferred activation target.",
        search_hint="lyra valid",
        deferred=True,
    )

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    agent = _make_agent(enabled_toolsets=[toolset], hermes_home=hermes_home)
    activated = agent._activate_deferred_tools_from_search_result(
        json.dumps(
            {
                "query": "mixed",
                "count": 2,
                "results": [
                    {"name": valid_name, "deferred": True},
                    {"name": "c2_missing_activation_tool", "deferred": True},
                ],
            }
        )
    )

    assert activated == {valid_name}
    assert agent._activated_deferred_tools == {valid_name}
    assert "c2_missing_activation_tool" not in agent._activated_deferred_tools


def test_deferred_and_always_load_on_same_tool_raises_value_error():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="deferred and always_load"):
        reg.register(
            name="c2_invalid_tool",
            toolset="c2_invalid_toolset",
            schema=_schema("c2_invalid_tool", "invalid"),
            handler=_dummy_handler,
            deferred=True,
            always_load=True,
        )
