"""Tests for the ``delegate_tool_reply`` explicit delivery channel.

Covers:
- handler: records to agent-instance state and fails closed without it
- extraction: no-call fallback, single call, multi-call append
- visibility: delegation_reply not in CONFIGURABLE_TOOLSETS, not in default
  tool definitions, but present in child toolsets built by _build_child_agent
  (validated via constructor-capture pattern from test_delegate.py)
- system prompt discipline injection

Real compaction, cleanup and transport regressions live in test_delegate_reply_runtime.py.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tools.delegate_tool as dt
from tools.delegate_tool_child_run import _extract_reply_deliverable
import tools.delegate_tool_reply as dtr
from tools.registry import registry


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _make_agent(subagent_id="child-123"):
    """A lightweight agent stand-in with real attribute semantics.

    ``MagicMock()`` returns a mock for ANY ``getattr`` (including
    ``_delegate_reply_chunks``), so the handler's ``is None`` check never
    fires and the list never gets created. ``SimpleNamespace`` has real
    attribute access with ``AttributeError`` on missing attrs, which mirrors
    how the handler interacts with a real AIAgent instance.
    """
    agent = SimpleNamespace()
    agent._subagent_id = subagent_id
    return agent


def test_handler_records_to_agent_instance():
    agent = _make_agent("child-123")
    result = dtr.delegate_tool_reply(content="my deliverable", parent_agent=agent)
    data = json.loads(result)
    assert data == {"acknowledged": True}
    assert agent._delegate_reply_chunks == ["my deliverable"]


def test_handler_multi_call_appends_to_instance_list():
    agent = _make_agent("child-456")
    dtr.delegate_tool_reply(content="chunk1", parent_agent=agent)
    dtr.delegate_tool_reply(content="chunk2", parent_agent=agent)
    assert agent._delegate_reply_chunks == ["chunk1", "chunk2"]


def test_handler_no_agent_fails_closed():
    result = dtr.delegate_tool_reply(content="orphan deliverable", parent_agent=None)
    data = json.loads(result)
    assert data["acknowledged"] is False
    assert "subagent context" in data["error"]


# ---------------------------------------------------------------------------
# Extraction: _extract_reply_deliverable (reads from agent instance)
# ---------------------------------------------------------------------------

def test_extraction_no_chunks_returns_none():
    child = MagicMock()
    # No _delegate_reply_chunks attribute set
    del child._delegate_reply_chunks
    assert _extract_reply_deliverable(child) is None


def test_extraction_single_chunk():
    child = MagicMock()
    child._delegate_reply_chunks = ["THE REPORT"]
    assert _extract_reply_deliverable(child) == "THE REPORT"


def test_extraction_multi_chunk_appends_in_order():
    child = MagicMock()
    child._delegate_reply_chunks = ["part1", "part2"]
    assert _extract_reply_deliverable(child) == "part1\n\npart2"


def test_extraction_empty_list_means_no_delivery_call():
    child = MagicMock()
    child._delegate_reply_chunks = []
    assert _extract_reply_deliverable(child) is None


def test_extraction_non_list_returns_none():
    child = MagicMock()
    child._delegate_reply_chunks = "not a list"
    assert _extract_reply_deliverable(child) is None


# ---------------------------------------------------------------------------
# Visibility / toolset membership (constructor-capture pattern)
# ---------------------------------------------------------------------------

def _make_mock_parent():
    """Create a mock parent matching test_delegate.py's _make_mock_parent."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = MagicMock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


@pytest.mark.parametrize("parent_depth, expected_role", [(0, "orchestrator"), (1, "leaf")])
def test_build_child_agent_includes_delegation_reply(parent_depth, expected_role):
    """Exercise the real _build_child_agent, not a hand-written append."""
    parent = _make_mock_parent()
    parent.enabled_toolsets = ["terminal", "file"]
    parent._delegate_depth = parent_depth

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=2),
        patch("tools.delegate_tool._get_orchestrator_enabled", return_value=True),
    ):
        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            MockAgent.return_value = mock_child

            dt._build_child_agent(
                task_index=0,
                goal="Test delivery channel",
                context=None,
                toolsets=["terminal", "file"],
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )

        enabled = MockAgent.call_args[1]["enabled_toolsets"]
        assert "delegation_reply" in enabled
        assert mock_child._delegate_role == expected_role
        assert mock_child._delegate_reply_chunks == []
        assert ("delegation" in enabled) == (expected_role == "orchestrator")
        assert parent.enabled_toolsets == ["terminal", "file"]
        from model_tools import get_tool_definitions
        definitions = get_tool_definitions(
            enabled_toolsets=enabled, disabled_toolsets=MockAgent.call_args[1]["disabled_toolsets"],
            quiet_mode=True, skip_tool_search_assembly=True,
        )
        assert "delegate_tool_reply" in {t["function"]["name"] for t in definitions}


def test_parent_disabled_toolsets_cannot_remove_delivery_channel():
    parent = _make_mock_parent()
    parent.enabled_toolsets = ["terminal", "file"]
    parent.disabled_toolsets = ["browser", "delegation_reply"]

    with patch("tools.delegate_tool._load_config", return_value={}):
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            dt._build_child_agent(
                task_index=0,
                goal="Test delivery channel",
                context=None,
                toolsets=["terminal", "file"],
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
            )

    kwargs = MockAgent.call_args[1]
    assert "delegation_reply" in kwargs["enabled_toolsets"]
    assert "delegation_reply" not in kwargs["disabled_toolsets"]
    assert "browser" in kwargs["disabled_toolsets"]


def test_delegation_reply_not_in_configurable_toolsets():
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS
    keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    assert "delegation_reply" not in keys


def test_delegation_reply_not_in_core_tools():
    from toolsets import _HERMES_CORE_TOOLS
    assert "delegate_tool_reply" not in _HERMES_CORE_TOOLS


@pytest.mark.parametrize("toolset", [None, "hermes-cli", "hermes-telegram"])
def test_parent_default_schemas_do_not_include_delivery(toolset):
    from model_tools import get_tool_definitions
    definitions = get_tool_definitions(
        enabled_toolsets=[toolset] if toolset else None, quiet_mode=True, skip_tool_search_assembly=True,
    )
    assert "delegate_tool_reply" not in {t["function"]["name"] for t in definitions}


def test_delegation_reply_toolset_resolves():
    from toolsets import get_toolset
    ts = get_toolset("delegation_reply")
    assert ts is not None
    assert "delegate_tool_reply" in ts["tools"]


def test_tool_registered_under_delegation_reply_toolset():
    entry = registry.get_entry("delegate_tool_reply")
    assert entry is not None
    assert entry.toolset == "delegation_reply"


# ---------------------------------------------------------------------------
# System prompt discipline injection
# ---------------------------------------------------------------------------

def test_system_prompt_includes_delivery_discipline():
    prompt = dt._build_child_system_prompt(
        "do the task", role="leaf", max_spawn_depth=2, child_depth=1,
    )
    assert "delegate_tool_reply" in prompt
    assert "Delivery Discipline" in prompt


def test_system_prompt_discipline_present_for_orchestrator():
    prompt = dt._build_child_system_prompt(
        "do the task", role="orchestrator", max_spawn_depth=2, child_depth=1,
    )
    assert "delegate_tool_reply" in prompt


# ---------------------------------------------------------------------------
# Dispatch path — agent-level interception
# ---------------------------------------------------------------------------

def test_dispatch_via_registry_fails_without_agent_reference():
    """Registry dispatch does NOT forward ``parent_agent`` — only task_id,
    session_id, user_task. This is the root cause of the anchorless delivery
    bug: when ``delegate_tool_reply`` was routed through registry.dispatch,
    the handler received ``parent_agent=None`` and could not record the chunk
    on the agent instance. The handler must fail closed rather than acknowledge
    a result that the parent can never recover.
    """
    # Simulate what registry.dispatch forwards to the handler.
    result = dtr._handle_delegate_tool_reply(
        {"content": "delivered via registry"},
        task_id="t1",
        session_id="s1",
    )
    data = json.loads(result)
    assert data["acknowledged"] is False


def test_agent_level_intercept_passes_agent_to_handler():
    """When the tool_executor intercepts ``delegate_tool_reply`` as an
    agent-level tool, it passes the agent instance directly to the handler —
    NOT through registry.dispatch. This is the fix for the anchorless delivery
    bug where registry.dispatch lost the agent reference."""
    agent = _make_agent("child-intercept-test")
    result = dtr.delegate_tool_reply(
        content="delivered via agent-level intercept",
        parent_agent=agent,
    )
    data = json.loads(result)
    assert data["acknowledged"] is True
    assert agent._delegate_reply_chunks == ["delivered via agent-level intercept"]


def test_delegate_reply_is_never_parallelized():
    from agent.tool_dispatch_helpers import _NEVER_PARALLEL_TOOLS

    assert "delegate_tool_reply" in _NEVER_PARALLEL_TOOLS


@pytest.fixture
def runtime_agent():
    """Real AIAgent dispatch surface with network/tool discovery mocked."""
    from run_agent import AIAgent

    tool_defs = [
        {
            "type": "function",
            "function": dtr.DELEGATE_TOOL_REPLY_SCHEMA,
        }
    ]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._delegate_reply_chunks = []
    yield agent
    agent.close()


def test_runtime_invoke_path_records_delivery(runtime_agent):
    result = runtime_agent._invoke_tool(
        "delegate_tool_reply",
        {"content": "concurrent-path result"},
        "task-1",
    )

    assert json.loads(result)["acknowledged"] is True
    assert runtime_agent._delegate_reply_chunks == ["concurrent-path result"]


def test_sequential_executor_path_records_delivery(runtime_agent):
    tool_call = SimpleNamespace(
        id="call-reply",
        type="function",
        function=SimpleNamespace(
            name="delegate_tool_reply",
            arguments=json.dumps({"content": "sequential-path result"}),
        ),
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages = []

    runtime_agent._execute_tool_calls_sequential(
        assistant_message,
        messages,
        "task-1",
    )

    assert runtime_agent._delegate_reply_chunks == ["sequential-path result"]
    assert len(messages) == 1
    assert json.loads(messages[0]["content"])["acknowledged"] is True
