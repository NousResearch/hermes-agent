"""Regression tests for delegate_task ACP routing.

Background: Prior to this PR, ``delegate_task(acp_command=...)`` accepted the
parameter but silently ignored it -- child agents used the parent's HTTP provider.
These tests ensure the parameter now actually routes to CopilotACPClient.
"""

import pytest
from unittest.mock import MagicMock, patch
from tools.delegate_tool import _build_child_agent


def _make_parent_agent():
    parent = MagicMock()
    parent.model = "anthropic/claude-opus-4.6"
    parent.provider = "anthropic"
    parent.base_url = "https://api.anthropic.com"
    parent.api_key = "sk-ant-fake"
    parent.api_mode = None
    parent.acp_command = None
    parent.acp_args = []
    parent.enabled_toolsets = ["terminal"]
    parent.valid_tool_names = {"terminal", "read_file"}
    parent.max_tokens = 4096
    parent.reasoning_config = None
    parent.prefill_messages = None
    parent.platform = "test"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent.session_id = "test-session"
    parent._active_children = []
    parent._delegate_depth = 0
    return parent


def test_delegate_without_acp_command_uses_parent_provider():
    parent = _make_parent_agent()
    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(
            task_index=0, goal="test", context=None, toolsets=None,
            model=None, max_iterations=5, parent_agent=parent,
        )
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["provider"] == "anthropic"
        assert not kwargs.get("acp_command")


def test_delegate_with_acp_command_switches_to_copilot_acp_provider():
    parent = _make_parent_agent()
    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(
            task_index=0, goal="test", context=None, toolsets=None,
            model=None, max_iterations=5, parent_agent=parent,
            override_acp_command="claude-agent-acp",
        )
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["provider"] == "copilot-acp", (
            f"Expected child provider to switch to copilot-acp when acp_command is set, "
            f"got {kwargs['provider']}"
        )
        assert kwargs["base_url"] == "acp://copilot"
        assert kwargs["acp_command"] == "claude-agent-acp"


def test_delegate_with_acp_command_respects_explicit_provider_override():
    """If override_provider is explicitly set, don't switch to copilot-acp."""
    parent = _make_parent_agent()
    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(
            task_index=0, goal="test", context=None, toolsets=None,
            model=None, max_iterations=5, parent_agent=parent,
            override_provider="openrouter",
            override_acp_command="claude-agent-acp",
        )
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["provider"] == "openrouter"


def test_delegate_with_acp_command_seeds_api_key_when_missing():
    """When acp_command triggers ACP switch and parent has no api_key, seed a stub."""
    parent = _make_parent_agent()
    parent.api_key = None
    parent._client_kwargs = {}
    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(
            task_index=0, goal="test", context=None, toolsets=None,
            model=None, max_iterations=5, parent_agent=parent,
            override_acp_command="claude-agent-acp",
        )
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["api_key"] == "acp-child-agent"


def test_delegate_with_acp_command_bypasses_provider_filters():
    """ACP switch should bypass parent provider allow/ignore lists."""
    parent = _make_parent_agent()
    parent.providers_allowed = ["anthropic"]
    parent.providers_ignored = ["copilot-acp"]
    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(
            task_index=0, goal="test", context=None, toolsets=None,
            model=None, max_iterations=5, parent_agent=parent,
            override_acp_command="claude-agent-acp",
        )
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["providers_allowed"] is None
        assert kwargs["providers_ignored"] is None


def test_copilot_acp_client_respects_empty_acp_args():
    """Regression: empty list for acp_args must be respected, not fall through to default."""
    from agent.copilot_acp_client import CopilotACPClient
    client = CopilotACPClient(acp_command="claude-agent-acp", acp_args=[])
    assert client._acp_args == [], (
        f"Expected empty args to be preserved, got {client._acp_args}"
    )
    client.close()


def test_copilot_acp_client_falls_through_to_default_args_when_none():
    """When acp_args is None, fall through to _resolve_args() default."""
    from agent.copilot_acp_client import CopilotACPClient
    client = CopilotACPClient(acp_command="copilot")
    # Default should be ["--acp", "--stdio"] from _resolve_args
    assert client._acp_args == ["--acp", "--stdio"]
    client.close()


def test_create_openai_client_routes_acp_marker_base_url():
    """_create_openai_client should route any acp:// base_url to ACP client."""
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)  # skip __init__
    agent.provider = "anthropic"
    agent._client_log_context = lambda: ""
    with patch("agent.copilot_acp_client.CopilotACPClient") as MockClient:
        MockClient.return_value = MagicMock()
        agent._create_openai_client(
            {"base_url": "acp://claude", "api_key": "x"},
            reason="test", shared=False,
        )
        MockClient.assert_called_once()
