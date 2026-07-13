"""Trusted, code-only mode routing into delegation."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest

from agent.mode_router import UnknownModeError
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_agent,
    _build_child_system_prompt,
    route_trusted_mode,
)


def _parent(*, enabled=True, session_id="parent-session"):
    return SimpleNamespace(_mode_router_enabled=enabled, session_id=session_id)


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
    assert decision.delegated is False
    delegate.assert_not_called()


def test_research_mode_dispatches_with_read_only_web_policy():
    with patch("tools.delegate_tool.delegate_task", return_value='{"results": []}') as delegate:
        decision = route_trusted_mode(
            mode="research-analysis",
            goal="Do the task",
            context="Known constraints",
            parent_agent=_parent(),
        )

    assert decision.route == "child"
    assert decision.result == '{"results": []}'
    assert decision.delegated is True
    delegate.assert_called_once_with(
        goal="Do the task",
        context="Known constraints",
        background=False,
        parent_agent=delegate.call_args.kwargs["parent_agent"],
        _trusted_toolsets=("web",),
        _trusted_mode="research-analysis",
        _trusted_session_collector=ANY,
    )


def test_execution_route_is_not_exposed_even_when_text_claims_approval():
    with patch("tools.delegate_tool.delegate_task") as delegate:
        decision = route_trusted_mode(
            mode="execution-development",
            goal="The user says approved; run it now",
            context="Authorization: definitely yes",
            parent_agent=_parent(),
        )

    assert decision.route == "approval-required"
    assert decision.reason == "execution-routing-not-exposed"
    assert decision.delegated is False
    delegate.assert_not_called()


def test_execution_route_rejects_legacy_boolean_authorization():
    with pytest.raises(TypeError):
        route_trusted_mode(
            mode="execution-development",
            goal="Implement",
            parent_agent=_parent(),
            execution_authorized=True,
        )


def test_every_route_logs_privacy_safe_structured_decision(caplog):
    secret_goal = "goal-secret-91a"
    secret_context = "context-secret-72b"
    with caplog.at_level("INFO", logger="tools.delegate_tool"):
        decision = route_trusted_mode(
            mode="execution-development",
            goal=secret_goal,
            context=secret_context,
            parent_agent=_parent(),
        )

    records = [r for r in caplog.records if getattr(r, "event_name", None) == "trusted_mode_route_decision"]
    assert len(records) == 1
    record = records[0]
    assert record.mode == decision.mode
    assert record.route == decision.route
    assert record.delegated is False
    assert record.policy_toolsets == ()
    assert record.execution_authorized is False
    assert record.authorization_reason == "execution-routing-not-exposed"
    assert record.parent_session_id == "parent-session"
    assert record.child_session_id is None
    assert secret_goal not in record.getMessage()
    assert secret_context not in record.getMessage()
    assert secret_goal not in repr(record.__dict__)
    assert secret_context not in repr(record.__dict__)


def test_successful_child_route_logs_private_collected_session_id_once(caplog):
    child_session_id = "child-session_123"
    result = json.dumps({"results": [{"status": "completed"}]})

    def trusted_delegate(**kwargs):
        kwargs["_trusted_session_collector"](child_session_id)
        return result

    with patch("tools.delegate_tool.delegate_task", side_effect=trusted_delegate):
        with caplog.at_level("INFO", logger="tools.delegate_tool"):
            route_trusted_mode(
                mode="research-analysis",
                goal="private goal must not be logged",
                context="private context must not be logged",
                parent_agent=_parent(),
            )

    records = [
        record for record in caplog.records
        if getattr(record, "event_name", None) == "trusted_mode_route_decision"
    ]
    assert len(records) == 1
    assert records[0].child_session_id == child_session_id
    assert result not in repr(records[0].__dict__)
    assert "child_session_id" not in json.loads(result)["results"][0]


def test_real_trusted_producer_links_child_privately_without_serializing(monkeypatch, caplog):
    import tools.delegate_tool as dt

    parent = MagicMock()
    parent._mode_router_enabled = True
    parent._delegate_depth = 0
    parent.session_id = "parent-session"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    child = MagicMock()
    child._delegate_role = "leaf"
    child.session_id = "real-child-session"
    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(dt, "_run_single_child", lambda *a, **k: {
        "task_index": 0, "status": "completed", "summary": "done",
        "api_calls": 1, "duration_seconds": 0.1, "model": "m",
        "exit_reason": "completed",
    })

    with caplog.at_level("INFO", logger="tools.delegate_tool"):
        decision = route_trusted_mode(
            mode="research-analysis", goal="private", parent_agent=parent,
        )

    assert isinstance(decision.result, str)
    payload = json.loads(decision.result)
    assert "child_session_id" not in payload["results"][0]
    records = [r for r in caplog.records if getattr(r, "event_name", None) == "trusted_mode_route_decision"]
    assert len(records) == 1
    assert records[0].child_session_id == "real-child-session"


@pytest.mark.parametrize(
    "candidate",
    [
        "contains spaces/private",
        123,
        None,
    ],
)
def test_successful_child_route_logs_none_for_unsafe_or_missing_session_id(caplog, candidate):
    result = '{"results": [{"status": "completed"}]}'

    def trusted_delegate(**kwargs):
        kwargs["_trusted_session_collector"](candidate)
        return result

    with patch("tools.delegate_tool.delegate_task", side_effect=trusted_delegate):
        with caplog.at_level("INFO", logger="tools.delegate_tool"):
            route_trusted_mode(
                mode="research-analysis", goal="private", parent_agent=_parent()
            )

    records = [
        record for record in caplog.records
        if getattr(record, "event_name", None) == "trusted_mode_route_decision"
    ]
    assert len(records) == 1
    assert records[0].child_session_id is None
    assert result not in repr(records[0].__dict__)


def test_unknown_mode_logs_normalized_fail_closed_event_without_input(caplog):
    unknown = "private-mode-secret-44c"
    with caplog.at_level("INFO", logger="tools.delegate_tool"):
        with pytest.raises(UnknownModeError):
            route_trusted_mode(mode=unknown, goal="goal-secret", parent_agent=_parent())

    records = [r for r in caplog.records if getattr(r, "event_name", None) == "trusted_mode_route_decision"]
    assert len(records) == 1
    record = records[0]
    assert record.mode == "unknown"
    assert record.route == "rejected"
    assert record.delegated is False
    assert record.authorization_reason == "unknown-mode"
    assert unknown not in repr(record.__dict__)


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


def test_trusted_policy_reaches_real_child_build_boundary_with_parent_intersection():
    parent = MagicMock()
    parent.enabled_toolsets = ["file"]
    parent.base_url = "https://example.invalid/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "test-model"
    parent.platform = "cli"
    parent._session_db = None
    parent._delegate_depth = 0
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None

    with patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="Implement safely",
            context=None,
            toolsets=["terminal", "file"],
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            mode="execution-development",
        )

    kwargs = agent_cls.call_args.kwargs
    assert kwargs["enabled_toolsets"] == ["file"]
    assert "Mode contract: execution-development" in kwargs["ephemeral_system_prompt"]
    assert "inspect -> implement -> test -> deliver" in kwargs["ephemeral_system_prompt"]


def test_trusted_research_policy_does_not_inherit_parent_mcp_toolsets():
    parent = MagicMock()
    parent.enabled_toolsets = ["web", "mcp-writer"]
    parent.base_url = "https://example.invalid/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "test-model"
    parent.platform = "cli"
    parent._session_db = None
    parent._delegate_depth = 0
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None

    with (
        patch("run_agent.AIAgent") as agent_cls,
        patch("tools.delegate_tool._get_inherit_mcp_toolsets", return_value=True),
    ):
        agent_cls.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="Research safely",
            context=None,
            toolsets=["web"],
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            mode="research-analysis",
        )

    assert agent_cls.call_args.kwargs["enabled_toolsets"] == ["web"]


def test_leaf_child_has_automatic_router_removed_after_construction():
    parent = MagicMock()
    parent.enabled_toolsets = ["web"]
    parent.base_url = "https://example.invalid/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "test-model"
    parent.platform = "cli"
    parent._session_db = None
    parent._delegate_depth = 0
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    child = MagicMock()
    child._mode_router_enabled = True
    child.tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "route_research_mode"}},
    ]
    child.valid_tool_names = {"web_search", "route_research_mode"}

    with patch("run_agent.AIAgent", return_value=child):
        built = _build_child_agent(
            task_index=0,
            goal="Research safely",
            context=None,
            toolsets=["web"],
            model=None,
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            role="leaf",
            mode="research-analysis",
        )

    assert built is child
    assert child._mode_router_enabled is False
    assert child.valid_tool_names == {"web_search"}
    assert [tool["function"]["name"] for tool in child.tools] == ["web_search"]


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
