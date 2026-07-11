from types import SimpleNamespace
import pytest

from agent.chat_completion_helpers import build_api_kwargs
from tools import approval
from tools.todo_tool import TodoStore


def test_exact_plan_capability_is_owner_bound_expiring_and_consumed(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    approval.clear_session("session-1")
    grant = approval.grant_plan_capability(
        session_key="session-1",
        plan_id="plan-1",
        exact_commands=["git status --short"],
        approved_by_user_id="owner-1",
        max_uses_per_command=1,
    )
    assert grant["command_count"] == 1
    assert approval.consume_plan_capability("session-1", "git status --short") == "plan-1"
    assert approval.consume_plan_capability("session-1", "git status --short") is None
    assert approval.consume_plan_capability("session-1", "git status") is None


def test_plan_capability_rejects_non_owner(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner-1"]}},
    )
    with pytest.raises(PermissionError):
        approval.grant_plan_capability(
            session_key="session-1",
            plan_id="plan-1",
            exact_commands=["git status"],
            approved_by_user_id="teammate",
        )


def test_pending_model_plan_forces_tool_choice_required():
    class _Transport:
        def build_kwargs(self, **kwargs):
            return kwargs

    store = TodoStore()
    store.write([{"id": "1", "content": "finish", "status": "pending"}])
    agent = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "todo"}}],
        _todo_store=store,
        api_mode="codex_responses",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        _base_url_hostname="chatgpt.com",
        _base_url_lower="https://chatgpt.com/backend-api/codex",
        reasoning_config={"effort": "max"},
        session_id="s1",
        max_tokens=None,
        request_overrides=None,
        _prepare_messages_for_non_vision_model=lambda messages: messages,
        _resolved_api_call_timeout=lambda: 60,
        _is_copilot_url=lambda: False,
        _github_models_reasoning_extra_body=lambda: None,
        _get_transport=lambda: _Transport(),
    )
    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "do it"}])
    assert kwargs["tool_choice"] == "required"

    store.write([{"id": "1", "content": "finish", "status": "completed"}])
    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "do it"}])
    assert kwargs["tool_choice"] is None
