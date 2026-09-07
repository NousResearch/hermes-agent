"""A delegated child inherits the parent's gateway identity (user/chat/thread) so its tool-call hooks
attribute work to the delegating user; ``platform`` stays ``"subagent"`` and the per-chat
``gateway_session_key`` is NOT inherited (a keyed subagent session row would shadow the parent chat).
A parent without gateway identity (CLI) forwards nothing."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS
from tools.delegate_tool import _build_child_agent


def _make_parent(**identity):
    parent = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1", api_key="test-key", provider="openrouter",
        api_mode="chat_completions", model="anthropic/claude-sonnet-4", platform="discord",
        providers_allowed=None, providers_ignored=None, providers_order=None, provider_sort=None,
        _session_db=None, _delegate_depth=0, _active_children=[], _active_children_lock=threading.Lock(),
        _print_fn=None, tool_progress_callback=None, thinking_callback=None,
        enabled_toolsets=None, disabled_toolsets=None,
    )
    for name in _GATEWAY_IDENTITY_PARAMS:
        setattr(parent, f"_{name}", identity.get(name))
    return parent


def _child_kwargs(parent):
    with patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent_cls.return_value = MagicMock()
        _build_child_agent(
            task_index=0, goal="do it", context=None, toolsets=None, model=None,
            max_iterations=5, task_count=1, parent_agent=parent,
        )
    return mock_agent_cls.call_args.kwargs


def test_child_inherits_parent_gateway_identity():
    parent = _make_parent(
        user_id="222", user_name="wife", chat_id="c1", chat_type="channel", thread_id="t9",
        gateway_session_key="k",
    )
    kwargs = _child_kwargs(parent)
    assert kwargs["user_id"] == "222"
    assert kwargs["user_name"] == "wife"
    assert kwargs["chat_id"] == "c1"
    assert kwargs["chat_type"] == "channel"
    assert kwargs["thread_id"] == "t9"
    assert kwargs.get("gateway_session_key") is None
    assert kwargs["platform"] == "subagent"


def test_child_of_identity_less_parent_gets_no_identity():
    kwargs = _child_kwargs(_make_parent())
    assert kwargs["platform"] == "subagent"
    for name in _GATEWAY_IDENTITY_PARAMS:
        assert kwargs.get(name) is None, name
