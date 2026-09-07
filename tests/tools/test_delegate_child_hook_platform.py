"""Inside a gateway turn, a delegate child's tool-hook ``platform`` stays ``"subagent"`` (the
construction identity, as ``_parent_identity_kwargs`` and hooks.md promise) even though the child
runs under a copy of the parent's session ContextVars; the human's user/chat identity is still
inherited, so a policy plugin can tell a subagent-originated call from the human's own."""

import contextvars
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS
from agent.inline_tool_executors import principal_hook_fields
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.delegate_tool import _build_child_agent

SESSION_KEY = "agent:main:discord:channel:c1"


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


def _fake_agent(**kwargs):
    """Stand-in AIAgent exposing the construction identity the way agent_init stores it."""
    child = SimpleNamespace(platform=kwargs.get("platform"), session_id="child-sess", _delegate_depth=1)
    for name in _GATEWAY_IDENTITY_PARAMS:
        setattr(child, f"_{name}", kwargs.get(name))
    return child


def _build_child(parent):
    with patch("run_agent.AIAgent", side_effect=_fake_agent):
        return _build_child_agent(
            task_index=0, goal="do it", context=None, toolsets=None, model=None,
            max_iterations=5, task_count=1, parent_agent=parent,
        )


@pytest.fixture
def gateway_turn():
    """Same binding gateway/run_turn.py performs (``_set_session_env``) before running the agent."""
    tokens = set_session_vars(
        platform="discord", chat_id="c1", chat_type="channel", user_id="A", user_name="alice",
        session_key=SESSION_KEY,
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


def test_child_hook_platform_is_subagent_inside_gateway_turn(gateway_turn):
    parent = _make_parent(user_id="A", user_name="alice", chat_id="c1", chat_type="channel",
                          gateway_session_key=SESSION_KEY)
    child = _build_child(parent)
    assert child.platform == "subagent"

    # delegate_tool_child_run / delegate_tool_dispatch run the child under copy_context().run; the
    # child's tool loop then calls tool_hook_ids -> principal_hook_fields.
    fields = contextvars.copy_context().run(principal_hook_fields, child)
    assert fields["user_id"] == "A"
    assert fields["chat_id"] == "c1"
    assert fields["gateway_session_key"] == SESSION_KEY  # the parent's chat, via the ContextVar
    assert fields["platform"] == "subagent", fields


def test_child_hook_platform_is_subagent_outside_gateway():
    """Control: with no session bound, the construction value is what hooks see."""
    reset_session_vars()
    child = _build_child(_make_parent(user_id="A"))
    fields = contextvars.copy_context().run(principal_hook_fields, child)
    assert fields["platform"] == "subagent"
    assert fields["user_id"] == "A"
