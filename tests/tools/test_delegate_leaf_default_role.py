"""delegate_task: a child is a leaf unless the caller asks for an orchestrator.

``delegation.max_spawn_depth`` is the ceiling of the delegation tree, not a
promotion rule. Before this contract every child below the ceiling was made an
orchestrator: with ``max_spawn_depth: 3`` a plain "go read these files" task
kept the ``delegation`` toolset and got the "Subagent Spawning" prompt section,
so the tree fanned out on its own even though no caller asked for nesting.
"""

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_dynamic_schema_overrides, delegate_task

_SPAWN_SECTION = "Subagent Spawning (Orchestrator Role)"
_CREDS = {"provider": None, "base_url": None, "api_key": None, "api_mode": None, "model": None}


def _parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://fixture.invalid/v1"
    parent.api_key = "fixture-only"
    parent.provider = "custom"
    parent.api_mode = "chat_completions"
    parent.model = "fixture"
    parent.platform = "cli"
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = ["terminal", "file", "delegation"]
    parent.disabled_toolsets = []
    parent.request_overrides = {}
    return parent


def _child():
    child = MagicMock()
    child.run_conversation.return_value = {"final_response": "done", "completed": True, "api_calls": 1, "messages": []}
    child._delegate_saved_tool_names = []
    child._credential_pool = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.model = "fixture"
    return child


def _spawn(cfg, *, depth=0, tasks=None, **kwargs):
    """Run delegate_task with a mocked AIAgent; return (child, AIAgent kwargs)."""
    with (
        patch("tools.delegate_tool._resolve_delegation_credentials", return_value=_CREDS),
        patch("tools.delegate_tool._load_config", return_value=cfg),
        patch("run_agent.AIAgent") as agent_class,
    ):
        child = _child()
        agent_class.return_value = child
        delegate_task(tasks=tasks or [{"goal": "gather facts"}], parent_agent=_parent(depth), **kwargs)
        return child, agent_class.call_args.kwargs


def _assert_leaf(child, agent_kwargs):
    assert child._delegate_role == "leaf"
    assert "delegation" not in agent_kwargs["enabled_toolsets"]
    assert _SPAWN_SECTION not in agent_kwargs["ephemeral_system_prompt"]


def _assert_orchestrator(child, agent_kwargs):
    assert child._delegate_role == "orchestrator"
    assert "delegation" in agent_kwargs["enabled_toolsets"]
    assert _SPAWN_SECTION in agent_kwargs["ephemeral_system_prompt"]


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_task_without_role_is_leaf_at_every_depth_below_the_ceiling(depth):
    child, kwargs = _spawn({"max_spawn_depth": 4}, depth=depth)
    _assert_leaf(child, kwargs)


def test_per_task_orchestrator_below_ceiling_gets_toolset_and_section():
    child, kwargs = _spawn({"max_spawn_depth": 3}, tasks=[{"goal": "lead", "role": "orchestrator"}])
    _assert_orchestrator(child, kwargs)
    assert "max_spawn_depth=3" in kwargs["ephemeral_system_prompt"]


def test_top_level_role_still_accepted_for_old_callers():
    child, kwargs = _spawn({"max_spawn_depth": 3}, role="orchestrator")
    _assert_orchestrator(child, kwargs)


def test_per_task_leaf_beats_top_level_orchestrator():
    child, kwargs = _spawn({"max_spawn_depth": 3}, role="orchestrator", tasks=[{"goal": "gather", "role": "leaf"}])
    _assert_leaf(child, kwargs)


def test_unknown_role_coerces_to_leaf():
    child, kwargs = _spawn({"max_spawn_depth": 3}, tasks=[{"goal": "gather", "role": "supervisor"}])
    _assert_leaf(child, kwargs)


def test_orchestrator_at_the_ceiling_downgrades_and_logs(caplog):
    caplog.set_level(logging.INFO, logger="tools.delegate_tool")
    # parent depth 1, max_spawn_depth 2 -> the child sits at the leaf floor
    child, kwargs = _spawn({"max_spawn_depth": 2}, depth=1, tasks=[{"goal": "lead", "role": "orchestrator"}])
    _assert_leaf(child, kwargs)
    assert any("downgraded to leaf" in r.getMessage() for r in caplog.records)


def test_kill_switch_forces_leaf_even_when_requested():
    child, kwargs = _spawn(
        {"max_spawn_depth": 3, "orchestrator_enabled": False}, tasks=[{"goal": "lead", "role": "orchestrator"}],
    )
    _assert_leaf(child, kwargs)


def _task_props(cfg):
    with patch("tools.delegate_tool._load_config", return_value=cfg):
        overrides = _build_dynamic_schema_overrides()
    return overrides, overrides["parameters"]["properties"]["tasks"]["items"]["properties"]


@pytest.mark.parametrize("cfg", [{}, {"max_spawn_depth": 1}, {"max_spawn_depth": 3, "orchestrator_enabled": False}])
def test_role_not_advertised_where_nesting_is_unavailable(cfg):
    """Flat installs (the default) pay no schema tokens for a knob they cannot use."""
    overrides, task_props = _task_props(cfg)
    assert "role" not in task_props
    assert "role" not in overrides["parameters"]["properties"]
    assert "role=" not in overrides["description"]


def test_role_advertised_per_task_when_nesting_is_available():
    overrides, task_props = _task_props({"max_spawn_depth": 3})
    assert task_props["role"]["enum"] == ["leaf", "orchestrator"]
    assert "Default leaf" in task_props["role"]["description"]
    # tasks[] is the only advertised shape; the top-level role stays unadvertised.
    assert "role" not in overrides["parameters"]["properties"]
    assert "role='orchestrator'" in overrides["description"]
    assert "max_spawn_depth=3" in overrides["description"]
