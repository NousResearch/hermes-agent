"""Child-facing nesting guidance must agree with depth-derived capability."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools import delegate_tool
from tools.registry import registry


@pytest.mark.parametrize("legacy_role", [None, "leaf", "orchestrator"])
@pytest.mark.parametrize(
    "parent_depth,max_depth,enabled",
    [(1, 5, True), (1, 3, True), (2, 3, True), (1, 5, False), (1, 5, 0), (1, 5, 0.0)],
)
def test_dispatched_child_prompt_matches_depth_capability(
    tmp_path, monkeypatch, legacy_role, parent_depth, max_depth, enabled,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parent = MagicMock()
    parent._delegate_depth = parent_depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._session_db = None
    parent.enabled_toolsets = ["terminal", "file", "delegation"]
    parent.disabled_toolsets = []
    parent.request_overrides = {}
    parent.provider = "custom"
    parent.base_url = "https://fixture.invalid/v1"
    parent.api_key = "fixture-only"
    parent.api_mode = "chat_completions"
    parent.model = "fixture"
    parent.platform = "cli"
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    child = MagicMock()
    child._credential_pool = None
    child._delegate_saved_tool_names = []
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.model = "fixture"
    child.run_conversation.return_value = {
        "final_response": "done", "completed": True, "api_calls": 1, "messages": [],
    }
    config = {"max_spawn_depth": max_depth, "orchestrator_enabled": enabled}
    args = {"tasks": [{"goal": "Inspect fixture"}], "background": False}
    if legacy_role is not None:
        args["role"] = legacy_role
    with (
        patch.object(delegate_tool, "_load_config", return_value=config),
        patch.object(delegate_tool, "_resolve_workspace_hint", return_value=str(tmp_path)),
        patch("run_agent.AIAgent", return_value=child) as agent_class,
    ):
        result = json.loads(registry.get_entry("delegate_task").handler(args, parent_agent=parent))

    child.run_conversation.assert_called_once()
    assert "error" not in result
    kwargs = agent_class.call_args.kwargs
    prompt = kwargs["ephemeral_system_prompt"]
    depth = parent_depth + 1
    can_delegate = enabled and depth < max_depth
    assert child._delegate_depth == depth
    assert child._delegate_role == ("orchestrator" if can_delegate else "leaf")
    assert ("delegation" in kwargs["enabled_toolsets"]) == can_delegate
    assert ("CAN spawn your own subagents" in prompt) == can_delegate
    # Neither sibling note may teach a nonexistent model-controlled role knob.
    assert "role=" not in prompt
    assert "`role`" not in prompt
    assert "Default is 'leaf'" not in prompt
    if can_delegate:
        assert "automatically" in prompt
        assert "depth" in prompt
        if depth + 1 >= max_depth:
            assert "children MUST be leaves" in prompt
        else:
            assert "children can themselves delegate because depth remains" in prompt
            assert "orchestrators or leaves" not in prompt


@pytest.mark.parametrize(
    "value,expected",
    [
        # Falsy scalars/containers an operator writes to disable spawning must
        # read as disabled, like every other delegation boolean.
        (0, False), (0.0, False), ([], False), ({}, False),
        # Explicit null is "unset" (the _knob convention), so the default applies.
        (None, True),
        # Strings and bools keep their existing coercion.
        ("0", False), ("false", False), ("off", False), ("yes", True),
        (False, False), (True, True), (1, True),
    ],
)
def test_orchestrator_enabled_falsy_scalars_disable_the_kill_switch(value, expected):
    """delegation.orchestrator_enabled: 0 must not silently enable child spawning."""
    with patch.object(delegate_tool, "_load_config", return_value={"orchestrator_enabled": value}):
        assert delegate_tool._get_orchestrator_enabled() is expected


@pytest.mark.parametrize("value,expected", [("false", False), ("0", False), ("off", False), (1, True), (True, True)])
def test_worktree_isolation_string_values_coerce_like_the_other_booleans(value, expected):
    """delegation.worktree_isolation: "false" (quoted YAML) must not enable isolation."""
    with patch.object(delegate_tool, "_load_config", return_value={"worktree_isolation": value}):
        assert delegate_tool._get_worktree_isolation() is expected


@pytest.mark.parametrize("value,expected_delegating", [(0, False), ("false", False), (True, True)])
def test_tool_description_matches_the_kill_switch(value, expected_delegating):
    """The delegate_task schema text must not advertise nested delegation when the
    operator switched orchestration off, whatever scalar type they used."""
    with patch.object(
        delegate_tool, "_load_config",
        return_value={"max_spawn_depth": 5, "orchestrator_enabled": value},
    ):
        description = delegate_tool._build_top_level_description()
    assert ("Children can themselves delegate" in description) is expected_delegating


@pytest.mark.parametrize("value,expected_open", [("false", False), ("0", False), (True, True)])
def test_worktree_isolation_gate_reads_the_config_value(value, expected_open):
    """_create_isolated_worktree must not open its gate for a quoted "false"."""
    from tools import subagent_worktree
    from tools.delegate_tool_child_run import _create_isolated_worktree

    with (
        patch.object(delegate_tool, "_load_config", return_value={"worktree_isolation": value}),
        patch.object(subagent_worktree, "local_backend_active", return_value=False) as backend_probe,
    ):
        assert _create_isolated_worktree(MagicMock(), "task-1", "sa-1") is None
    assert backend_probe.called is expected_open
