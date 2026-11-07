"""Routed children budget workspace context using their own window (#108891).

The model-map fixture uses the public delegation provider contract. Routing, child
construction, metadata resolution and context-file rendering use the real core.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

from agent.delegation_provider import DelegationProvider


class _ModelMap(DelegationProvider):
    name = "model_map"

    def task_parameters(self, config):
        return {"option": {"type": "string", "enum": ["worker"]}}

    def route_task(self, task, config):
        return self.settings(config)["options"][task["option"]]


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    # The marker is in the middle, outside both retained ends of a 128K-window cap.
    contents = "a" * 50_000 + "\nMIDDLE_WORKSPACE_RULE\n" + "z" * 50_000
    (workspace / "AGENTS.md").write_text(contents, encoding="utf-8")
    return workspace


def _config(home, parent_window, child_window, **extra):
    home.mkdir(parents=True, exist_ok=True)
    config = {
        "model": {"default": "parent-model", "provider": "custom",
                  "base_url": "http://127.0.0.1:9/v1", "context_length": parent_window},
        "model_overrides": {"custom": {"child-model": {"context_window": child_window}}},
        "delegation": {"backend": "model_map", "model": "not-the-routed-model",
                       "backends": {"model_map": {"options": {"worker": {
                           "model": "child-model", "provider": "custom",
                           "base_url": "http://127.0.0.1:9/v1", "api_key": "test-key",
                       }}}}},
        **extra,
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return config


def _parent(workspace):
    from run_agent import AIAgent

    parent = AIAgent(model="parent-model", provider="custom", api_key="test-key",
                     base_url="http://127.0.0.1:9/v1", api_mode="chat_completions",
                     platform="cli", quiet_mode=True, skip_context_files=True,
                     skip_memory=True, save_trajectories=False, enabled_toolsets=["file"], cwd=str(workspace))
    parent.cwd = str(workspace)
    return parent


@pytest.mark.parametrize("parent_window, child_window", [(128_000, 1_000_000), (1_000_000, 128_000)])
def test_model_map_child_context_uses_its_resolved_window(tmp_path, monkeypatch, parent_window, child_window):
    from agent.delegation_registry import _registry
    from tools import delegate_tool as dt
    from tools.delegate_tool_routes import resolve_task_routes

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = _config(home, parent_window, child_window)
    parent = _parent(_workspace(tmp_path))
    provider = _ModelMap()
    previous = _registry.get_provider("model_map")
    _registry.register(provider)
    child = None
    try:
        tasks = [{"goal": "Read workspace rules", "option": "worker"}]
        routes, error = resolve_task_routes(tasks, config["delegation"], parent)
        assert error is None and routes[0].error is None
        children, error = dt._build_children(
            tasks, [None], routes, top_role="leaf", max_iterations=4,
            parent_agent=parent, live_deleg_id=None, live_writers=[],
        )
        assert error is None
        child = children[0][2]
        assert child.model == "child-model"
        assert child.context_compressor.context_length == child_window
        assert parent.context_compressor.context_length == parent_window
        prompt = child.ephemeral_system_prompt
        assert ("MIDDLE_WORKSPACE_RULE" in prompt) == (child_window == 1_000_000)
        assert ("truncated AGENTS.md" in prompt) == (child_window == 128_000)
        assert len(prompt) > 100_000 if child_window == 1_000_000 else 20_000 < len(prompt) < 40_000
    finally:
        if child is not None:
            child.close()
        parent.close()
        _registry.restore_registration("model_map", provider, previous)


@pytest.mark.parametrize("window", [None, 0, -1, True, "1000000"])
def test_unresolved_child_window_keeps_floor_not_parent_window(tmp_path, monkeypatch, window):
    from tests.tools.test_delegate import _make_mock_parent
    from tools.delegate_tool import _build_child_agent

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _config(home, 1_000_000, 1_000_000)
    parent = _make_mock_parent()
    parent.cwd = str(_workspace(tmp_path))
    parent.context_compressor = SimpleNamespace(context_length=1_000_000)
    child = MagicMock(context_compressor=SimpleNamespace(context_length=window))

    def construct(**kwargs):
        child.ephemeral_system_prompt = kwargs["ephemeral_system_prompt"]
        return child

    with patch("run_agent.AIAgent", side_effect=construct):
        _build_child_agent(task_index=0, goal="Read rules", context=None, toolsets=["file"],
                           model=None, max_iterations=4, task_count=1, parent_agent=parent)
    prompt = child.ephemeral_system_prompt
    assert "MIDDLE_WORKSPACE_RULE" not in prompt and "truncated AGENTS.md" in prompt
    assert len(prompt) < 20_000


def test_profile_context_cap_is_read_in_child_scope(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from tools.delegate_tool import _build_child_agent, _route_overrides
    from tools.delegate_tool_routes import _route_to_task_route

    home = tmp_path / "os-home" / ".hermes"
    monkeypatch.setenv("HOME", str(home.parent))
    monkeypatch.setenv("HERMES_HOME", str(home))
    _config(home, 1_000_000, 1_000_000, context_file_max_chars=20_000)
    child_home = home / "profiles" / "worker"
    _config(child_home, 1_000_000, 1_000_000, context_file_max_chars=150_000)
    parent = _parent(_workspace(tmp_path))
    child = None
    try:
        route = _route_to_task_route({"profile": "worker"}, parent, {})
        child = _build_child_agent(task_index=0, goal="Read rules", context=None, toolsets=["file"],
                                   model=route.creds["model"], max_iterations=4, task_count=1,
                                   parent_agent=parent, **_route_overrides(route))
        assert "MIDDLE_WORKSPACE_RULE" in child.ephemeral_system_prompt
        assert "truncated AGENTS.md" not in child.ephemeral_system_prompt
        assert get_hermes_home() == home
    finally:
        if child is not None:
            child.close()
        parent.close()
