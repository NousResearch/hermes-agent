"""Tests for profile delegation guardrails:
1. Profile memory & context bootstrap (skip_memory=False, skip_context_files=False when profile_home is set).
2. Session attribution in SQLite init model config (delegated_profile and delegate_identity).
3. Monotonic non-widening delegation ceilings (spawn depth and orchestrator role).
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    import pm.environments
    monkeypatch.setattr(pm.environments, "activate_dependencies", lambda *a, **kw: None)
    home = tmp_path / "os-home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text("model:\n  default: parent-model\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _parent(*, session_id="parent-sess", depth=0, max_spawn_depth=None, orchestrator_enabled=None):
    parent = MagicMock()
    parent.session_id = session_id
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "parent-model", "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db = MagicMock(db_path="/tmp/test_parent.db", reserved_session_id="child-sess-1")
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent._credential_pool = None
    parent.request_overrides = {}
    if max_spawn_depth is not None:
        parent._delegate_max_spawn_depth = max_spawn_depth
    if orchestrator_enabled is not None:
        parent._delegate_orchestrator_enabled = orchestrator_enabled
    return parent


def test_profile_route_bootstraps_memory_and_context_files(tmp_path):
    """When profile_home is set, the child AIAgent is built with skip_context_files=False and skip_memory=False.
    When profile_home is None, default subagent isolation (skip=True) is preserved."""
    from tools.delegate_tool import _build_child_agent

    profile_dir = tmp_path / "profiles" / "specialist"
    profile_dir.mkdir(parents=True)

    captured = {}

    def mock_agent(**kwargs):
        captured.update(kwargs)
        child = MagicMock()
        child.session_id = "child-sess"
        return child

    parent = _parent()

    # 1. Profiled child
    with patch("run_agent.AIAgent", side_effect=mock_agent):
        _build_child_agent(
            task_index=0,
            goal="specialist work",
            context="ctx",
            toolsets=None,
            model="test-model",
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            profile_home=profile_dir,
        )
    assert captured.get("skip_context_files") is False
    assert captured.get("skip_memory") is False
    assert captured.get("load_soul_identity") is True

    # 2. Non-profiled child
    captured.clear()
    with patch("run_agent.AIAgent", side_effect=mock_agent):
        _build_child_agent(
            task_index=0,
            goal="ordinary work",
            context="ctx",
            toolsets=None,
            model="test-model",
            max_iterations=10,
            task_count=1,
            parent_agent=parent,
            profile_home=None,
        )
    assert captured.get("skip_context_files") is True
    assert captured.get("skip_memory") is True
    assert "load_soul_identity" not in captured


def test_profile_route_persists_session_attribution(tmp_path):
    """Profile-routed child persists delegated_profile and delegate_identity onto _session_init_model_config."""
    from tools.delegate_tool import _build_child_agent

    profile_dir = tmp_path / "profiles" / "analyst"
    profile_dir.mkdir(parents=True)

    parent = _parent(session_id="commissioner-sess-99")

    def mock_agent(**kwargs):
        child = MagicMock()
        child.session_id = "analyst-sess-1"
        child._session_init_model_config = {}
        return child

    with patch("run_agent.AIAgent", side_effect=mock_agent):
        child = _build_child_agent(
            task_index=0,
            goal="data analysis",
            context=None,
            toolsets=None,
            model="test-model",
            max_iterations=5,
            task_count=1,
            parent_agent=parent,
            profile_home=profile_dir,
        )

    assert child._delegated_profile == "analyst"
    assert child._delegate_identity == "profile:analyst"
    assert child._session_init_model_config.get("delegated_profile") == "analyst"
    assert child._session_init_model_config.get("delegate_identity") == "profile:analyst"
    assert child._session_init_model_config.get("_delegate_from") == "commissioner-sess-99"


def test_profile_delegation_monotonic_ceiling_cannot_be_enlarged_by_target(tmp_path):
    """Security finding F4: Restrictive commissioner (depth: 1, orchestrator: false) delegating
    to a profile whose config has max_spawn_depth: 3 and orchestrator_enabled: true remains a leaf."""
    from tools.delegate_tool import _build_child_agent

    profile_dir = tmp_path / "profiles" / "permissive_worker"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text(
        yaml.safe_dump({"delegation": {"max_spawn_depth": 3, "orchestrator_enabled": True}}),
        encoding="utf-8",
    )

    parent = _parent(max_spawn_depth=1, orchestrator_enabled=False)

    def mock_agent(**kwargs):
        child = MagicMock()
        child.session_id = "child-sess"
        return child

    with (
        patch("run_agent.AIAgent", side_effect=mock_agent),
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=3),
        patch("tools.delegate_tool._get_orchestrator_enabled", return_value=True),
    ):
        child = _build_child_agent(
            task_index=0,
            goal="try escalating permissions",
            context=None,
            toolsets=None,
            model="test-model",
            max_iterations=5,
            task_count=1,
            parent_agent=parent,
            profile_home=profile_dir,
        )

    assert child._delegate_role == "leaf"
    assert child._delegate_max_spawn_depth == 1
    assert child._delegate_orchestrator_enabled is False


def test_profile_delegation_ceiling_narrowed_by_target(tmp_path):
    """Permissive commissioner (depth: 3) narrowed by restrictive target config (depth: 1) to leaf."""
    from tools.delegate_tool import _build_child_agent

    profile_dir = tmp_path / "profiles" / "leaf_worker"
    profile_dir.mkdir(parents=True)

    parent = _parent(max_spawn_depth=3, orchestrator_enabled=True)

    def mock_agent(**kwargs):
        child = MagicMock()
        child.session_id = "child-sess"
        return child

    with (
        patch("run_agent.AIAgent", side_effect=mock_agent),
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=1),
        patch("tools.delegate_tool._get_orchestrator_enabled", return_value=False),
    ):
        child = _build_child_agent(
            task_index=0,
            goal="restricted target",
            context=None,
            toolsets=None,
            model="test-model",
            max_iterations=5,
            task_count=1,
            parent_agent=parent,
            profile_home=profile_dir,
        )

    assert child._delegate_role == "leaf"
    assert child._delegate_max_spawn_depth == 1
    assert child._delegate_orchestrator_enabled is False


def test_profile_delegation_nested_ceiling_preservation():
    """Child delegating at depth >= ceiling is refused pre-spawn."""
    from tools.delegate_tool import delegate_task
    import json

    parent = _parent(depth=2, max_spawn_depth=2)

    res = json.loads(delegate_task(
        tasks=[{"goal": "nested task that exceeds ceiling"}],
        parent_agent=parent,
    ))
    assert "error" in res
    assert "Delegation depth limit reached" in res["error"]
