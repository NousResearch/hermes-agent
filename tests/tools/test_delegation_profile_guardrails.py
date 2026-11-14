"""Tests for profile delegation guardrails:
1. Profile memory bootstrap: the target's MEMORY.md/USER.md load without re-enabling context files or providers.
2. Session attribution in the child's init model config (delegated_profile and delegate_identity).
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


def test_profile_route_loads_profile_memory_without_context_or_provider_side_effects(isolated_home):
    """A profile child gets the target profile's MEMORY.md/USER.md (not the caller's) while keeping
    skip_context_files=True (project context already rides the ephemeral prompt once) and the old
    skip_memory value (no external memory provider starts as a side effect)."""
    from tools.delegate_tool import _build_child_agent
    from tools.memory_tool_store import ENTRY_DELIMITER

    (isolated_home / "memories").mkdir(exist_ok=True)
    (isolated_home / "memories" / "MEMORY.md").write_text("CALLER-ONLY NOTE", encoding="utf-8")
    profile_dir = isolated_home / "profiles" / "specialist"
    (profile_dir / "memories").mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("model:\n  default: target-model\n", encoding="utf-8")
    (profile_dir / "memories" / "MEMORY.md").write_text(
        ENTRY_DELIMITER.join(["Specialist owns the ledger schema", "Ledger lives in ops/ledger"]), encoding="utf-8")
    (profile_dir / "memories" / "USER.md").write_text("Owner prefers terse reports", encoding="utf-8")

    captured = {}

    def mock_agent(**kwargs):
        captured.clear()
        captured.update(kwargs)
        child = MagicMock()
        child.session_id = "child-sess"
        child._memory_store = None  # what _init_memory leaves on a memory_agent_context="subagent" child
        return child

    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=mock_agent):
        child = _build_child_agent(
            task_index=0, goal="specialist work", context="ctx", toolsets=None, model="test-model",
            max_iterations=10, task_count=1, parent_agent=parent, profile_home=profile_dir,
        )
    assert captured["skip_context_files"] is True
    assert captured["skip_memory"] is True
    assert captured["load_soul_identity"] is True
    memory_block = child._memory_store.format_for_system_prompt("memory")
    assert "Specialist owns the ledger schema" in memory_block and "Ledger lives in ops/ledger" in memory_block
    assert "CALLER-ONLY NOTE" not in memory_block
    assert "Owner prefers terse reports" in child._memory_store.format_for_system_prompt("user")
    assert (child._memory_enabled, child._user_profile_enabled) == (True, True)

    with patch("run_agent.AIAgent", side_effect=mock_agent):
        child = _build_child_agent(
            task_index=0, goal="ordinary work", context="ctx", toolsets=None, model="test-model",
            max_iterations=10, task_count=1, parent_agent=parent, profile_home=None,
        )
    assert (captured["skip_context_files"], captured["skip_memory"]) == (True, True)
    assert "load_soul_identity" not in captured
    assert child._memory_store is None


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


def test_default_profile_route_is_attributed_as_default(isolated_home):
    """The default profile's home is the Hermes root, whose basename (``.hermes``) is not its id."""
    from tools.delegate_tool import _build_child_agent

    child = MagicMock()
    child.session_id, child._session_init_model_config = "default-child", {}
    with patch("run_agent.AIAgent", return_value=child):
        _build_child_agent(
            task_index=0, goal="default route", context=None, toolsets=None, model="test-model",
            max_iterations=5, task_count=1, parent_agent=_parent(), profile_home=isolated_home,
        )
    assert child._delegated_profile == "default"
    assert child._session_init_model_config["delegate_identity"] == "profile:default"


def test_profile_delegation_monotonic_ceiling_cannot_be_enlarged_by_target(tmp_path, monkeypatch):
    """Security finding F4: Restrictive commissioner (depth: 1, orchestrator: false) delegating
    to a profile whose config has max_spawn_depth: 3 and orchestrator_enabled: true remains a leaf."""
    from tools.delegate_tool import _build_child_agent

    profile_dir = tmp_path / "profiles" / "permissive_worker"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text(
        yaml.safe_dump({"delegation": {"max_spawn_depth": 3, "orchestrator_enabled": True}}),
        encoding="utf-8",
    )

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    from tools.delegate_tool import _get_max_spawn_depth, _get_orchestrator_enabled
    from tools.delegate_tool_routes import delegated_profile_scope
    with delegated_profile_scope(profile_dir):  # control: the real readers see the permissive target
        assert (_get_max_spawn_depth(), _get_orchestrator_enabled()) == (3, True)
    parent = _parent(max_spawn_depth=1, orchestrator_enabled=False)

    def mock_agent(**kwargs):
        child = MagicMock()
        child.session_id = "child-sess"
        return child

    with patch("run_agent.AIAgent", side_effect=mock_agent):
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


def test_stamped_ceiling_refuses_a_spawn_the_live_config_would_allow(isolated_home, monkeypatch):
    """delegate_task mins the live max_spawn_depth (3) with the child's stamped ceiling (1): a depth-1
    child is refused even though the config alone would admit it."""
    import json
    from tools.delegate_tool import _get_max_spawn_depth, delegate_task

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (isolated_home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "parent-model"}, "delegation": {"max_spawn_depth": 3},
    }))
    assert _get_max_spawn_depth() == 3  # control: without the stamp, depth 1 < 3 would spawn
    res = json.loads(delegate_task(
        tasks=[{"goal": "nested task past the stamped ceiling"}],
        parent_agent=_parent(depth=1, max_spawn_depth=1),
    ))
    assert "Delegation depth limit reached (depth=1, max_spawn_depth=1)" in res["error"]


def test_target_profile_narrows_using_real_config(tmp_path, isolated_home, monkeypatch):
    """Independent review regression: target limits are read from real profile config."""
    from tools.delegate_tool import _build_child_agent

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (isolated_home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "parent-model"},
        "delegation": {"max_spawn_depth": 3, "orchestrator_enabled": True},
    }))
    target = isolated_home / "profiles" / "leaf_worker"
    target.mkdir(parents=True)
    (target / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "target-model"},
        "delegation": {"max_spawn_depth": 1, "orchestrator_enabled": False},
    }))
    from tools.delegate_tool import _get_max_spawn_depth, _get_orchestrator_enabled
    from tools.delegate_tool_routes import delegated_profile_scope
    assert (_get_max_spawn_depth(), _get_orchestrator_enabled()) == (3, True)
    with delegated_profile_scope(target):
        assert (_get_max_spawn_depth(), _get_orchestrator_enabled()) == (1, False)
    parent = _parent(max_spawn_depth=3, orchestrator_enabled=True)
    parent._session_db = None
    child = MagicMock()
    child.session_id = "review-child"
    child._session_init_model_config = {}
    with patch("run_agent.AIAgent", return_value=child):
        _build_child_agent(
            task_index=0, goal="review target narrowing", context=None,
            toolsets=None, model="test-model", max_iterations=5,
            task_count=1, parent_agent=parent, profile_home=target,
        )
    assert child._delegate_max_spawn_depth == 1
    assert child._delegate_orchestrator_enabled is False
    assert child._delegate_role == "leaf"

