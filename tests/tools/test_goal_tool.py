"""Tests for the model-callable set_goal tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so goal state never touches the real session DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class TestSetGoalTool:
    def test_sets_goal_for_current_session(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from tools.goal_tool import set_goal_tool

        result = json.loads(
            set_goal_tool(
                goal="finish the pull request",
                session_id="tool-session",
                max_turns=7,
            )
        )

        assert result["success"] is True
        assert result["action"] == "set"
        assert result["goal"] == "finish the pull request"
        assert result["status"] == "active"
        assert result["max_turns"] == 7
        assert "standing goal" in result["message"].lower()

        mgr = GoalManager(session_id="tool-session")
        assert mgr.is_active()
        assert mgr.state.goal == "finish the pull request"
        assert mgr.state.max_turns == 7

    def test_omitted_max_turns_uses_supplied_default(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from tools.goal_tool import set_goal_tool

        result = json.loads(
            set_goal_tool(
                goal="use the configured budget",
                session_id="default-budget-session",
                default_max_turns=4,
            )
        )

        assert result["success"] is True
        assert result["max_turns"] == 4
        assert GoalManager(session_id="default-budget-session").state.max_turns == 4

    def test_rejects_max_turns_above_configured_budget(self, hermes_home):
        from tools.goal_tool import set_goal_tool

        result = json.loads(
            set_goal_tool(
                goal="do not overrun",
                session_id="budget-cap-session",
                max_turns=99,
                default_max_turns=5,
            )
        )

        assert result["success"] is False
        assert "exceeds" in result["error"].lower()

    def test_rejects_non_integral_float_max_turns(self, hermes_home):
        from tools.goal_tool import set_goal_tool

        result = json.loads(
            set_goal_tool(
                goal="do not truncate floats",
                session_id="float-budget-session",
                max_turns=3.7,
                default_max_turns=5,
            )
        )

        assert result["success"] is False
        assert "positive integer" in result["error"].lower()

    def test_reports_error_when_goal_state_does_not_persist(self, hermes_home, monkeypatch):
        from hermes_cli import goals
        from tools.goal_tool import set_goal_tool

        monkeypatch.setattr(goals, "save_goal", lambda session_id, state: False)

        result = json.loads(
            set_goal_tool(
                goal="must persist to be useful",
                session_id="missing-persist-session",
                default_max_turns=5,
            )
        )

        assert result["success"] is False
        assert "persist" in result["error"].lower()

    def test_reports_error_when_stale_same_goal_row_remains_after_save_failure(
        self, hermes_home, monkeypatch
    ):
        from hermes_cli import goals
        from tools.goal_tool import set_goal_tool

        old = goals.GoalState(
            goal="same visible goal",
            status="active",
            turns_used=0,
            max_turns=5,
            created_at=1.0,
            updated_at=1.0,
        )
        goals.save_goal("stale-same-goal-session", old)
        monkeypatch.setattr(goals, "save_goal", lambda session_id, state: False)

        result = json.loads(
            set_goal_tool(
                goal="same visible goal",
                session_id="stale-same-goal-session",
                default_max_turns=5,
            )
        )

        assert result["success"] is False
        assert "persist" in result["error"].lower()

    def test_rejects_non_string_goal(self, hermes_home):
        from tools.goal_tool import set_goal_tool

        result = json.loads(set_goal_tool(goal=["bad"], session_id="tool-session"))

        assert result["success"] is False
        assert "string" in result["error"].lower()

    def test_existing_goal_manager_observes_tool_set_goal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from tools.goal_tool import set_goal_tool

        mgr = GoalManager(session_id="cached-session", default_max_turns=6)
        assert not mgr.is_active()

        result = json.loads(
            set_goal_tool(
                goal="wake cached manager",
                session_id="cached-session",
                default_max_turns=6,
            )
        )

        assert result["success"] is True
        assert mgr.state.goal == "wake cached manager"
        assert mgr.state.max_turns == 6

    def test_refresh_preserves_in_memory_goal_when_db_load_fails(self, hermes_home, monkeypatch):
        from hermes_cli import goals

        mgr = goals.GoalManager(session_id="transient-db-failure", default_max_turns=6)
        mgr.set("survive transient db failure")
        monkeypatch.setattr(goals, "load_goal", lambda session_id: None)

        assert mgr.is_active()
        assert mgr.state.goal == "survive transient db failure"

    def test_refresh_preserves_newer_in_memory_goal_after_save_failure(self, hermes_home, monkeypatch):
        from hermes_cli import goals

        mgr = goals.GoalManager(session_id="stale-row-after-save-failure", default_max_turns=6)
        mgr.set("old persisted goal")
        monkeypatch.setattr(goals, "save_goal", lambda session_id, state: False)

        mgr.set("new in-memory goal")

        assert mgr.state.goal == "new in-memory goal"
        assert mgr.is_active()

    def test_dirty_manager_observes_later_external_clear(self, hermes_home, monkeypatch):
        from hermes_cli import goals

        original_save_goal = goals.save_goal
        mgr = goals.GoalManager(session_id="dirty-manager-external-clear", default_max_turns=6)
        mgr.set("shared goal")
        monkeypatch.setattr(goals, "save_goal", lambda session_id, state: False)
        mgr.pause("local save failed")
        assert mgr.state.status == "paused"
        mgr._dirty_since = 1.0

        monkeypatch.setattr(goals, "save_goal", original_save_goal)
        goals.GoalManager(session_id="dirty-manager-external-clear", default_max_turns=6).clear()

        assert mgr.state is None
        assert not mgr.has_goal()

    def test_requires_active_session(self, hermes_home):
        from tools.goal_tool import set_goal_tool

        result = json.loads(set_goal_tool(goal="do the thing", session_id=""))

        assert result["success"] is False
        assert "active session" in result["error"].lower()

    def test_rejects_empty_goal(self, hermes_home):
        from tools.goal_tool import set_goal_tool

        result = json.loads(set_goal_tool(goal="   ", session_id="tool-session"))

        assert result["success"] is False
        assert "empty" in result["error"].lower()


class TestSetGoalToolSchema:
    def test_registered_in_goal_toolset_not_core_default(self):
        from model_tools import get_tool_definitions
        from toolsets import _HERMES_CORE_TOOLS

        assert "set_goal" not in _HERMES_CORE_TOOLS

        goal_names = {
            tool["function"]["name"]
            for tool in get_tool_definitions(enabled_toolsets=["goal"], quiet_mode=True)
        }
        assert "set_goal" in goal_names

        cli_names = {
            tool["function"]["name"]
            for tool in get_tool_definitions(enabled_toolsets=["hermes-cli"], quiet_mode=True)
        }
        assert "set_goal" not in cli_names

    def test_registry_dispatch_accepts_session_id(self):
        """set_goal is registry-backed; generic fallback forwards session_id."""
        from model_tools import _AGENT_LOOP_TOOLS, handle_function_call
        from hermes_cli.goals import GoalManager

        assert "set_goal" not in _AGENT_LOOP_TOOLS

        # Without session_id the tool itself fails closed.
        bare = json.loads(handle_function_call("set_goal", {"goal": "x"}))
        assert bare.get("success") is False
        assert "session" in bare.get("error", "").lower()


class TestSetGoalLiveAIAgentPath:
    def test_invoke_tool_sets_goal_via_registry_fallback(self, hermes_home, monkeypatch):
        """End-to-end: AIAgent._invoke_tool -> runtime helpers -> handle_function_call."""
        from hermes_cli.goals import GoalManager, load_goal
        from run_agent import AIAgent

        # Ensure discovery has registered set_goal.
        import tools.goal_tool  # noqa: F401

        agent = object.__new__(AIAgent)
        agent.session_id = "live-aiagent-set-goal"
        agent.valid_tool_names = {"set_goal"}
        agent.enabled_toolsets = ["goal"]
        agent.disabled_toolsets = None
        agent._current_turn_id = "turn-1"
        agent._current_api_request_id = "req-1"
        agent._todo_store = None
        agent._memory_store = None
        agent._memory_manager = None
        agent.clarify_callback = None
        agent.read_terminal_callback = None

        # Avoid plugin/middleware side effects that require a fully-built agent.
        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *a, **k: None,
        )

        result = json.loads(
            agent._invoke_tool(
                "set_goal",
                {"goal": "live path works", "max_turns": 3},
                "task-live",
                tool_call_id="tc-set-goal",
            )
        )

        assert result["success"] is True
        assert result["goal"] == "live path works"
        assert result["max_turns"] == 3

        persisted = load_goal("live-aiagent-set-goal")
        assert persisted is not None
        assert persisted.goal == "live path works"
        assert GoalManager(session_id="live-aiagent-set-goal").is_active()
