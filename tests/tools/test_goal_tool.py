import json

from hermes_cli import goals


def test_set_goal_tool_sets_goal_and_subgoals_for_task_session(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    from tools.goal_tool import set_goal

    try:
        raw = set_goal(
            goal="Ship the goal bridge",
            subgoals=["Expose an agent-callable tool", "Persist subgoals"],
            task_id="sid-tool-goal",
        )
        result = json.loads(raw)

        assert result["success"] is True
        assert result["goal"] == "Ship the goal bridge"
        assert result["subgoals"] == ["Expose an agent-callable tool", "Persist subgoals"]
        state = goals.GoalManager("sid-tool-goal").state
        assert state is not None
        assert state.status == "active"
        assert state.goal == "Ship the goal bridge"
        assert state.subgoals == ["Expose an agent-callable tool", "Persist subgoals"]
    finally:
        goals._DB_CACHE.clear()


def test_set_goal_tool_uses_session_context_when_task_id_missing(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.goal_tool import set_goal

    tokens = set_session_vars(session_id="sid-context-goal")
    try:
        raw = set_goal(goal="Use context session", task_id=None)
        result = json.loads(raw)

        assert result["success"] is True
        state = goals.GoalManager("sid-context-goal").state
        assert state is not None
        assert state.goal == "Use context session"
    finally:
        clear_session_vars(tokens)
        goals._DB_CACHE.clear()


def test_set_goal_tool_requires_session_id(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()

    from tools.goal_tool import set_goal

    try:
        raw = set_goal(goal="No session", task_id=None)
        result = json.loads(raw)

        assert result["success"] is False
        assert "session" in result["error"].lower()
    finally:
        goals._DB_CACHE.clear()
