import json

from hermes_cli.goals import GoalContract, GoalState
from tools.goal_control_tool import GOAL_CONTROL_SCHEMA, _available, goal_control


class _Manager:
    state = None

    def __init__(self, session_id, default_max_turns=20):
        self.session_id = session_id
        self.default_max_turns = default_max_turns

    def is_active(self):
        return self.state is not None and self.state.status == "active"

    def has_goal(self):
        return self.state is not None and self.state.status in {"active", "paused"}

    def status_line(self):
        return "active" if self.has_goal() else "empty"

    def render_contract(self):
        if self.state is None:
            return "(no goal)"
        return self.state.contract.render_block() or "(no completion contract)"

    def set(self, objective, max_turns=None, contract=None):
        type(self).state = GoalState(
            goal=objective,
            max_turns=max_turns or self.default_max_turns,
            contract=contract or GoalContract(),
        )
        return type(self).state


def _setup(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "session-1")
    monkeypatch.setattr("hermes_cli.goals.GoalManager", _Manager)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"goals": {"max_turns": 120, "model_tool_enabled": True}},
    )
    _Manager.state = None


def test_model_tool_is_opt_in(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "session-1")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"goals": {}})
    assert _available() is False

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"goals": {"model_tool_enabled": True}},
    )
    assert _available() is True


def test_start_persists_one_goal(monkeypatch):
    _setup(monkeypatch)

    first = json.loads(goal_control({"action": "start", "objective": "perfect the project"}))
    second = json.loads(goal_control({"action": "start", "objective": "replace it"}))

    assert first["started"] is True
    assert first["max_turns"] == 120
    assert second["started"] is False
    assert _Manager.state.goal == "perfect the project"


def test_start_persists_completion_contract_and_preserves_it(monkeypatch):
    _setup(monkeypatch)
    contract = {
        "outcome": "Project is production-ready",
        "verification": "Focused tests and build pass",
        "constraints": "Keep the public API stable",
        "boundaries": "Only edit this repository",
        "stop_when": "A paid service or account approval is required",
    }

    first = json.loads(
        goal_control({"action": "start", "objective": "perfect the project", "contract": contract})
    )
    second = json.loads(
        goal_control(
            {
                "action": "start",
                "objective": "replace it",
                "contract": {"outcome": "different"},
            }
        )
    )

    assert first["started"] is True
    assert first["contract_attached"] is True
    assert "Focused tests and build pass" in first["contract"]
    assert second["started"] is False
    assert _Manager.state.goal == "perfect the project"
    assert _Manager.state.contract.to_dict() == contract


def test_start_without_contract_keeps_legacy_behavior_and_status_is_read_only(monkeypatch):
    _setup(monkeypatch)

    started = json.loads(goal_control({"action": "start", "objective": "finish it"}))
    before = _Manager.state
    status = json.loads(goal_control({"action": "status"}))

    assert started["contract_attached"] is False
    assert _Manager.state.contract.is_empty()
    assert status["active"] is True
    assert status["contract"] == "(no completion contract)"
    assert _Manager.state is before


def test_contract_schema_rejects_unknown_fields():
    parameters = GOAL_CONTROL_SCHEMA["parameters"]
    contract = parameters["properties"]["contract"]

    assert parameters["additionalProperties"] is False
    assert contract["additionalProperties"] is False
    assert set(contract["properties"]) == {
        "outcome",
        "verification",
        "constraints",
        "boundaries",
        "stop_when",
    }


def test_start_requires_session_and_objective(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    assert "error" in json.loads(goal_control({"action": "start", "objective": "x"}))

    monkeypatch.setenv("HERMES_SESSION_ID", "session-1")
    assert "error" in json.loads(goal_control({"action": "start"}))
