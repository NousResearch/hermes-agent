"""Focused tests for targeted live subagent control."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tools import delegate_tool as dt


class Controller:
    pass


@pytest.fixture(autouse=True)
def _clear_registry():
    with dt._active_subagents_lock:
        dt._active_subagents.clear()
    yield
    with dt._active_subagents_lock:
        dt._active_subagents.clear()


def _seed(subagent_id: str, controller, *, goal: str = "task"):
    agent = MagicMock()
    agent.steer.return_value = True
    dt._register_subagent(
        {
            "agent": agent,
            "root_agent": controller,
            "subagent_id": subagent_id,
            "parent_id": None,
            "goal": goal,
            "status": "running",
            "depth": 1,
            "started_at": 1.0,
            "tool_count": 2,
        },
    )
    return agent


def test_status_is_scoped_to_callers_delegation_tree_and_redacts_agents():
    mine = Controller()
    theirs = Controller()
    _seed("subagent-mine", mine)
    _seed("subagent-theirs", theirs)

    visible = dt.list_active_subagents(controller=mine)

    assert [item["subagent_id"] for item in visible] == ["subagent-mine"]
    assert visible[0]["tool_count"] == 2
    assert "agent" not in visible[0]
    assert "root_agent" not in visible[0]


def test_steer_targets_exactly_one_child():
    controller = Controller()
    target = _seed("subagent-target", controller)
    sibling = _seed("subagent-sibling", controller)

    assert dt.steer_subagent("subagent-target", "focus on tests", controller=controller)

    target.steer.assert_called_once_with("focus on tests")
    sibling.steer.assert_not_called()


def test_steer_rejects_blank_instruction_and_foreign_controller():
    owner = Controller()
    stranger = Controller()
    target = _seed("subagent-target", owner)

    assert not dt.steer_subagent("subagent-target", "   ", controller=owner)
    assert not dt.steer_subagent("subagent-target", "new goal", controller=stranger)
    target.steer.assert_not_called()


def test_steer_rejects_missing_or_failing_child_method():
    controller = Controller()
    missing = _seed("subagent-missing", controller)
    missing.steer = None
    failing = _seed("subagent-failing", controller)
    failing.steer.side_effect = RuntimeError("child already closed")

    assert not dt.steer_subagent("subagent-missing", "new goal", controller=controller)
    assert not dt.steer_subagent("subagent-failing", "new goal", controller=controller)


def test_unscoped_operator_control_can_target_explicit_child():
    controller = Controller()
    target = _seed("subagent-target", controller)

    assert dt.steer_subagent("subagent-target", "operator instruction")
    assert dt.interrupt_subagent("subagent-target")
    target.steer.assert_called_once_with("operator instruction")
    target.interrupt.assert_called_once()


def test_steer_does_not_hold_registry_lock_while_calling_child():
    controller = Controller()
    target = _seed("subagent-target", controller)

    def steer_side_effect(_instruction):
        done = threading.Event()
        worker = threading.Thread(
            target=lambda: (dt._unregister_subagent("subagent-target"), done.set())
        )
        worker.start()
        assert done.wait(1), "registry lock was held while child.steer() ran"
        worker.join(timeout=1)
        return True

    target.steer.side_effect = steer_side_effect

    assert dt.steer_subagent("subagent-target", "new direction", controller=controller)
    assert dt.list_active_subagents(controller=controller) == []


def test_interrupt_targets_one_child_and_respects_ownership():
    owner = Controller()
    stranger = Controller()
    target = _seed("subagent-target", owner)
    sibling = _seed("subagent-sibling", owner)

    assert not dt.interrupt_subagent("subagent-target", controller=stranger)
    assert dt.interrupt_subagent("subagent-target", controller=owner)

    target.interrupt.assert_called_once()
    sibling.interrupt.assert_not_called()


def test_orchestrator_can_control_siblings_in_same_top_level_tree():
    """The authorization boundary is the caller's top-level delegation tree."""
    root = Controller()
    orchestrator = Controller()
    setattr(orchestrator, "_delegation_root_agent", root)
    sibling = _seed("subagent-sibling", root)

    visible = dt.list_active_subagents(controller=orchestrator)

    assert [item["subagent_id"] for item in visible] == ["subagent-sibling"]
    assert dt.interrupt_subagent("subagent-sibling", controller=orchestrator)
    sibling.interrupt.assert_called_once_with(
        "Interrupted by controller (subagent-sibling)"
    )


def test_delegate_task_control_status_steer_and_interrupt():
    controller = Controller()
    child = _seed("subagent-one", controller)

    status = json.loads(dt.delegate_task(action="status", parent_agent=controller))
    assert status["count"] == 1
    assert status["active"][0]["subagent_id"] == "subagent-one"

    steered = json.loads(
        dt.delegate_task(
            action="steer",
            subagent_id="subagent-one",
            instruction="produce a minimal patch",
            parent_agent=controller,
        )
    )
    assert steered == {
        "action": "steer",
        "accepted": True,
        "subagent_id": "subagent-one",
    }
    child.steer.assert_called_once_with("produce a minimal patch")

    interrupted = json.loads(
        dt.delegate_task(
            action="interrupt",
            subagent_id="subagent-one",
            parent_agent=controller,
        )
    )
    assert interrupted["accepted"] is True
    child.interrupt.assert_called_once()


def test_delegate_task_steer_reaches_selected_child_next_model_iteration():
    controller = Controller()
    child = object.__new__(AIAgent)
    setattr(child, "_pending_steer", None)
    setattr(child, "_pending_steer_lock", threading.Lock())
    dt._register_subagent({
        "agent": child,
        "root_agent": controller,
        "subagent_id": "subagent-real-steer",
        "parent_id": None,
        "goal": "arbitrary task",
        "status": "running",
        "depth": 1,
        "started_at": 1.0,
        "tool_count": 0,
        "last_tool": None,
    })

    result = json.loads(
        dt.delegate_task(
            action="steer",
            subagent_id="subagent-real-steer",
            instruction="change direction now",
            parent_agent=controller,
        )
    )
    messages = [
        {"role": "tool", "content": "original result", "tool_call_id": "call-1"}
    ]
    child._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)

    assert result["accepted"] is True
    assert "original result" in messages[0]["content"]
    assert "change direction now" in messages[0]["content"]
    assert child._pending_steer is None


def test_model_dispatch_forwards_live_control_arguments():
    parent = object.__new__(AIAgent)
    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return "{}"

    with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
        AIAgent._dispatch_delegate_task(
            parent,
            {
                "action": "steer",
                "subagent_id": "root-a:4",
                "instruction": "use the new API",
            },
        )

    assert captured["action"] == "steer"
    assert captured["subagent_id"] == "root-a:4"
    assert captured["instruction"] == "use the new API"
    assert captured["parent_agent"] is parent


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"action": "unknown"}, "action must be one of"),
        ({"action": "steer"}, "subagent_id is required"),
        (
            {"action": "steer", "subagent_id": "subagent-one", "instruction": " "},
            "instruction is required",
        ),
        (
            {"action": "interrupt", "subagent_id": "missing"},
            "No controllable live subagent",
        ),
    ],
)
def test_delegate_task_control_rejects_malformed_requests(kwargs, needle):
    result = dt.delegate_task(parent_agent=Controller(), **kwargs)
    assert needle in result


def test_delegate_task_schema_exposes_live_controls():
    props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert props["action"]["enum"] == ["spawn", "status", "steer", "interrupt"]
    assert "subagent_id" in props
    assert "instruction" in props
