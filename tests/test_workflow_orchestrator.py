import json
from dataclasses import dataclass
from typing import Any

import pytest

from agent.workflow_orchestrator import (
    WorkflowOrchestrator,
    _extract_json_object,
    _non_thinking_model,
)


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


class _Agent:
    provider = "custom"
    model = "kr/claude-opus-4.8-thinking"
    base_url = "http://homeserverkenny:20128/v1"
    api_key = "test-key"


def _response(content: str) -> _Response:
    return _Response(choices=[_Choice(message=_Message(content=content))])


def _planner_payload(*, mode: str = "parallel", subtasks: list[dict[str, Any]]) -> str:
    return json.dumps({"mode": mode, "rationale": "test plan", "subtasks": subtasks})


def test_extract_json_object_accepts_markdown_fences_and_prose():
    payload = _extract_json_object('Sure:\n```json\n{"mode":"parallel","subtasks":[]}\n```')

    assert payload == {"mode": "parallel", "subtasks": []}


@pytest.mark.parametrize(
    ("input_model", "expected"),
    [
        ("kr/claude-opus-4.8-thinking-agentic", "kr/claude-opus-4.8"),
        ("kr/claude-opus-4.8-thinking", "kr/claude-opus-4.8"),
        ("cx/gpt-5.5-xhigh", "cx/gpt-5.5"),
        ("cx/gpt-5.5-high", "cx/gpt-5.5"),
        ("cx/gpt-5.5-low", "cx/gpt-5.5"),
        ("kr/claude-opus-4.8", "kr/claude-opus-4.8"),
    ],
)
def test_non_thinking_model_strips_9router_effort_suffixes(input_model, expected):
    assert _non_thinking_model(input_model) == expected


def test_one_subtask_plan_returns_inline_fallback_without_delegate_or_synthesis():
    calls: list[dict[str, Any]] = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        return _response(_planner_payload(subtasks=[{"goal": "Answer directly", "context": "No fan-out needed."}]))

    def fake_delegate(**kwargs):  # pragma: no cover - should never run
        raise AssertionError("delegate_task should not run for one-subtask plans")

    orchestrator = WorkflowOrchestrator(
        _Agent(),
        call_llm_fn=fake_llm,
        delegate_fn=fake_delegate,
        max_children_fn=lambda: 3,
    )

    result = orchestrator.run("what time is it?")

    assert result.delegated is False
    assert result.final_response == ""
    assert [task.goal for task in result.plan.subtasks] == ["Answer directly"]
    assert len(calls) == 1
    assert calls[0]["tools"] is None
    assert calls[0]["model"] == "kr/claude-opus-4.8"


def test_parallel_plan_runs_delegate_in_waves_capped_by_max_children_then_synthesizes():
    llm_calls: list[dict[str, Any]] = []
    delegate_calls: list[list[dict[str, Any]]] = []
    subtasks = [
        {"goal": "Task A", "context": "A context"},
        {"goal": "Task B", "context": "B context"},
        {"goal": "Task C", "context": "C context"},
        {"goal": "Task D", "context": "D context"},
    ]

    def fake_llm(**kwargs):
        llm_calls.append(kwargs)
        if len(llm_calls) == 1:
            return _response(_planner_payload(subtasks=subtasks))
        return _response("Final synthesized answer")

    def fake_delegate(*, tasks, parent_agent):
        assert parent_agent is _agent
        delegate_calls.append(tasks)
        return json.dumps(
            {
                "results": [
                    {"task_index": idx, "status": "completed", "summary": task["goal"]}
                    for idx, task in enumerate(tasks)
                ],
                "total_duration_seconds": 1.25,
            }
        )

    _agent = _Agent()
    orchestrator = WorkflowOrchestrator(
        _agent,
        call_llm_fn=fake_llm,
        delegate_fn=fake_delegate,
        max_children_fn=lambda: 3,
    )

    result = orchestrator.run("big task")

    assert result.delegated is True
    assert result.final_response == "Final synthesized answer"
    assert [len(wave) for wave in delegate_calls] == [3, 1]
    assert [[task["goal"] for task in wave] for wave in delegate_calls] == [
        ["Task A", "Task B", "Task C"],
        ["Task D"],
    ]
    assert [entry["task_index"] for entry in result.child_results] == [0, 1, 2, 3]
    assert [entry["wave"] for entry in result.child_results] == [1, 1, 1, 2]
    assert result.total_duration_seconds == 2.5
    assert len(llm_calls) == 2
    assert llm_calls[0]["tools"] is None
    assert llm_calls[1]["tools"] is None


def test_sequential_plan_runs_one_task_at_a_time_and_passes_prior_summary_forward():
    llm_calls: list[dict[str, Any]] = []
    delegate_calls: list[list[dict[str, Any]]] = []
    subtasks = [
        {"goal": "Implement middleware", "context": "Create code."},
        {"goal": "Write tests", "context": "Test final API."},
        {"goal": "Write docs", "context": "Document final API."},
    ]

    def fake_llm(**kwargs):
        llm_calls.append(kwargs)
        if len(llm_calls) == 1:
            return _response(_planner_payload(mode="sequential", subtasks=subtasks))
        return _response("Sequential synthesis")

    def fake_delegate(*, tasks, parent_agent):
        delegate_calls.append(tasks)
        summary = f"finished {tasks[0]['goal']}"
        return json.dumps(
            {
                "results": [{"task_index": 0, "status": "completed", "summary": summary}],
                "total_duration_seconds": 0.5,
            }
        )

    orchestrator = WorkflowOrchestrator(
        _Agent(),
        call_llm_fn=fake_llm,
        delegate_fn=fake_delegate,
        max_children_fn=lambda: 3,
    )

    result = orchestrator.run("dependent task")

    assert result.final_response == "Sequential synthesis"
    assert [len(wave) for wave in delegate_calls] == [1, 1, 1]
    assert "Previous workflow results" not in delegate_calls[0][0]["context"]
    assert "finished Implement middleware" in delegate_calls[1][0]["context"]
    assert "finished Write tests" in delegate_calls[2][0]["context"]
    assert [entry["task_index"] for entry in result.child_results] == [0, 1, 2]
    assert [entry["wave"] for entry in result.child_results] == [1, 2, 3]
    assert result.total_duration_seconds == 1.5


def test_delegate_error_raises_runtime_error():
    def fake_llm(**kwargs):
        return _response(
            _planner_payload(
                subtasks=[
                    {"goal": "Task A", "context": "A"},
                    {"goal": "Task B", "context": "B"},
                ]
            )
        )

    def fake_delegate(*, tasks, parent_agent):
        return json.dumps({"error": "Too many tasks"})

    orchestrator = WorkflowOrchestrator(
        _Agent(),
        call_llm_fn=fake_llm,
        delegate_fn=fake_delegate,
        max_children_fn=lambda: 3,
    )

    with pytest.raises(RuntimeError, match="Too many tasks"):
        orchestrator.run("bad task")
