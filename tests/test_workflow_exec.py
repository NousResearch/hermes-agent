"""Tests for workflow_exec.py — workflow parsing, validation, sorting, cycle detection, and parallel execution."""

import json
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.workflow_exec import (
    WorkflowExecutor,
    parse_workflow_graph,
    validate_workflow_structure,
    execute_workflow_tool,
)


def test_parse_workflow_graph():
    # Valid dict input
    g = {"nodes": [{"id": "1", "data": {"goal": "g1"}}], "edges": []}
    nodes, edges = parse_workflow_graph(g)
    assert len(nodes) == 1
    assert len(edges) == 0

    # Valid JSON string input
    nodes, edges = parse_workflow_graph(json.dumps(g))
    assert len(nodes) == 1
    assert len(edges) == 0

    # Invalid JSON string
    with pytest.raises(ValueError, match="Invalid JSON format"):
        parse_workflow_graph("{invalid}")

    # Fallback to list of nodes directly
    nodes, edges = parse_workflow_graph([{"id": "1", "data": {"goal": "g1"}}])
    assert len(nodes) == 1
    assert len(edges) == 0

    # None input
    nodes, edges = parse_workflow_graph(None)
    assert len(nodes) == 0
    assert len(edges) == 0


def test_validate_workflow_structure():
    # Valid structure
    nodes = [{"id": "1", "data": {"goal": "g1"}}]
    edges = []
    assert validate_workflow_structure(nodes, edges) is None

    # Invalid node type
    assert "must be an object" in validate_workflow_structure(["string_node"], [])

    # Missing id
    assert "missing a valid 'id'" in validate_workflow_structure([{"data": {"goal": "g1"}}], [])

    # Missing data
    assert "missing a valid 'data' object" in validate_workflow_structure([{"id": "1"}], [])

    # Missing goal
    assert "must have a non-empty string 'goal' or 'prompt'" in validate_workflow_structure([{"id": "1", "data": {}}], [])

    # Prompt is accepted as an alias for goal
    assert validate_workflow_structure([{"id": "1", "type": "agent", "data": {"prompt": "p1"}}], []) is None

    # Start and End control nodes do not require a goal
    assert validate_workflow_structure([
        {"id": "start", "type": "start", "data": {}},
        {"id": "end", "type": "end", "data": {}},
        {"id": "agent", "type": "agent", "data": {"goal": "g1"}}
    ], []) is None

    # Invalid edge references
    nodes = [{"id": "1", "data": {"goal": "g1"}}, {"id": "2", "data": {"goal": "g2"}}]
    edges = [{"source": "1", "target": "3"}]
    assert "references non-existent target node" in validate_workflow_structure(nodes, edges)


def test_cycle_detection():
    # DAG: A -> B -> C
    nodes = [
        {"id": "A", "data": {"goal": "task A"}},
        {"id": "B", "data": {"goal": "task B"}},
        {"id": "C", "data": {"goal": "task C"}},
    ]
    edges = [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
    ]
    executor = WorkflowExecutor(nodes, edges)
    assert not executor.has_cycle()

    # Cycle: A -> B -> C -> A
    edges.append({"source": "C", "target": "A"})
    executor = WorkflowExecutor(nodes, edges)
    assert executor.has_cycle()

    # Self loop: A -> A
    edges_self = [{"source": "A", "target": "A"}]
    executor_self = WorkflowExecutor(nodes, edges_self)
    assert executor_self.has_cycle()


@patch("tools.workflow_exec.delegate_task")
def test_successful_dag_execution(mock_delegate):
    # A -> B
    #   -> C
    # B and C are independent but run after A
    nodes = [
        {"id": "A", "data": {"goal": "task A"}},
        {"id": "B", "data": {"goal": "task B"}},
        {"id": "C", "data": {"goal": "task C"}},
    ]
    edges = [
        {"source": "A", "target": "B"},
        {"source": "A", "target": "C"},
    ]

    execution_order = []

    def side_effect(goal, **kwargs):
        execution_order.append(goal)
        return json.dumps({
            "results": [{
                "status": "completed",
                "summary": f"Done {goal}",
                "duration_seconds": 0.01,
                "api_calls": 1
            }]
        })

    mock_delegate.side_effect = side_effect

    parent_agent = MagicMock()
    parent_agent._interrupt_requested = False
    executor = WorkflowExecutor(nodes, edges, parent_agent=parent_agent)
    result = executor.execute()

    assert result["status"] == "completed"
    assert "A" in result["results"]
    assert "B" in result["results"]
    assert "C" in result["results"]
    assert result["results"]["A"]["status"] == "completed"
    assert result["results"]["B"]["status"] == "completed"
    assert result["results"]["C"]["status"] == "completed"

    # A must run before B and C
    assert execution_order[0] == "task A"
    assert set(execution_order[1:]) == {"task B", "task C"}


@patch("tools.workflow_exec.delegate_task")
def test_parallel_execution(mock_delegate):
    # Two independent nodes: A and B
    nodes = [
        {"id": "A", "data": {"goal": "sleep A"}},
        {"id": "B", "data": {"goal": "sleep B"}},
    ]
    edges = []

    def side_effect(goal, **kwargs):
        time.sleep(0.1)
        return json.dumps({
            "results": [{
                "status": "completed",
                "summary": f"Slept {goal}",
                "duration_seconds": 0.1,
                "api_calls": 1
            }]
        })

    mock_delegate.side_effect = side_effect

    parent_agent = MagicMock()
    parent_agent._interrupt_requested = False
    executor = WorkflowExecutor(nodes, edges, parent_agent=parent_agent, max_workers=2)

    start_time = time.monotonic()
    result = executor.execute()
    elapsed = time.monotonic() - start_time

    assert result["status"] == "completed"
    # If ran sequentially, elapsed would be >= 0.2s.
    # In parallel, it should be less than 0.18s.
    assert elapsed < 0.18


@patch("tools.workflow_exec.delegate_task")
def test_error_propagation_and_abort(mock_delegate):
    # A -> B -> C
    # D (independent)
    nodes = [
        {"id": "A", "data": {"goal": "fail A"}},
        {"id": "B", "data": {"goal": "task B"}},
        {"id": "C", "data": {"goal": "task C"}},
        {"id": "D", "data": {"goal": "task D"}},
    ]
    edges = [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
    ]

    def side_effect(goal, **kwargs):
        if "fail" in goal:
            return json.dumps({
                "results": [{
                    "status": "failed",
                    "error": "Task failed purposefully",
                    "summary": "Failure summary"
                }]
            })
        return json.dumps({
            "results": [{
                "status": "completed",
                "summary": f"Completed {goal}",
                "duration_seconds": 0.01,
                "api_calls": 1
            }]
        })

    mock_delegate.side_effect = side_effect

    parent_agent = MagicMock()
    parent_agent._interrupt_requested = False
    executor = WorkflowExecutor(nodes, edges, parent_agent=parent_agent)
    result = executor.execute()

    assert result["status"] == "failed"
    assert result["results"]["A"]["status"] == "failed"
    assert result["results"]["B"]["status"] == "aborted"
    assert result["results"]["C"]["status"] == "aborted"
    assert result["results"]["D"]["status"] == "completed"

    # B and C should not have been called at all
    called_goals = [
        args[0][0] if args[0] else args[1].get("goal")
        for args in mock_delegate.call_args_list
    ]
    assert "task B" not in called_goals
    assert "task C" not in called_goals
    assert "task D" in called_goals


@patch("tools.workflow_exec.delegate_task")
def test_interrupt_propagation(mock_delegate):
    # A -> B
    nodes = [
        {"id": "A", "data": {"goal": "sleep A"}},
        {"id": "B", "data": {"goal": "task B"}},
    ]
    edges = [{"source": "A", "target": "B"}]

    parent_agent = MagicMock()
    parent_agent._interrupt_requested = False

    def side_effect(goal, **kwargs):
        # Trigger parent agent interrupt while running A
        parent_agent._interrupt_requested = True
        time.sleep(0.05)
        return json.dumps({
            "results": [{
                "status": "completed",
                "summary": "Finished A before interrupt checked",
                "duration_seconds": 0.05
            }]
        })

    mock_delegate.side_effect = side_effect

    executor = WorkflowExecutor(nodes, edges, parent_agent=parent_agent)
    result = executor.execute()

    # The execution should be marked failed due to interrupt
    assert result["status"] == "failed"
    # B should be aborted due to interrupt check in main loop
    assert result["results"]["B"]["status"] == "aborted"
    assert "interrupted" in result["results"]["B"]["error"]


@patch("tools.workflow_exec.delegate_task")
@patch("run_agent.AIAgent")
def test_execute_workflow_graph_success(mock_ai_agent, mock_delegate):
    # Setup graph
    graph = {
        "nodes": [{"id": "1", "data": {"goal": "task 1"}}],
        "edges": []
    }
    
    mock_delegate.return_value = json.dumps({
        "results": [{
            "status": "completed",
            "summary": "Completed successfully",
            "duration_seconds": 0.01,
            "api_calls": 1
        }]
    })

    log_lines = []
    def log_callback(msg):
        log_lines.append(msg)

    # Instantiate mock agent instance returned by mock_ai_agent
    mock_agent_instance = MagicMock()
    mock_agent_instance._interrupt_requested = False
    mock_ai_agent.return_value = mock_agent_instance

    from tools.workflow_exec import execute_workflow_graph
    ok = execute_workflow_graph(graph, log_callback)

    assert ok is True
    assert any("[COMPLETED] Node 1: task 1" in line for line in log_lines)
    assert any("Result Summary: Completed successfully" in line for line in log_lines)
    mock_agent_instance.close.assert_called_once()
