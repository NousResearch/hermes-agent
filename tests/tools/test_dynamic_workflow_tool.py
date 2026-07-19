"""Tests for the dynamic_workflow tool."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import dynamic_workflow_tool as dwt


@pytest.fixture(autouse=True)
def _clean_workflows():
    dwt._reset_for_tests()
    yield
    dwt._reset_for_tests()


def _call(args, parent_agent=None):
    return json.loads(dwt.handle_dynamic_workflow(args, parent_agent=parent_agent))


def _parent(session_id):
    return SimpleNamespace(session_id=session_id)


def test_create_validates_dag_and_reports_ready_nodes():
    result = _call(
        {
            "action": "create",
            "workflow_id": "wf_demo",
            "objective": "Research a release",
            "nodes": [
                {
                    "node_id": "source_triage",
                    "phase_id": "investigate",
                    "phase_title": "Investigate",
                    "title": "Source triage",
                    "goal": "Find sources",
                },
                {
                    "node_id": "synthesis",
                    "phase": "Synthesize",
                    "goal": "Synthesize source findings",
                    "depends_on": ["source_triage"],
                },
            ],
        }
    )

    workflow = result["workflow"]
    assert workflow["workflow_id"] == "wf_demo"
    assert workflow["status"] == "ready"
    assert workflow["ready_node_ids"] == ["source_triage"]
    assert workflow["nodes"][0]["phase_id"] == "investigate"
    assert workflow["nodes"][0]["phase_title"] == "Investigate"
    assert workflow["nodes"][0]["title"] == "Source triage"
    assert workflow["nodes"][1]["phase_id"] == "synthesize"
    assert workflow["nodes"][1]["phase_title"] == "Synthesize"
    assert [node["node_id"] for node in workflow["nodes"]] == [
        "source_triage",
        "synthesis",
    ]


def test_create_rejects_cycles():
    result = _call(
        {
            "action": "create",
            "workflow_id": "wf_cycle",
            "objective": "Bad graph",
            "nodes": [
                {"node_id": "a", "goal": "A", "depends_on": ["b"]},
                {"node_id": "b", "goal": "B", "depends_on": ["a"]},
            ],
        }
    )

    assert "error" in result
    assert "cycle detected" in result["error"]


def test_workflows_are_scoped_to_parent_session():
    parent_a = _parent("session-a")
    parent_b = _parent("session-b")

    created_a = _call(
        {
            "action": "create",
            "workflow_id": "wf_shared",
            "objective": "Session A workflow",
            "nodes": [{"node_id": "a", "goal": "A"}],
        },
        parent_agent=parent_a,
    )
    assert created_a["workflow"]["objective"] == "Session A workflow"

    missing_from_b = _call(
        {"action": "status", "workflow_id": "wf_shared"},
        parent_agent=parent_b,
    )
    assert missing_from_b["error"] == "unknown workflow_id: wf_shared"

    created_b = _call(
        {
            "action": "create",
            "workflow_id": "wf_shared",
            "objective": "Session B workflow",
            "nodes": [{"node_id": "b", "goal": "B"}],
        },
        parent_agent=parent_b,
    )
    assert created_b["workflow"]["objective"] == "Session B workflow"

    status_a = _call({"action": "status"}, parent_agent=parent_a)
    status_b = _call({"action": "status"}, parent_agent=parent_b)

    assert [wf["objective"] for wf in status_a["workflows"]] == ["Session A workflow"]
    assert [wf["objective"] for wf in status_b["workflows"]] == ["Session B workflow"]


def test_dispatch_ready_uses_delegate_task_background(monkeypatch):
    calls = []

    def fake_delegate_task(**kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "status": "dispatched",
                "delegation_id": f"deleg_{len(calls)}",
                "subagent_id": f"sa_{len(calls)}",
                "child_session_id": f"sess_{len(calls)}",
            }
        )

    from tools import delegate_tool

    monkeypatch.setattr(delegate_tool, "delegate_task", fake_delegate_task)
    parent = object()

    result = _call(
        {
            "action": "create",
            "workflow_id": "wf_dispatch",
            "objective": "Parallel source review",
            "context": "Use public docs only.",
            "dispatch_ready": True,
            "nodes": [
                {
                    "node_id": "web",
                    "phase_id": "investigate",
                    "phase_title": "Investigate",
                    "title": "Web source review",
                    "goal": "Search web sources",
                    "toolsets": ["web"],
                },
                {
                    "node_id": "files",
                    "phase_id": "investigate",
                    "phase_title": "Investigate",
                    "title": "Local note review",
                    "goal": "Inspect local notes",
                    "toolsets": ["file"],
                },
            ],
        },
        parent_agent=parent,
    )

    assert result["dispatched"] == [
        {
            "node_id": "web",
            "delegation_id": "deleg_1",
            "subagent_id": "sa_1",
            "child_session_id": "sess_1",
        },
        {
            "node_id": "files",
            "delegation_id": "deleg_2",
            "subagent_id": "sa_2",
            "child_session_id": "sess_2",
        },
    ]
    assert [call["background"] for call in calls] == [True, True]
    assert [call["parent_agent"] for call in calls] == [parent, parent]
    assert "workflow_id: wf_dispatch" in calls[0]["context"]
    assert "node_id: web" in calls[0]["context"]
    assert "phase: Investigate" in calls[0]["context"]
    assert "task_title: Web source review" in calls[0]["context"]
    assert calls[0]["toolsets"] == ["web"]
    assert calls[0]["_observability_context"] == {
        "workflow_id": "wf_dispatch",
        "workflow_node_id": "web",
        "workflow_phase_id": "investigate",
        "workflow_phase_title": "Investigate",
        "workflow_task_title": "Web source review",
        "workflow_objective": "Parallel source review",
        "task_prompt": "Search web sources",
        "task_context": "",
    }
    assert result["workflow"]["nodes"][0]["status"] == "dispatched"
    assert result["workflow"]["nodes"][0]["subagent_id"] == "sa_1"


def test_record_result_then_model_can_extend_graph_with_dependent_node():
    _call(
        {
            "action": "create",
            "workflow_id": "wf_extend",
            "objective": "Reassess after source triage",
            "nodes": [{"node_id": "source_triage", "goal": "Find source gaps"}],
        }
    )

    recorded = _call(
        {
            "action": "record_result",
            "workflow_id": "wf_extend",
            "node_id": "source_triage",
            "status": "completed",
            "summary": "Missing pricing source; add a targeted search.",
            "result": {"missing": ["pricing"]},
        }
    )
    assert recorded["workflow"]["status"] == "completed"

    extended = _call(
        {
            "action": "add_nodes",
            "workflow_id": "wf_extend",
            "nodes": [
                {
                    "node_id": "pricing_gap",
                    "goal": "Search only for pricing evidence",
                    "depends_on": ["source_triage"],
                }
            ],
        }
    )

    assert extended["workflow"]["status"] == "ready"
    assert extended["workflow"]["ready_node_ids"] == ["pricing_gap"]
    assert [node["node_id"] for node in extended["workflow"]["nodes"]] == [
        "source_triage",
        "pricing_gap",
    ]


def test_record_result_cannot_complete_node_before_dependencies():
    _call(
        {
            "action": "create",
            "workflow_id": "wf_deps",
            "objective": "Respect readiness",
            "nodes": [
                {"node_id": "first", "goal": "First"},
                {"node_id": "second", "goal": "Second", "depends_on": ["first"]},
            ],
        }
    )

    early = _call(
        {
            "action": "record_result",
            "workflow_id": "wf_deps",
            "node_id": "second",
            "status": "completed",
            "summary": "too early",
        }
    )
    assert early["error"] == "node second cannot complete before its dependencies complete"

    _call(
        {
            "action": "record_result",
            "workflow_id": "wf_deps",
            "node_id": "first",
            "status": "completed",
            "summary": "first done",
        }
    )
    later = _call(
        {
            "action": "record_result",
            "workflow_id": "wf_deps",
            "node_id": "second",
            "status": "completed",
            "summary": "now ready",
        }
    )
    assert later["workflow"]["status"] == "completed"


def test_status_reconciles_completed_async_workflow_node(monkeypatch):
    def fake_delegate_task(**kwargs):
        return json.dumps({"status": "dispatched", "delegation_id": "deleg_done"})

    def fake_delegations():
        return [
            {
                "delegation_id": "deleg_done",
                "status": "completed",
                "summary": "Worker finished cleanly.",
                "completed_at": 123.0,
                "duration_seconds": 4.5,
                "api_calls": 2,
                "input_tokens": 100,
                "output_tokens": 25,
                "reasoning_tokens": 10,
                "cost_usd": 0.01,
                "model": "test-model",
            }
        ]

    from tools import delegate_tool
    import tools.async_delegation as ad

    monkeypatch.setattr(delegate_tool, "delegate_task", fake_delegate_task)
    monkeypatch.setattr(ad, "list_async_delegations", fake_delegations)

    _call(
        {
            "action": "create",
            "workflow_id": "wf_reconcile",
            "objective": "Auto reconcile",
            "dispatch_ready": True,
            "nodes": [{"node_id": "worker", "goal": "Do work"}],
        },
        parent_agent=object(),
    )

    status = _call({"action": "status", "workflow_id": "wf_reconcile"})

    node = status["workflow"]["nodes"][0]
    assert status["workflow"]["status"] == "completed"
    assert node["status"] == "completed"
    assert node["summary"] == "Worker finished cleanly."
    assert node["completed_at"] == 123.0
    assert node["duration_seconds"] == 4.5
    assert node["api_calls"] == 2
    assert node["input_tokens"] == 100
    assert node["output_tokens"] == 25
    assert node["reasoning_tokens"] == 10
    assert node["cost_usd"] == 0.01
    assert node["model"] == "test-model"
    assert node["async_completion_reconciled"] is True
    assert (
        dwt.is_async_completion_reconciled(
            "deleg_done", workflow_id="wf_reconcile", node_id="worker"
        )
        is True
    )


def test_dispatch_ready_unlocks_dependents_after_async_reconcile(monkeypatch):
    dispatches = []

    def fake_delegate_task(**kwargs):
        dispatches.append(kwargs["goal"])
        return json.dumps(
            {"status": "dispatched", "delegation_id": f"deleg_{len(dispatches)}"}
        )

    def fake_delegations():
        return [
            {
                "delegation_id": "deleg_1",
                "status": "completed",
                "summary": "First worker done.",
            }
        ]

    from tools import delegate_tool
    import tools.async_delegation as ad

    monkeypatch.setattr(delegate_tool, "delegate_task", fake_delegate_task)
    monkeypatch.setattr(ad, "list_async_delegations", fake_delegations)

    _call(
        {
            "action": "create",
            "workflow_id": "wf_unlock",
            "objective": "Unlock dependents",
            "dispatch_ready": True,
            "max_dispatch": 1,
            "nodes": [
                {"node_id": "first", "goal": "First"},
                {"node_id": "second", "goal": "Second", "depends_on": ["first"]},
            ],
        },
        parent_agent=object(),
    )

    dispatched = _call(
        {
            "action": "dispatch_ready",
            "workflow_id": "wf_unlock",
            "max_dispatch": 1,
        },
        parent_agent=object(),
    )

    assert dispatches == ["First", "Second"]
    assert dispatched["dispatched"] == [
        {"node_id": "second", "delegation_id": "deleg_2"}
    ]
    assert [node["status"] for node in dispatched["workflow"]["nodes"]] == [
        "completed",
        "dispatched",
    ]


def test_cancel_interrupts_dispatched_workflow_children(monkeypatch):
    def fake_delegate_task(**kwargs):
        return json.dumps({"status": "dispatched", "delegation_id": "deleg_a"})

    interrupted = []

    def fake_interrupt(delegation_id, reason="cancelled"):
        interrupted.append((delegation_id, reason))
        return True

    from tools import delegate_tool
    import tools.async_delegation as ad

    monkeypatch.setattr(delegate_tool, "delegate_task", fake_delegate_task)
    monkeypatch.setattr(ad, "interrupt_delegation", fake_interrupt)

    _call(
        {
            "action": "create",
            "workflow_id": "wf_cancel",
            "objective": "Cancelable",
            "dispatch_ready": True,
            "nodes": [{"node_id": "worker", "goal": "Do work"}],
        },
        parent_agent=object(),
    )

    cancelled = _call(
        {"action": "cancel", "workflow_id": "wf_cancel", "interrupt": True}
    )

    assert cancelled["interrupted_delegation_ids"] == ["deleg_a"]
    assert interrupted == [("deleg_a", "dynamic_workflow cancelled")]
    node = cancelled["workflow"]["nodes"][0]
    assert node["status"] == "dispatched"
    assert node["cancel_requested"] is True
