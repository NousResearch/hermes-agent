"""The parent-model delegate dispatch preserves public worker routing and controls."""

import dataclasses
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from agent.subagent_lifecycle import SubagentLifecycleService, SubagentState
from agent.worker_store import WorkerStore
from hermes_state import SessionDB


def test_parent_dispatch_forwards_worker_fields_and_keeps_safe_task_overrides():
    import run_agent

    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return "{}"

    parent = SimpleNamespace(_delegate_depth=0)
    args = {
        "action": "wait",
        "worker_id": "worker-fixture",
        "run_id": "run-fixture",
        "message": "continue",
        "timeout_seconds": 12,
        "profile": "review",
        "provider": "provider-b",
        "model": "review-model",
        "reasoning_effort": "high",
        "reconciliation_disposition": "accepted_unknown_no_replay",
        "tasks": [{
            "goal": "review",
            "profile": "review",
            "provider": "provider-b",
            "model": "review-model",
            "reasoning_effort": "high",
            "acp_command": "must-not-pass",
            "acp_args": ["--must-not-pass"],
        }],
    }
    with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
        result = run_agent.AIAgent._dispatch_delegate_task(parent, args)

    assert result == "{}"
    for field in (
        "action", "worker_id", "run_id", "message", "timeout_seconds",
        "profile", "provider", "model", "reasoning_effort",
        "reconciliation_disposition",
    ):
        assert captured[field] == args[field]
    task = captured["tasks"][0]
    assert task["profile"] == "review"
    assert task["provider"] == "provider-b"
    assert task["model"] == "review-model"
    assert task["reasoning_effort"] == "high"
    assert "acp_command" not in task and "acp_args" not in task
    assert captured["background"] is True


def test_parent_dispatch_reaches_compact_discovery_and_durable_controls(tmp_path, monkeypatch):
    import run_agent
    from tools import delegate_tool

    cfg = {
        "profiles": {
            "review": {
                "description": "bounded reviewer",
                "instructions": "private startup instruction",
                "provider": "fixture",
                "model": "review-model",
            },
        },
    }
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: cfg)
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)
    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(
        _delegate_depth=0,
        session_id="parent-dispatch",
        model="parent-model",
        _session_db=db,
    )
    compact = json.loads(run_agent.AIAgent._dispatch_delegate_task(parent, {"action": "discover"}))
    assert compact["profiles"][0]["name"] == "review"
    assert "instructions" not in compact["profiles"][0]
    detailed = json.loads(run_agent.AIAgent._dispatch_delegate_task(
        parent, {"action": "discover", "profile": "review"}))
    assert detailed["profiles"][0]["instructions"] == "private startup instruction"

    store = WorkerStore(db)
    store.ensure_schema()
    worker = store.create_worker(parent.session_id, profile="review")
    store.enqueue_run(worker["worker_id"], parent.session_id, goal="done")
    active = store.claim_next_run(worker["worker_id"], parent.session_id)
    store.finish_run(
        active["run_id"], parent.session_id, active["lease_token"],
        status="SUCCEEDED", result={"summary": "done"})
    status = json.loads(run_agent.AIAgent._dispatch_delegate_task(
        parent, {"action": "status", "worker_id": worker["worker_id"]}))
    assert status["success"] is True
    assert status["worker_id"] == worker["worker_id"]
    message = json.loads(run_agent.AIAgent._dispatch_delegate_task(
        parent, {"action": "message", "worker_id": worker["worker_id"], "message": "follow up"}))
    assert message["status"] == "PENDING"
    acknowledged = json.loads(run_agent.AIAgent._dispatch_delegate_task(
        parent, {"action": "ack", "worker_id": worker["worker_id"], "run_id": active["run_id"]}))
    assert acknowledged["acknowledged"] is True

    uncertain_worker = store.create_worker(parent.session_id, profile="review")
    uncertain = store.enqueue_run(uncertain_worker["worker_id"], parent.session_id, goal="external effect")
    uncertain = store.claim_run(uncertain["run_id"], parent.session_id)
    store.mark_tool_boundary(
        uncertain["run_id"], parent.session_id, uncertain["lease_token"],
        tool_call_id="external-effect")
    store.finish_run(
        uncertain["run_id"], parent.session_id, uncertain["lease_token"],
        status="INTERRUPTED", result={"summary": None})
    reconciled = json.loads(run_agent.AIAgent._dispatch_delegate_task(
        parent,
        {
            "action": "reconcile",
            "worker_id": uncertain_worker["worker_id"],
            "run_id": uncertain["run_id"],
            "reconciliation_disposition": "accepted_unknown_no_replay",
            "message": "Operator accepts the unknown outcome without replay.",
        },
    ))
    assert reconciled["reconciled"] is True
    audit = store.get_run(uncertain["run_id"], parent.session_id)["result"]["reconciliation"]
    assert audit["affected_tool_calls"] == [
        {"tool_call_id": "external-effect", "prior_status": "INFLIGHT"}
    ]
    inspected_reconciliation = json.loads(run_agent.AIAgent._dispatch_delegate_task(
        parent,
        {
            "action": "inspect",
            "worker_id": uncertain_worker["worker_id"],
            "run_id": uncertain["run_id"],
        },
    ))
    assert inspected_reconciliation["run"]["result"]["reconciliation"]["note"].startswith(
        "Operator accepts")
    db.close()


def test_opaque_runtime_metadata_stays_out_of_public_and_parent_json(tmp_path, monkeypatch):
    """Provider clients may own locks; receipts expose typed scalar metadata only."""
    import run_agent
    from tools import delegate_tool

    class OpaqueRuntimeMetadata:
        def __init__(self):
            self.lock = threading.Lock()

    cfg = {"max_concurrent_children": 1}
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: cfg)
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)
    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(
        _delegate_depth=0,
        session_id="parent-opaque",
        model="parent-model",
        _session_db=db,
    )
    child = SimpleNamespace(
        _subagent_id="sa-opaque",
        _delegate_role="leaf",
        _delegate_depth=1,
        provider=OpaqueRuntimeMetadata(),
        model=OpaqueRuntimeMetadata(),
        ephemeral_system_prompt="frozen prompt",
        valid_tool_names={"read_file"},
        _worker_route_receipt={},
    )
    service = SubagentLifecycleService(lambda: parent)
    handle = service.adopt_delegate_child(
        child,
        goal="serialize safely",
        context=None,
        profile=None,
        creds={},
        cfg=cfg,
    )
    assert handle is not None
    assert handle.provider is None and handle.model is None
    service.complete_adopted_child(
        child,
        {
            "status": "completed",
            "summary": "done",
            "exit_reason": "completed",
            "api_calls": 1,
            "duration_seconds": 0.01,
            "cost_status": "unknown",
        },
    )
    result = service.result(handle)
    assert result.terminal_state is SubagentState.SUCCEEDED
    json.dumps(handle.to_dict())
    json.dumps(dataclasses.asdict(result))

    inspected = json.loads(run_agent.AIAgent._dispatch_delegate_task(
        parent,
        {"action": "inspect", "worker_id": handle.worker_id, "run_id": handle.run_id},
    ))
    assert inspected["run"]["result"]["summary"] == "done"
    assert inspected["run"]["result"]["cost"] == {"status": "unknown", "usd": None}
    db.close()
