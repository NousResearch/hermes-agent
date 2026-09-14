from __future__ import annotations

import pytest

from workstation.journal import ExecutionJournal
from workstation.memory import MemoryKind, ProceduralMemory
from workstation.replay import PortableTrace, PortableTraceBuilder, ReplayEngine
from workstation.contracts import ExecutionEventKind


def test_typed_memory_is_temporal_and_workspace_scoped(tmp_path):
    memory = ProceduralMemory(storage_path=tmp_path / "memory.json")
    record = memory.record_memory(
        MemoryKind.DECISION,
        "Use the staging endpoint",
        workspace_id="repo-a",
        source="task-1",
        created_at="2026-09-01T00:00:00+00:00",
        observed_at="2026-09-01T00:00:00+00:00",
        last_verified_at="2026-09-01T00:00:00+00:00",
    )
    assert record.kind == MemoryKind.DECISION
    assert memory.list_memory(workspace_id="repo-a")[0].content == "Use the staging endpoint"
    assert memory.list_memory(workspace_id="repo-b") == []
    reloaded = ProceduralMemory(storage_path=tmp_path / "memory.json")
    assert reloaded.get_memory(record.record_id) is not None

    snapshot = memory.snapshot(tmp_path / "memory.snapshot")
    memory.record_memory(MemoryKind.FACT, "temporary fact", workspace_id="repo-a")
    memory.restore_snapshot(snapshot)
    assert memory.list_memory(kind=MemoryKind.FACT, workspace_id="repo-a") == []


def test_memory_compaction_is_explicit_scoped_and_persistent(tmp_path):
    path = tmp_path / "memory.json"
    memory = ProceduralMemory(storage_path=path)
    procedure = memory.record_success("example.com", "keep procedure", name="durable procedure")
    for index in range(4):
        memory.record_memory(
            MemoryKind.TASK_CONTEXT,
            f"task-{index}",
            workspace_id="task-a",
            created_at=f"2026-09-01T00:00:0{index}+00:00",
            observed_at=f"2026-09-01T00:00:0{index}+00:00",
        )
    memory.record_memory(MemoryKind.FACT, "keep this fact", workspace_id="task-a")
    memory.record_memory(MemoryKind.TASK_CONTEXT, "other workspace", workspace_id="task-b")

    assert memory.compact_memory(max_records=2, workspace_id="task-a", kind=MemoryKind.TASK_CONTEXT) == 2
    remaining = memory.list_memory(include_stale=True)
    assert {record.content for record in remaining} >= {"task-2", "task-3", "keep this fact", "other workspace"}
    assert len([record for record in remaining if record.workspace_id == "task-a" and record.kind == MemoryKind.TASK_CONTEXT]) == 2

    reloaded = ProceduralMemory(storage_path=path)
    assert len([record for record in reloaded.list_memory(include_stale=True) if record.workspace_id == "task-a" and record.kind == MemoryKind.TASK_CONTEXT]) == 2
    assert reloaded.get_procedure(procedure.id) is not None

    stale_writer = ProceduralMemory(storage_path=path)
    compactor = ProceduralMemory(storage_path=path)
    assert compactor.compact_memory(max_records=1, workspace_id="task-a", kind=MemoryKind.TASK_CONTEXT) == 1
    stale_writer.record_memory(MemoryKind.EVIDENCE, "new evidence", workspace_id="task-a")
    after_stale_write = ProceduralMemory(storage_path=path)
    task_context = [
        record
        for record in after_stale_write.list_memory(include_stale=True)
        if record.workspace_id == "task-a" and record.kind == MemoryKind.TASK_CONTEXT
    ]
    assert len(task_context) == 1
    assert "new evidence" in {record.content for record in after_stale_write.list_memory(include_stale=True)}


def test_memory_compaction_rejects_negative_limit(tmp_path):
    memory = ProceduralMemory(storage_path=tmp_path / "memory.json")
    with pytest.raises(ValueError, match="non-negative"):
        memory.compact_memory(max_records=-1)


def test_portable_trace_redacts_sensitive_fields_and_replays_offline(tmp_path):
    journal = ExecutionJournal("task-replay", "session-replay", file_path=tmp_path / "journal.jsonl")
    journal.record(
        ExecutionEventKind.NAVIGATION,
        "Opened account",
        url="https://example.com/account?access_token=secret-value",
        metadata={"password": "never-export", "step": 1},
    )
    trace = PortableTraceBuilder.from_journal(
        journal,
        model_id="model-a",
        provider="provider-a",
        policy_version="policy-1",
    )
    serialized = trace.to_dict()
    assert "secret-value" not in str(serialized)
    assert "never-export" not in str(serialized)

    observed: list[str] = []
    report = ReplayEngine().replay(trace, lambda event: observed.append(event["kind"]))
    assert report.replayed_events == 1
    assert report.side_effects_performed is False
    assert observed == ["navigation"]

    fork = trace.fork_at(0, model_id="model-b", provider="provider-b")
    assert fork.parent_trace_id == trace.trace_id
    assert fork.model_id == "model-b"
    trace_path = tmp_path / "trace.json"
    trace.save(trace_path)
    assert PortableTrace.load(trace_path).trace_id == trace.trace_id
