from __future__ import annotations

from workstation.memory import ProceduralMemory
from workstation.journal import ExecutionJournal
from workstation.routines import (
    DeterministicRoutineRunner,
    RoutineExecutionStatus,
    RoutinePromotionService,
)


def test_procedure_requires_validation_before_deterministic_replay(tmp_path):
    memory = ProceduralMemory(storage_path=tmp_path / "procedures.json")
    procedure = memory.record_success(
        "example.com",
        "search catalog",
        [{
            "action": "click",
            "target": "#search",
            "fallback_anchors": [{"type": "testid", "value": "catalog-search"}],
            "required": True,
        }],
    )
    journal = ExecutionJournal("task-routine", "session-routine", file_path=tmp_path / "journal.jsonl")
    runner = DeterministicRoutineRunner(memory, journal)
    blocked = runner.run(procedure.id, {"elements": []}, lambda *_: None)
    assert blocked.status == RoutineExecutionStatus.BLOCKED

    promotion = RoutinePromotionService(memory)
    promotion.validate(procedure.id, evidence=[{"kind": "assertion", "summary": "search works"}])
    promotion.promote(procedure.id)

    calls: list[tuple[str, str]] = []
    result = runner.run(
        procedure.id,
        {"elements": [{"ref": "node-search", "attributes": {"data-testid": "catalog-search"}}]},
        lambda step, target, _context: calls.append((step.action, target)),
    )
    assert result.status == RoutineExecutionStatus.COMPLETED
    assert calls == [("click", "node-search")]
    assert journal.read_events()[-1].metadata["procedure_id"] == procedure.id


def test_drift_fails_closed_and_records_failure(tmp_path):
    memory = ProceduralMemory(storage_path=tmp_path / "procedures.json")
    procedure = memory.record_success(
        "example.com",
        "submit form",
        [{"action": "click", "target": "#submit", "required": True}],
    )
    promotion = RoutinePromotionService(memory)
    promotion.validate(procedure.id, evidence=[{"kind": "test", "summary": "validated"}])
    promotion.promote(procedure.id)

    result = DeterministicRoutineRunner(memory).run(
        procedure.id,
        {"elements": []},
        lambda *_: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    assert result.status == RoutineExecutionStatus.DRIFT
    assert memory.get_procedure(procedure.id).failure_count == 1
