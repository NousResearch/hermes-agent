"""Validated deterministic replay over the canonical procedural memory owner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from workstation.contracts import ExecutionEventKind
from workstation.journal import ExecutionJournal
from workstation.memory import ProcedureLifecycle, ProceduralMemory, ProcedureStep


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RoutineExecutionStatus(str, Enum):
    BLOCKED = "blocked"
    DRIFT = "drift"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class RoutineExecutionResult:
    status: RoutineExecutionStatus
    procedure_id: str
    version: int | None = None
    completed_steps: int = 0
    reason: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    finished_at: str = field(default_factory=_utc_now)


class RoutinePromotionService:
    def __init__(self, memory: ProceduralMemory) -> None:
        self.memory = memory

    def validate(
        self,
        procedure_id: str,
        *,
        evidence: list[dict[str, Any]],
        validator: Callable[[Any], bool] | None = None,
    ) -> Any:
        procedure = self.memory.get_procedure(procedure_id)
        if procedure is None:
            raise KeyError(procedure_id)
        if not evidence:
            raise ValueError("validation evidence is required")
        if validator is not None and not validator(procedure):
            raise ValueError("procedure validation failed")
        procedure.lifecycle = ProcedureLifecycle.VALIDATED
        procedure.validation_evidence = list(evidence)
        procedure.validated_at = _utc_now()
        return self.memory.update_procedure(procedure)

    def promote(self, procedure_id: str) -> Any:
        procedure = self.memory.get_procedure(procedure_id)
        if procedure is None:
            raise KeyError(procedure_id)
        if procedure.lifecycle != ProcedureLifecycle.VALIDATED:
            raise ValueError("procedure must be validated before promotion")
        procedure.lifecycle = ProcedureLifecycle.PROMOTED
        procedure.promoted_at = _utc_now()
        return self.memory.update_procedure(procedure)


class DeterministicRoutineRunner:
    def __init__(self, memory: ProceduralMemory, journal: ExecutionJournal | None = None) -> None:
        self.memory = memory
        self.journal = journal

    def run(
        self,
        procedure_id: str,
        context: dict[str, Any],
        execute_step: Callable[[ProcedureStep, str, dict[str, Any]], Any],
        *,
        check_precondition: Callable[[str, dict[str, Any]], bool] | None = None,
        check_postcondition: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> RoutineExecutionResult:
        procedure = self.memory.get_procedure(procedure_id)
        if procedure is None:
            self._record(ExecutionEventKind.ERROR, "routine blocked: procedure not found", procedure_id)
            return RoutineExecutionResult(RoutineExecutionStatus.BLOCKED, procedure_id, reason="procedure not found")
        if procedure.lifecycle != ProcedureLifecycle.PROMOTED:
            self._record(ExecutionEventKind.ERROR, "routine blocked: procedure is not promoted", procedure_id)
            return RoutineExecutionResult(
                RoutineExecutionStatus.BLOCKED,
                procedure_id,
                version=procedure.version,
                reason="procedure is not promoted",
            )

        for precondition in procedure.preconditions:
            if check_precondition is not None and not check_precondition(precondition, context):
                return self._drift(procedure, f"precondition failed: {precondition}")

        elements = context.get("elements")
        completed = 0
        for step in procedure.steps:
            target = step.target
            if step.action not in {"navigate", "wait"} and step.target:
                if elements is None or not elements:
                    return self._drift(procedure, f"no live element for required anchor: {step.target}", completed)
                target = step.resolve_anchor(elements)
                if not target:
                    return self._drift(procedure, f"anchor drifted: {step.target}", completed)
            try:
                execute_step(step, target, context)
            except Exception as exc:
                self.memory.record_failure(procedure.id, completed, str(exc))
                self._record(ExecutionEventKind.ERROR, f"routine failed: {exc}", procedure.id, completed_steps=completed)
                return RoutineExecutionResult(
                    RoutineExecutionStatus.FAILED,
                    procedure.id,
                    version=procedure.version,
                    completed_steps=completed,
                    reason=str(exc),
                )
            completed += 1

        for postcondition in procedure.postconditions:
            if check_postcondition is not None and not check_postcondition(postcondition, context):
                return self._drift(procedure, f"postcondition failed: {postcondition}", completed)
        result = RoutineExecutionResult(
            RoutineExecutionStatus.COMPLETED,
            procedure.id,
            version=procedure.version,
            completed_steps=completed,
        )
        self._record(
            ExecutionEventKind.TASK_COMPLETED,
            "deterministic routine completed",
            procedure.id,
            version=procedure.version,
            completed_steps=completed,
        )
        return result

    def _drift(self, procedure: Any, reason: str, completed: int = 0) -> RoutineExecutionResult:
        self.memory.record_failure(procedure.id, completed, reason)
        self._record(ExecutionEventKind.ERROR, f"routine drift: {reason}", procedure.id, completed_steps=completed)
        return RoutineExecutionResult(
            RoutineExecutionStatus.DRIFT,
            procedure.id,
            version=procedure.version,
            completed_steps=completed,
            reason=reason,
        )

    def _record(self, kind: ExecutionEventKind, message: str, procedure_id: str, **metadata: Any) -> None:
        if self.journal is not None:
            self.journal.record(kind, message, metadata={"procedure_id": procedure_id, **metadata})
