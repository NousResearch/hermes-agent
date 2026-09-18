"""Validated deterministic replay over the canonical procedural memory owner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Any, Callable

from workstation.contracts import ExecutionEventKind
from workstation.journal import ExecutionJournal
from workstation.memory import ProcedureLifecycle, ProceduralMemory, ProcedureStep


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def procedure_fingerprint(procedure):
    from workstation.recipes import digest
    return digest({k: v for k, v in procedure.to_dict().items() if k in {
        'id', 'version', 'scope', 'steps', 'preconditions', 'postconditions',
        'runtime_family', 'capability_fingerprint'}})


def semantic_browser_elements(observation):
    """Decode the native snapshot inventory only, excluding untrusted page text."""
    if isinstance(observation.get('elements'), list):
        return observation['elements'][:200]
    inventory = str(observation.get('snapshot', '')).partition('Interactive elements:\n')[2].partition('\nPage text:')[0]
    elements = []
    for line in inventory.splitlines()[:200]:
        match = re.fullmatch(r'- \[(@e\d+)\] (\S+)(?: ("(?:\\.|[^"\\])*"))?(?: value=".*")?(?: disabled)?', line)
        if match:
            try:
                label = json.loads(match[3]) if match[3] else ''
            except ValueError:
                continue
            elements.append({'ref': match[1], 'role': match[2], 'label': label})
    return elements


def compiled_routine_request(request, memory):
    """Select exact promoted work and lower it into existing WorkItem checkpoints.

    Only structured semantic condition contracts can be lowered automatically.
    Other routine formats continue to the adaptive path, never implicit success.
    """
    procedure = memory.find_promoted(fingerprint=request.get('operation_fingerprint'),
        scope=request.get('recipe_scope'), preconditions=request.get('routine_preconditions'))
    if procedure is None:
        return request
    if request.get('mutation_target', {}).get('scope') == 'external':
        raise ValueError('External routine COMMIT requires a persisted readback contract, not same-session snapshot')
    def expected(conditions):
        result = {}
        for condition in conditions:
            value = json.loads(condition)
            if not isinstance(value, dict) or not value:
                raise ValueError('Routine requires structured semantic condition contracts')
            result.update(value)
        return result
    before, after = expected(procedure.preconditions), expected(procedure.postconditions)
    if not before or not after:
        raise ValueError('Routine requires verified preconditions and semantic postconditions')
    steps = []
    for n, step in enumerate(procedure.steps):
        tool = {'navigate': 'browser_navigate', 'click': 'browser_click', 'type': 'browser_type',
                'press': 'browser_press', 'scroll': 'browser_scroll'}.get(step.action)
        if not tool:
            raise ValueError('Routine action is not admitted for automatic compilation')
        args = {'url': step.target} if step.action == 'navigate' else (
            {'text': step.value} if step.action == 'type' else {'key': step.value} if step.action == 'press' else
            {'direction': step.value} if step.action == 'scroll' else {})
        node = {'id': f'action_{n}', 'tool': tool, 'args': args, 'expect': {'success': True}}
        if step.action in {'click', 'type'}:
            if not step.fallback_anchors:
                raise ValueError('Routine requires reacquirable semantic anchors')
            node['semantic_anchor'] = step.fallback_anchors[0]
        if steps:
            node['depends_on'] = [steps[-1]['id']]
        steps.append(node)
    steps.append({'id': 'verify', 'tool': 'browser_snapshot', 'args': {},
                  'verifies': [s['id'] for s in steps], 'expect': after})
    from workstation.recipes import digest
    return {**request, 'operation_key': request.get('operation_key') or 'routine.' + procedure.id + '.' + digest(request.get('items', [{}]))[:16],
            'kind': 'browser_transaction', 'items': request.get('items', [{}]), 'steps': steps,
            'preflight': [{'tool': 'browser_snapshot', 'args': {}, 'expect': before}],
            'routine_id': procedure.id, 'routine_version': procedure.version}


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
    reasoning_handoff: dict[str, Any] | None = None


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

    def experience_candidate(self, outcome, contract, *, site: str, steps: list[dict],
                             repeatable: bool, metrics: dict | None = None):
        from workstation.contracts import AcceptanceEvaluator
        from workstation.recipes import sanitize
        if not repeatable or not steps or not outcome.evidence_refs or AcceptanceEvaluator().evaluate(outcome, contract):
            return None
        procedure = self.memory.record_success(site, sanitize(outcome.objective), sanitize(steps))
        procedure.validation_evidence = [sanitize({"uri": e.uri, "kind": e.kind}) for e in outcome.evidence_refs]
        procedure.savings = {"baseline": sanitize(metrics or outcome.metrics)}
        # A successful occurrence seeds a candidate; compatible replay must validate it.
        procedure.lifecycle = ProcedureLifecycle.DISCOVERED
        return self.memory.update_procedure(procedure)


class DeterministicRoutineRunner:
    def __init__(self, memory: ProceduralMemory, journal: ExecutionJournal | None = None) -> None:
        self.memory = memory
        self.journal = journal

    def run_matching(self, *, fingerprint, scope, preconditions, context, execute_step,
                     check_precondition=None, check_postcondition=None):
        """Harness-selected exact procedure; no model-supplied routine key needed."""
        procedure = self.memory.find_promoted(fingerprint=fingerprint, scope=scope,
                                              preconditions=preconditions)
        if procedure is None:
            return None  # Caller continues through bounded adaptive dispatch.
        return self.run(procedure.id, context, execute_step,
                        check_precondition=check_precondition, check_postcondition=check_postcondition)

    def run(
        self,
        procedure_id: str,
        context: dict[str, Any],
        execute_step: Callable[[ProcedureStep, str, dict[str, Any]], Any],
        *,
        check_precondition: Callable[[str, dict[str, Any]], bool] | None = None,
        check_postcondition: Callable[[str, dict[str, Any]], bool] | None = None,
        resume_ref: str | None = None,
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
            if check_precondition is None or not check_precondition(precondition, context):
                return self._drift(procedure, f"precondition failed: {precondition}")

        completed = 0
        if resume_ref:
            from workstation.artifacts import ArtifactStore
            from workstation.recipes import digest
            state = ArtifactStore().read_json(resume_ref)
            prior = state.get('context', {})
            if not state.get('safe_to_resume') or prior.get('procedure_id') != procedure.id or prior.get('version') != procedure.version or prior.get('fingerprint') != procedure_fingerprint(procedure):
                return RoutineExecutionResult(RoutineExecutionStatus.BLOCKED, procedure.id, reason='incompatible reasoning checkpoint')
            completed = prior['completed_steps']
        for step in procedure.steps[completed:]:
            elements = context.get('elements')  # Navigation may replace the live DOM.
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
                    reasoning_handoff=self._uncertain_handoff(procedure, completed, str(exc)),
                )
            completed += 1

        for postcondition in procedure.postconditions:
            if check_postcondition is None or not check_postcondition(postcondition, context):
                return self._drift(procedure, f"postcondition failed: {postcondition}", completed)
        result = RoutineExecutionResult(
            RoutineExecutionStatus.COMPLETED,
            procedure.id,
            version=procedure.version,
            completed_steps=completed,
        )
        self._record(
            ExecutionEventKind.ACTION,
            "deterministic routine completed",
            procedure.id,
            version=procedure.version,
            completed_steps=completed,
        )
        return result

    def _uncertain_handoff(self, procedure, completed, reason):
        from workstation.reasoning_handoff import needs_reasoning
        from workstation.artifacts import ArtifactStore
        return needs_reasoning(ArtifactStore(), procedure.id,
            completed_until=f'step_{completed - 1}' if completed else None,
            expected='confirmed effect or reconciled dispatch', observed=reason, safe_to_resume=False,
            context={'procedure_id': procedure.id, 'version': procedure.version})

    def _drift(self, procedure: Any, reason: str, completed: int = 0) -> RoutineExecutionResult:
        from workstation.artifacts import ArtifactStore
        from workstation.reasoning_handoff import needs_reasoning
        from workstation.recipes import digest
        self.memory.record_failure(procedure.id, completed, reason)
        procedure = self.memory.get_procedure(procedure.id)
        handoff = needs_reasoning(ArtifactStore(), procedure.id,
            completed_until=f'step_{completed - 1}' if completed else None,
            expected='compatible semantic anchors and procedure conditions', observed=reason,
            safe_to_resume=completed < len(procedure.steps),
            context={'procedure_id': procedure.id, 'version': procedure.version, 'completed_steps': completed,
                'fingerprint': procedure_fingerprint(procedure)})
        self._record(ExecutionEventKind.ERROR, f"routine drift: {reason}", procedure.id, completed_steps=completed)
        return RoutineExecutionResult(
            RoutineExecutionStatus.DRIFT,
            procedure.id,
            version=procedure.version,
            completed_steps=completed,
            reason=reason,
            reasoning_handoff=handoff,
        )

    def _record(self, kind: ExecutionEventKind, message: str, procedure_id: str, **metadata: Any) -> None:
        if self.journal is not None:
            self.journal.record(kind, message, metadata={"procedure_id": procedure_id, **metadata})
