from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MessageOrigin(str, Enum):
    HUMAN = "human"
    AGENT = "agent"
    WORKER = "worker"
    RUNTIME = "runtime"
    SYSTEM_EVENT = "system_event"
    CONNECTOR = "connector"


class IntentAuthority(str, Enum):
    CREATE_WORK = "create_work"
    DELEGATE_WORK = "delegate_work"
    UPDATE_WORK = "update_work"
    OBSERVATION_ONLY = "observation_only"


@dataclass(slots=True)
class MessageEnvelope:
    origin: MessageOrigin
    intent_authority: IntentAuthority
    session_id: str
    content: str
    parent_task_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def can_create_work(self) -> bool:
        # Constructed by trusted ingress; never deserialize authority from content.
        return bool(self.session_id) and (
            (self.origin == MessageOrigin.HUMAN and self.intent_authority == IntentAuthority.CREATE_WORK)
            or (self.origin == MessageOrigin.AGENT
                and self.intent_authority == IntentAuthority.DELEGATE_WORK
                and bool(self.parent_task_id))
        )


class OutcomeStatus(str, Enum):
    VERIFIED_COMPLETED = "verified_completed"
    WAITING_FOR_HUMAN = "waiting_for_human"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class AcceptanceContract:
    policy: str = "evidence"
    required_deliverables: list[str] = field(default_factory=list)
    required_verifiers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskOutcome:
    task_id: str
    session_id: str
    objective: str
    status: OutcomeStatus
    summary: str
    result_ref: str | None = None
    deliverables: list[str] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    verifier_results: list[dict[str, Any]] = field(default_factory=list)
    pending_items: list[str] = field(default_factory=list)
    planned_handoffs: list[str] = field(default_factory=list)
    unplanned_human_rescues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str = field(default_factory=utc_now)
    uncertain_mutation: bool = False
    run_id: str | None = None
    operation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class AcceptanceEvaluator:
    def evaluate(self, outcome: TaskOutcome, contract: AcceptanceContract) -> list[str]:
        reasons = []
        if outcome.status != OutcomeStatus.VERIFIED_COMPLETED:
            reasons.append("outcome_not_verified")
        if outcome.pending_items:
            reasons.append("blocking_pending_items")
        if outcome.uncertain_mutation:
            reasons.append("uncertain_mutation_requires_review")
        evidence_uris = {e.uri for e in outcome.evidence_refs if e.uri}
        passed = {v.get("verifier") for v in outcome.verifier_results
                  if v.get("passed") is True and v.get("evidence_ref") in evidence_uris}
        if any(v.get("required", True) and v.get("passed") is not True for v in outcome.verifier_results):
            reasons.append("verifier_failed")
        if set(contract.required_verifiers) - passed:
            reasons.append("required_verifier_missing")
        if set(contract.required_deliverables) - set(outcome.deliverables):
            reasons.append("required_deliverable_missing")
        if contract.policy == "advisory":
            if not outcome.summary.strip():
                reasons.append("advisory_result_missing")
        elif contract.policy == "evidence":
            if not passed or not outcome.evidence_refs:
                reasons.append("verification_evidence_missing")
        else:
            reasons.append("unknown_acceptance_policy")
        return reasons


class ExecutionEventKind(str, Enum):
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    BROWSER_ATTACHED = "browser_attached"
    NAVIGATION = "navigation"
    ACTION = "action"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    RETRY = "retry"
    FOLLOWUP_CREATED = "followup_created"
    SCREENSHOT = "screenshot"
    ERROR = "error"
    TASK_COMPLETED = "task_completed"
    TOOL_CALL = "tool_call"
    WORKER_MESSAGE = "worker_message"
    PROGRESS = "progress"
    USAGE = "usage"
    DELIVERABLE = "deliverable"
    RECOVERY = "recovery"
    LIFECYCLE = "lifecycle"


@dataclass(slots=True)
class EvidenceRef:
    kind: str
    uri: str
    summary: str = ""
    sha256: str | None = None
    run_id: str | None = None
    operation_id: str | None = None
    task_id: str | None = None
    verifier: str | None = None


@dataclass(slots=True)
class DiscoveredTask:
    title: str
    parent_task_id: str
    discovered_by: str
    reason: str
    origin_session_id: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    task_id: str | None = None
    required_for_parent: bool = False

    def validate(self) -> None:
        required = {
            "title": self.title,
            "parent_task_id": self.parent_task_id,
            "discovered_by": self.discovered_by,
            "reason": self.reason,
            "origin_session_id": self.origin_session_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Discovered task missing required fields: {', '.join(missing)}")


@dataclass(slots=True)
class ExecutionEvent:
    kind: ExecutionEventKind
    task_id: str
    session_id: str
    message: str
    timestamp: str = field(default_factory=utc_now)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    browser_tab_id: str | None = None
    url: str | None = None
    risk: RiskLevel = RiskLevel.LOW
    evidence: list[EvidenceRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    operation_id: str | None = None
    schema_version: int | None = None
    sequence_number: int | None = None
    previous_event_hash: str | None = None
    event_hash: str | None = None
    writer_id: str | None = None
    environment: str | None = None
    build_sha: str | None = None
    workstation_version: str | None = None
    test_case_id: str | None = None
    evaluation_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["risk"] = self.risk.value
        return data


@dataclass(slots=True)
class BrowserTaskReport:
    task_id: str
    session_id: str
    objective: str
    result: str
    completed: bool
    actions: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)
    modified_items: list[str] = field(default_factory=list)
    errors_and_retries: list[str] = field(default_factory=list)
    discovered_tasks: list[DiscoveredTask] = field(default_factory=list)
    pending_items: list[str] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)
    verifier_results: list[dict[str, Any]] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    uncertain_mutation: bool = False
    acceptance_contract: AcceptanceContract = field(default_factory=AcceptanceContract)
    outcome_status: OutcomeStatus | None = None
    repeatability_hint: bool = False
    procedure_steps: list[dict[str, Any]] = field(default_factory=list)
    procedure_scope: str = ""
    run_id: str | None = None
    operation_id: str | None = None

    def to_kanban_metadata(self) -> dict[str, Any]:
        return {
            "workstation": {
                "report_version": 1,
                "session_id": self.session_id,
                "run_id": self.run_id,
                "operation_id": self.operation_id,
                "sites": sorted(set(self.sites)),
                "modified_items": self.modified_items,
                "errors_and_retries": self.errors_and_retries,
                "followups": [
                    {
                        "task_id": task.task_id,
                        "title": task.title,
                        "parent_task_id": task.parent_task_id,
                        "discovered_by": task.discovered_by,
                        "reason": task.reason,
                        "origin_session_id": task.origin_session_id,
                        "required_for_parent": task.required_for_parent,
                    }
                    for task in self.discovered_tasks
                ],
                "pending_items": self.pending_items,
                "approvals": self.approvals,
                "duration_seconds": self.duration_seconds,
                "tokens": {"input": self.input_tokens, "output": self.output_tokens},
                "cost_usd": self.cost_usd,
            }
        }
