"""Evidence-backed runtime contracts shared by Workstation surfaces.

This module deliberately contains projections and transport contracts, not a
second task/session store. Canonical task lineage remains in Hermes/Kanban,
browser ownership remains in BrowserTask, and durable action history remains in
ExecutionJournal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Iterable
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class ExecutionStatus(str, Enum):
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting-for-human"
    BLOCKED = "blocked"
    STALLED = "stalled"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    HOLD = "hold"


@dataclass(slots=True)
class OperationalEvidence:
    kind: str
    reference: str
    observed_at: str = field(default_factory=_utc_now)
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_live(self, now: str | None = None) -> bool:
        if not self.reference.strip():
            return False
        if not self.expires_at:
            return True
        return _parse_time(self.expires_at) > _parse_time(now or _utc_now())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceState:
    """Operational state projection; ``running`` is earned by live evidence."""

    task_id: str
    session_id: str
    status: ExecutionStatus = ExecutionStatus.READY
    started_at: str | None = None
    last_activity: str | None = None
    evidence: list[OperationalEvidence] = field(default_factory=list)
    recovery_strategy: str | None = None
    blocked_reason: str | None = None
    approval_id: str | None = None
    version: int = 1

    def transition(
        self,
        status: ExecutionStatus,
        *,
        now: str | None = None,
        reason: str | None = None,
        recovery_strategy: str | None = None,
        approval_id: str | None = None,
    ) -> None:
        stamp = now or _utc_now()
        if status == ExecutionStatus.RUNNING and not self.live_evidence(stamp):
            self.status = ExecutionStatus.STALLED
            self.recovery_strategy = recovery_strategy or "attach-live-evidence"
            self.blocked_reason = reason or "running state has no live operational evidence"
            self.last_activity = stamp
            return
        self.status = status
        self.last_activity = stamp
        if status == ExecutionStatus.RUNNING and self.started_at is None:
            self.started_at = stamp
        if reason is not None:
            self.blocked_reason = reason
        if recovery_strategy is not None:
            self.recovery_strategy = recovery_strategy
        if approval_id is not None:
            self.approval_id = approval_id

    def add_evidence(
        self,
        kind: str,
        reference: str,
        *,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        now: str | None = None,
    ) -> OperationalEvidence:
        stamp = now or _utc_now()
        expires_at = None
        if ttl_seconds is not None:
            expires_at = (
                _parse_time(stamp) + timedelta(seconds=max(0.0, ttl_seconds))
            ).isoformat()
        item = OperationalEvidence(
            kind=kind,
            reference=reference,
            observed_at=stamp,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        self.evidence.append(item)
        self.last_activity = stamp
        if self.status in {
            ExecutionStatus.READY,
            ExecutionStatus.QUEUED,
            ExecutionStatus.STALLED,
        }:
            self.status = ExecutionStatus.RUNNING
            self.started_at = self.started_at or stamp
            self.blocked_reason = None
        return item

    def live_evidence(self, now: str | None = None) -> list[OperationalEvidence]:
        return [item for item in self.evidence if item.is_live(now)]

    def reconcile(self, *, now: str | None = None, stale_after_seconds: float = 120.0) -> ExecutionStatus:
        stamp = now or _utc_now()
        if self.status != ExecutionStatus.RUNNING:
            return self.status
        live = self.live_evidence(stamp)
        last_activity = self.last_activity or self.started_at
        stale = False
        if not live:
            stale = True
        elif last_activity:
            stale = (_parse_time(stamp) - _parse_time(last_activity)).total_seconds() > stale_after_seconds
        if stale:
            self.status = ExecutionStatus.STALLED
            self.blocked_reason = self.blocked_reason or "operational evidence became stale"
            self.recovery_strategy = self.recovery_strategy or "reconcile-or-handoff"
            self.last_activity = stamp
        return self.status

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceState":
        return cls(
            task_id=str(data["task_id"]),
            session_id=str(data["session_id"]),
            status=ExecutionStatus(data.get("status", ExecutionStatus.READY.value)),
            started_at=data.get("started_at"),
            last_activity=data.get("last_activity"),
            evidence=[
                OperationalEvidence(
                    kind=str(item["kind"]),
                    reference=str(item["reference"]),
                    observed_at=str(item.get("observed_at", _utc_now())),
                    expires_at=item.get("expires_at"),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in data.get("evidence", [])
                if isinstance(item, dict) and item.get("kind") and item.get("reference")
            ],
            recovery_strategy=data.get("recovery_strategy"),
            blocked_reason=data.get("blocked_reason"),
            approval_id=data.get("approval_id"),
            version=int(data.get("version", 1)),
        )


class EvidenceStateStore:
    """Durable projection of canonical task evidence, not a task store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._states: dict[str, EvidenceState] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return
        for item in data.get("states", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict) or not item.get("task_id"):
                continue
            try:
                state = EvidenceState.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue
            self._states[state.task_id] = state

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + f".{threading.get_ident()}.tmp")
        temp.write_text(
            json.dumps({"schema_version": 1, "states": [state.to_dict() for state in self._states.values()]}, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def upsert(self, state: EvidenceState) -> EvidenceState:
        with self._lock:
            self._states[state.task_id] = state
            self._persist()
        return state

    def get(self, task_id: str) -> EvidenceState | None:
        return self._states.get(task_id)

    def reconcile_all(
        self,
        *,
        now: str | None = None,
        stale_after_seconds: float = 120.0,
    ) -> list[EvidenceState]:
        changed: list[EvidenceState] = []
        with self._lock:
            for state in self._states.values():
                before = state.status
                state.reconcile(now=now, stale_after_seconds=stale_after_seconds)
                if state.status != before:
                    changed.append(state)
            if changed:
                self._persist()
        return changed


@dataclass(slots=True)
class RuntimeEvent:
    type: str
    task_id: str | None = None
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt-{uuid4().hex}")
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventSubscription:
    def __init__(self, capacity: int) -> None:
        self._queue: queue.Queue[RuntimeEvent | None] = queue.Queue(maxsize=max(1, capacity))
        self._closed = False

    def put(self, event: RuntimeEvent) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
                return False
            except queue.Empty:
                return False

    def get(self, timeout: float | None = None) -> RuntimeEvent:
        item = self._queue.get(timeout=timeout)
        if item is None:
            raise RuntimeError("event subscription closed")
        return item

    def close(self) -> None:
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except queue.Empty:
                pass


class RuntimeEventBus:
    """Non-blocking fan-out bus with bounded subscriber queues."""

    def __init__(self, journal: Any | None = None) -> None:
        self._subscriptions: list[EventSubscription] = []
        self._lock = threading.Lock()
        self._closed = False
        self.dropped_events = 0
        self._journal = journal
        self.journal_errors: list[str] = []

    def subscribe(self, *, capacity: int = 128) -> EventSubscription:
        subscription = EventSubscription(capacity)
        with self._lock:
            if self._closed:
                subscription.close()
            else:
                self._subscriptions.append(subscription)
        return subscription

    def publish(self, event: RuntimeEvent) -> None:
        if self._journal is not None and event.task_id and event.session_id:
            try:
                from workstation.contracts import ExecutionEventKind

                kind_map = {
                    "task.created": ExecutionEventKind.TASK_CREATED,
                    "task.started": ExecutionEventKind.TASK_STARTED,
                    "task.completed": ExecutionEventKind.TASK_COMPLETED,
                    "error": ExecutionEventKind.ERROR,
                    "approval.requested": ExecutionEventKind.APPROVAL_REQUESTED,
                    "worker.message": ExecutionEventKind.WORKER_MESSAGE,
                    "worker.result": ExecutionEventKind.DELIVERABLE,
                    "tool.call": ExecutionEventKind.TOOL_CALL,
                    "progress": ExecutionEventKind.PROGRESS,
                    "usage": ExecutionEventKind.USAGE,
                    "recovery": ExecutionEventKind.RECOVERY,
                    "lifecycle": ExecutionEventKind.LIFECYCLE,
                }
                kind = kind_map.get(event.type, ExecutionEventKind.ACTION)
                self._journal.record(kind, event.type, metadata=dict(event.payload))
            except Exception as exc:
                # A journal adapter must not block unrelated subscribers, but
                # persistence failure remains observable to the control plane.
                self.journal_errors.append(str(exc))
        with self._lock:
            subscriptions = list(self._subscriptions)
        for subscription in subscriptions:
            if not subscription.put(event):
                self.dropped_events += 1

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subscriptions = list(self._subscriptions)
            self._subscriptions.clear()
        for subscription in subscriptions:
            subscription.close()


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = "cancelled"

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


def run_with_deadline(
    fn: Callable[[], Any],
    *,
    timeout_seconds: float,
    cancellation: CancellationToken | None = None,
) -> Any:
    if cancellation and cancellation.cancelled:
        raise RuntimeError(cancellation.reason)
    result_holder: list[Any] = []
    error_holder: list[BaseException] = []

    def invoke() -> None:
        try:
            result_holder.append(fn())
        except BaseException as exc:  # propagate the operation's original error
            error_holder.append(exc)

    thread = threading.Thread(target=invoke, name="hermes-deadline-operation", daemon=True)
    thread.start()
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while thread.is_alive():
        if cancellation and cancellation.cancelled:
            raise RuntimeError(cancellation.reason)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"operation exceeded deadline of {timeout_seconds}s")
        thread.join(timeout=min(0.05, remaining))
    if cancellation and cancellation.cancelled:
        raise RuntimeError(cancellation.reason)
    if error_holder:
        raise error_holder[0]
    return result_holder[0] if result_holder else None


@dataclass(slots=True)
class TypedResource:
    resource_type: str
    resource_id: str
    task_id: str | None = None
    session_id: str | None = None
    permissions: tuple[str, ...] = ()
    state: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["permissions"] = list(self.permissions)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedResource":
        return cls(
            resource_type=str(data["resource_type"]),
            resource_id=str(data["resource_id"]),
            task_id=data.get("task_id"),
            session_id=data.get("session_id"),
            permissions=tuple(str(item) for item in data.get("permissions", [])),
            state=dict(data.get("state", {})),
            updated_at=str(data.get("updated_at", _utc_now())),
        )


class ResourceRegistry:
    """Typed resource projection shared by clients; not a task database."""

    def __init__(self, path: Path | None = None) -> None:
        self._resources: dict[tuple[str, str], TypedResource] = {}
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return
        for item in data.get("resources", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict):
                continue
            try:
                resource = TypedResource.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue
            self._resources[(resource.resource_type, resource.resource_id)] = resource

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + f".{threading.get_ident()}.tmp")
        temp.write_text(
            json.dumps({"schema_version": 1, "resources": self.snapshot()}, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def upsert(self, resource: TypedResource) -> TypedResource:
        with self._lock:
            self._resources[(resource.resource_type, resource.resource_id)] = resource
            self._persist()
        return resource

    def get(self, resource_type: str, resource_id: str) -> TypedResource | None:
        return self._resources.get((resource_type, resource_id))

    def snapshot(self) -> list[dict[str, Any]]:
        return [resource.to_dict() for resource in self._resources.values()]


@dataclass(slots=True)
class BudgetTracker:
    max_actions: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    actions: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    def consume(self, *, actions: int = 0, tokens: int = 0, cost_usd: float = 0.0) -> None:
        next_actions = self.actions + max(0, actions)
        next_tokens = self.tokens + max(0, tokens)
        next_cost = self.cost_usd + max(0.0, cost_usd)
        if self.max_actions is not None and next_actions > self.max_actions:
            raise BudgetExceeded("action budget exceeded")
        if self.max_tokens is not None and next_tokens > self.max_tokens:
            raise BudgetExceeded("token budget exceeded")
        if self.max_cost_usd is not None and next_cost > self.max_cost_usd:
            raise BudgetExceeded("cost budget exceeded")
        self.actions, self.tokens, self.cost_usd = next_actions, next_tokens, next_cost

    def remaining(self) -> dict[str, int | float | None]:
        return {
            "actions": None if self.max_actions is None else self.max_actions - self.actions,
            "tokens": None if self.max_tokens is None else self.max_tokens - self.tokens,
            "cost_usd": None if self.max_cost_usd is None else self.max_cost_usd - self.cost_usd,
        }


class BudgetExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class HumanHandoff:
    handoff_id: str
    task_id: str
    session_id: str
    reason: str
    scope: dict[str, str]
    status: str = "waiting-for-human"
    requested_at: str = field(default_factory=_utc_now)
    resumed_at: str | None = None
    returned_by: str | None = None


class HumanHandoffManager:
    """Explicitly transfers task control without changing task/page identity."""

    def __init__(self, path: Path | None = None) -> None:
        self._handoffs: dict[str, HumanHandoff] = {}
        self.path = Path(path) if path is not None else None
        self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return
        for item in data.get("handoffs", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict) or not item.get("handoff_id"):
                continue
            try:
                self._handoffs[str(item["handoff_id"])] = HumanHandoff(**item)
            except (TypeError, ValueError):
                continue

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + f".{threading.get_ident()}.tmp")
        temp.write_text(
            json.dumps({"schema_version": 1, "handoffs": [asdict(item) for item in self._handoffs.values()]}, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def request(
        self,
        *,
        task_id: str,
        session_id: str,
        reason: str,
        scope: dict[str, str],
    ) -> HumanHandoff:
        if not reason.strip() or not task_id.strip() or not session_id.strip():
            raise ValueError("task, session and reason are required for handoff")
        handoff = HumanHandoff(
            handoff_id=f"handoff-{uuid4().hex}",
            task_id=task_id,
            session_id=session_id,
            reason=reason,
            scope=dict(scope),
        )
        self._handoffs[handoff.handoff_id] = handoff
        self._persist()
        return handoff

    def resume(self, handoff_id: str, *, returned_by: str) -> bool:
        handoff = self._handoffs.get(handoff_id)
        if handoff is None or handoff.status != "waiting-for-human":
            return False
        if not returned_by.strip():
            raise ValueError("returned_by is required")
        handoff.status = "ready"
        handoff.returned_by = returned_by
        handoff.resumed_at = _utc_now()
        self._persist()
        return True

    def get(self, handoff_id: str) -> HumanHandoff | None:
        return self._handoffs.get(handoff_id)


@dataclass(slots=True)
class RoutingCandidate:
    model_id: str
    provider: str
    quality: float
    cost_usd: float
    latency_ms: float
    available: bool = True


@dataclass(slots=True)
class RouteDecision:
    model_id: str
    provider: str
    reason: str
    score: float
    candidates_considered: list[str]
    fallback: bool = False
    decided_at: str = field(default_factory=_utc_now)


class ModelRouter:
    """Deterministic, observable model choice; provider calls remain elsewhere."""

    def __init__(self, candidates: Iterable[RoutingCandidate]) -> None:
        self.candidates = list(candidates)

    def choose(
        self,
        *,
        complexity: float,
        risk: float,
        max_cost_usd: float | None = None,
        max_latency_ms: float | None = None,
    ) -> RouteDecision:
        considered = [candidate.model_id for candidate in self.candidates]
        eligible = [
            candidate
            for candidate in self.candidates
            if candidate.available
            and (max_cost_usd is None or candidate.cost_usd <= max_cost_usd)
            and (max_latency_ms is None or candidate.latency_ms <= max_latency_ms)
        ]
        if not eligible:
            raise RuntimeError("no available model satisfies routing constraints")
        complexity = min(1.0, max(0.0, complexity))
        risk = min(1.0, max(0.0, risk))

        def score(candidate: RoutingCandidate) -> float:
            quality_weight = 1.0 + complexity + risk
            cost_penalty = candidate.cost_usd * 5.0
            latency_penalty = candidate.latency_ms / 2000.0
            return candidate.quality * quality_weight - cost_penalty - latency_penalty

        selected = max(eligible, key=score)
        return RouteDecision(
            model_id=selected.model_id,
            provider=selected.provider,
            reason=(
                f"selected for complexity={complexity:.2f}, risk={risk:.2f}; "
                f"score={score(selected):.3f}"
            ),
            score=score(selected),
            candidates_considered=considered,
            fallback=selected is not self.candidates[0] if self.candidates else False,
        )
