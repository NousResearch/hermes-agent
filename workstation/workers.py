from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque
import json
import logging
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Optional
from uuid import uuid4

from hermes_constants import get_hermes_home
from workstation.contracts import ExecutionEventKind, RiskLevel
from workstation.journal import ExecutionJournal
from workstation.runtime import RuntimeEvent, RuntimeEventBus

_log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkerStatus(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass(slots=True)
class WorkerHarnessInfo:
    worker_id: str
    name: str
    harness_type: str  # "cli" | "sdk" | "mcp" | "subagent"
    executable: str
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: WorkerStatus = WorkerStatus.AVAILABLE

    def is_installed(self) -> bool:
        if not self.executable:
            return True
        return shutil.which(self.executable) is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["installed"] = self.is_installed()
        return data


@dataclass(slots=True)
class DelegatedTaskHandoff:
    handoff_id: str
    worker_id: str
    parent_task_id: str
    session_id: str
    subtask_prompt: str
    created_at: str = field(default_factory=_utc_now)
    completed_at: str | None = None
    result: str | None = None
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class PersistentWorkerStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class WorkerMessage:
    worker_id: str
    content: str
    sender_id: str
    message_id: str = field(default_factory=lambda: f"msg-{uuid4().hex}")
    created_at: str = field(default_factory=_utc_now)
    kind: str = "message"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerMessage":
        return cls(
            worker_id=str(data["worker_id"]),
            content=str(data["content"]),
            sender_id=str(data.get("sender_id", "hermes")),
            message_id=str(data.get("message_id", f"msg-{uuid4().hex}")),
            created_at=str(data.get("created_at", _utc_now())),
            kind=str(data.get("kind", "message")),
        )


@dataclass(slots=True)
class WorkerResultEnvelope:
    worker_id: str
    parent_task_id: str
    session_id: str
    sequence: int
    status: str
    deliverables: list[str] = field(default_factory=list)
    semantic_status: str = "completed"
    model: str | None = None
    provider: str | None = None
    duration_seconds: float = 0.0
    usage: dict[str, int | float] = field(default_factory=dict)
    cost_usd: float | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=_utc_now)


@dataclass(slots=True)
class WorkerExecutionOutput:
    """Optional structured executor result for operational metadata."""

    deliverables: list[str] = field(default_factory=list)
    semantic_status: str = "completed"
    model: str | None = None
    provider: str | None = None
    usage: dict[str, int | float] = field(default_factory=dict)
    cost_usd: float | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class _PersistentWorkerRecord:
    worker_id: str
    parent_task_id: str
    session_id: str
    status: PersistentWorkerStatus = PersistentWorkerStatus.STOPPED
    last_sequence: int = 0
    created_at: str = field(default_factory=_utc_now)
    pending_messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class PersistentWorker:
    def __init__(
        self,
        record: _PersistentWorkerRecord,
        executor_fn: Callable[[str], Any],
        *,
        journal: ExecutionJournal | None = None,
        event_bus: RuntimeEventBus | None = None,
        on_update: Callable[["PersistentWorker"], None] | None = None,
    ) -> None:
        self.worker_id = record.worker_id
        self.parent_task_id = record.parent_task_id
        self.session_id = record.session_id
        self.status = record.status
        self._sequence = record.last_sequence
        self._executor_fn = executor_fn
        self._journal = journal
        self._event_bus = event_bus
        self._on_update = on_update
        self._messages: deque[WorkerMessage] = deque()
        self._messages.extend(
            WorkerMessage.from_dict(item)
            for item in record.pending_messages
            if isinstance(item, dict)
        )
        self._results: list[WorkerResultEnvelope] = []
        self._condition = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> "PersistentWorker":
        with self._condition:
            if self._thread and self._thread.is_alive():
                return self
            self._stop = False
            self.status = PersistentWorkerStatus.READY
            self._thread = threading.Thread(
                target=self._run,
                name=f"hermes-worker-{self.worker_id}",
                daemon=True,
            )
            self._thread.start()
        self._notify_update()
        return self

    def enqueue(self, message: WorkerMessage, *, priority: bool = False) -> None:
        with self._condition:
            if self.status in {PersistentWorkerStatus.STOPPING, PersistentWorkerStatus.STOPPED}:
                raise RuntimeError(f"worker '{self.worker_id}' is stopped")
            if priority:
                self._messages.appendleft(message)
            else:
                self._messages.append(message)
            self._condition.notify_all()
        self._notify_update()
        if self._journal:
            self._journal.record(
                ExecutionEventKind.WORKER_MESSAGE,
                f"Persistent worker '{self.worker_id}' received {message.kind}",
                metadata={
                    "worker_id": self.worker_id,
                    "message_id": message.message_id,
                    "sender_id": message.sender_id,
                    "kind": message.kind,
                },
            )
        if self._event_bus:
            self._event_bus.publish(
                RuntimeEvent(
                    type="worker.message",
                    task_id=self.parent_task_id,
                    session_id=self.session_id,
                    payload={
                        "worker_id": self.worker_id,
                        "message_id": message.message_id,
                        "sender_id": message.sender_id,
                        "kind": message.kind,
                    },
                )
            )

    def pause(self) -> None:
        with self._condition:
            if self.status in {PersistentWorkerStatus.READY, PersistentWorkerStatus.RUNNING}:
                self.status = PersistentWorkerStatus.PAUSED
            self._condition.notify_all()
        self._notify_update()

    def resume(self) -> None:
        with self._condition:
            if self.status == PersistentWorkerStatus.PAUSED:
                self.status = PersistentWorkerStatus.READY
            self._condition.notify_all()
        self._notify_update()

    def stop(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._stop = True
            self.status = PersistentWorkerStatus.STOPPING
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=max(0.0, timeout))
        with self._condition:
            if self._thread and self._thread.is_alive():
                self.status = PersistentWorkerStatus.FAILED
            else:
                self.status = PersistentWorkerStatus.STOPPED
        self._notify_update()

    def wait(self, timeout: float | None = None, *, after_sequence: int = 0) -> WorkerResultEnvelope | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                for result in self._results:
                    if result.sequence > after_sequence:
                        return result
                if self.status in {PersistentWorkerStatus.STOPPED, PersistentWorkerStatus.FAILED}:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stop and (self.status == PersistentWorkerStatus.PAUSED or not self._messages):
                    self._condition.wait(timeout=0.5)
                if self._stop:
                    return
                message = self._messages.popleft()
                self.status = PersistentWorkerStatus.RUNNING
            self._notify_update()

            started = time.monotonic()
            try:
                output = self._executor_fn(message.content)
                if isinstance(output, WorkerExecutionOutput):
                    execution = output
                elif isinstance(output, dict):
                    execution = WorkerExecutionOutput(
                        deliverables=[str(item) for item in output.get("deliverables", [])],
                        semantic_status=str(output.get("semantic_status", output.get("status", "completed"))),
                        model=output.get("model"),
                        provider=output.get("provider"),
                        usage=dict(output.get("usage", {})),
                        cost_usd=output.get("cost_usd"),
                        evidence=list(output.get("evidence", [])),
                    )
                else:
                    deliverables = output if isinstance(output, list) else [str(output)]
                    execution = WorkerExecutionOutput(deliverables=[str(item) for item in deliverables])
                result = WorkerResultEnvelope(
                    worker_id=self.worker_id,
                    parent_task_id=self.parent_task_id,
                    session_id=self.session_id,
                    sequence=self._sequence + 1,
                    status="completed",
                    deliverables=execution.deliverables,
                    semantic_status=execution.semantic_status,
                    model=execution.model,
                    provider=execution.provider,
                    duration_seconds=time.monotonic() - started,
                    usage=execution.usage,
                    cost_usd=execution.cost_usd,
                    evidence=execution.evidence,
                )
                if self._journal:
                    self._journal.record(
                        ExecutionEventKind.DELIVERABLE,
                        f"Persistent worker '{self.worker_id}' delivered a result",
                        metadata={
                            "worker_id": self.worker_id,
                            "sequence": result.sequence,
                            "semantic_status": result.semantic_status,
                            "model": result.model,
                            "provider": result.provider,
                            "usage": result.usage,
                            "cost_usd": result.cost_usd,
                            "evidence": result.evidence,
                        },
                    )
            except Exception as exc:
                result = WorkerResultEnvelope(
                    worker_id=self.worker_id,
                    parent_task_id=self.parent_task_id,
                    session_id=self.session_id,
                    sequence=self._sequence + 1,
                    status="failed",
                    semantic_status="failed",
                    duration_seconds=time.monotonic() - started,
                    error=str(exc),
                )
                if self._journal:
                    self._journal.record(
                        ExecutionEventKind.ERROR,
                        f"Persistent worker '{self.worker_id}' failed",
                        metadata={"worker_id": self.worker_id, "sequence": result.sequence},
                    )
            with self._condition:
                self._sequence = result.sequence
                self._results.append(result)
                self.status = PersistentWorkerStatus.READY if not self._stop else PersistentWorkerStatus.STOPPING
                self._condition.notify_all()
            self._notify_update()
            if self._event_bus:
                self._event_bus.publish(
                    RuntimeEvent(
                        type="worker.result" if result.status == "completed" else "error",
                        task_id=self.parent_task_id,
                        session_id=self.session_id,
                        payload={
                            "worker_id": self.worker_id,
                            "sequence": result.sequence,
                            "status": result.status,
                            "semantic_status": result.semantic_status,
                            "deliverables": result.deliverables,
                            "model": result.model,
                            "provider": result.provider,
                            "usage": result.usage,
                            "cost_usd": result.cost_usd,
                            "evidence": result.evidence,
                            "error": result.error,
                        },
                    )
                )

    def _notify_update(self) -> None:
        if self._on_update:
            self._on_update(self)


class WorkerRegistry:
    """Registry and orchestrator for specialized worker agents (Codex, Claude Code, Antigravity, OpenCode).

    Retains canonical Hermes task/session/card lineage while delegating bounded subtasks.
    """

    def __init__(self, storage_path: Path | None = None, *, event_bus: RuntimeEventBus | None = None) -> None:
        self._workers: dict[str, WorkerHarnessInfo] = {}
        self._handoffs: dict[str, DelegatedTaskHandoff] = {}
        self.storage_path = Path(storage_path or get_hermes_home() / "workstation" / "workers.json")
        self._persistent_records: dict[str, _PersistentWorkerRecord] = {}
        self._persistent_workers: dict[str, PersistentWorker] = {}
        self._persistent_lock = threading.Lock()
        self._event_bus = event_bus
        self._load_persistent_records()
        self._register_default_known_workers()

    def _load_persistent_records(self) -> None:
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return
        for item in data.get("workers", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict) or not item.get("worker_id"):
                continue
            try:
                record = _PersistentWorkerRecord(
                    worker_id=str(item["worker_id"]),
                    parent_task_id=str(item["parent_task_id"]),
                    session_id=str(item["session_id"]),
                    status=PersistentWorkerStatus(item.get("status", PersistentWorkerStatus.STOPPED.value)),
                    last_sequence=int(item.get("last_sequence", 0)),
                    created_at=str(item.get("created_at", _utc_now())),
                    pending_messages=[
                        dict(message)
                        for message in item.get("pending_messages", [])
                        if isinstance(message, dict)
                    ],
                )
            except (KeyError, TypeError, ValueError):
                continue
            record.status = PersistentWorkerStatus.STOPPED
            self._persistent_records[record.worker_id] = record

    def _persist_persistent_records(self) -> None:
        with self._persistent_lock:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.storage_path.with_suffix(self.storage_path.suffix + f".{threading.get_ident()}.tmp")
            temp.write_text(
                json.dumps({"workers": [record.to_dict() for record in self._persistent_records.values()]}, indent=2),
                encoding="utf-8",
            )
            # Windows security/indexing software can briefly hold either the
            # destination or the just-written temp file. Keep the atomic
            # replace contract, but tolerate a short transient sharing lock;
            # persistent failures still surface to the caller.
            for attempt in range(8):
                try:
                    temp.replace(self.storage_path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.01 * (attempt + 1))

    def _sync_persistent_worker(self, worker: PersistentWorker) -> None:
        record = self._persistent_records.get(worker.worker_id)
        if record is None:
            record = _PersistentWorkerRecord(worker.worker_id, worker.parent_task_id, worker.session_id)
            self._persistent_records[worker.worker_id] = record
        record.status = worker.status
        record.last_sequence = worker._sequence
        with worker._condition:
            record.pending_messages = [message.to_dict() for message in worker._messages]
        self._persist_persistent_records()

    def start_persistent_worker(
        self,
        worker_id: str,
        parent_task_id: str,
        session_id: str,
        *,
        executor_fn: Callable[[str], Any],
        journal: ExecutionJournal | None = None,
        event_bus: RuntimeEventBus | None = None,
    ) -> PersistentWorker:
        worker = self._persistent_workers.get(worker_id)
        if worker is not None and worker.status not in {PersistentWorkerStatus.STOPPED, PersistentWorkerStatus.FAILED}:
            return worker
        record = _PersistentWorkerRecord(worker_id, parent_task_id, session_id)
        worker = PersistentWorker(
            record,
            executor_fn,
            journal=journal,
            event_bus=event_bus or self._event_bus,
            on_update=self._sync_persistent_worker,
        )
        self._persistent_records[worker_id] = record
        self._persistent_workers[worker_id] = worker
        worker.start()
        return worker

    def reconstruct_worker(
        self,
        worker_id: str,
        *,
        executor_fn: Callable[[str], Any],
        journal: ExecutionJournal | None = None,
        event_bus: RuntimeEventBus | None = None,
    ) -> PersistentWorker:
        record = self._persistent_records.get(worker_id)
        if record is None:
            raise KeyError(worker_id)
        worker = PersistentWorker(
            record,
            executor_fn,
            journal=journal,
            event_bus=event_bus or self._event_bus,
            on_update=self._sync_persistent_worker,
        )
        self._persistent_workers[worker_id] = worker
        worker.start()
        return worker

    def get_persistent_worker(self, worker_id: str) -> PersistentWorker:
        worker = self._persistent_workers.get(worker_id)
        if worker is None:
            raise KeyError(worker_id)
        return worker

    def send_message(self, worker_id: str, content: str, *, sender_id: str = "hermes") -> WorkerMessage:
        message = WorkerMessage(worker_id=worker_id, content=content, sender_id=sender_id)
        self.get_persistent_worker(worker_id).enqueue(message)
        return message

    def steer(self, worker_id: str, content: str, *, sender_id: str = "hermes") -> WorkerMessage:
        message = WorkerMessage(worker_id=worker_id, content=content, sender_id=sender_id, kind="steer")
        self.get_persistent_worker(worker_id).enqueue(message, priority=True)
        return message

    def wait(self, worker_id: str, timeout: float | None = None, *, after_sequence: int = 0) -> WorkerResultEnvelope | None:
        return self.get_persistent_worker(worker_id).wait(timeout, after_sequence=after_sequence)

    def pause_worker(self, worker_id: str) -> None:
        self.get_persistent_worker(worker_id).pause()

    def resume_worker(self, worker_id: str) -> None:
        self.get_persistent_worker(worker_id).resume()

    def stop_worker(self, worker_id: str, timeout: float = 5.0) -> None:
        self.get_persistent_worker(worker_id).stop(timeout)

    def _register_default_known_workers(self) -> None:
        defaults = [
            WorkerHarnessInfo(
                worker_id="antigravity",
                name="Antigravity Coding Assistant",
                harness_type="subagent",
                executable="agy",
                capabilities=["code_editing", "codebase_research", "terminal_execution", "tdd"],
            ),
            WorkerHarnessInfo(
                worker_id="claude-code",
                name="Claude Code CLI",
                harness_type="cli",
                executable="claude",
                capabilities=["code_editing", "terminal_execution", "large_refactoring"],
            ),
            WorkerHarnessInfo(
                worker_id="codex",
                name="OpenAI Codex Harness",
                harness_type="cli",
                executable="codex",
                capabilities=["code_generation", "unit_test_authoring"],
            ),
            WorkerHarnessInfo(
                worker_id="opencode",
                name="OpenCode Worker",
                harness_type="cli",
                executable="opencode",
                capabilities=["code_editing", "git_operations"],
            ),
            WorkerHarnessInfo(
                worker_id="k-tools-neo",
                name="K-Tools-Neo Host Automation",
                harness_type="cli",
                executable="ktools",
                capabilities=["windows_automation", "clipboard", "system_diagnostics"],
            ),
        ]
        for w in defaults:
            self.register_worker(w)

    def register_worker(self, worker: WorkerHarnessInfo) -> None:
        self._workers[worker.worker_id] = worker

    def get_worker(self, worker_id: str) -> WorkerHarnessInfo | None:
        return self._workers.get(worker_id)

    def list_workers(self, capability_filter: str | None = None) -> list[WorkerHarnessInfo]:
        workers = list(self._workers.values())
        if capability_filter:
            return [w for w in workers if capability_filter in w.capabilities]
        return workers

    def delegate_subtask(
        self,
        worker_id: str,
        parent_task_id: str,
        session_id: str,
        prompt: str,
        *,
        executor_fn: Optional[Callable[[str], str]] = None,
        journal: Optional[ExecutionJournal] = None,
    ) -> DelegatedTaskHandoff:
        """Delegate a bounded subtask to a worker agent while preserving canonical lineage."""
        worker = self.get_worker(worker_id)
        if not worker:
            raise ValueError(f"Worker '{worker_id}' not found in registry")

        from uuid import uuid4
        handoff_id = f"handoff-{uuid4().hex[:8]}"

        handoff = DelegatedTaskHandoff(
            handoff_id=handoff_id,
            worker_id=worker_id,
            parent_task_id=parent_task_id,
            session_id=session_id,
            subtask_prompt=prompt,
        )
        self._handoffs[handoff_id] = handoff

        # Record in canonical ExecutionJournal
        if journal:
            journal.record(
                kind=ExecutionEventKind.TASK_STARTED,
                message=f"Delegated subtask to worker '{worker.name}' ({worker_id})",
                metadata={
                    "worker_id": worker_id,
                    "handoff_id": handoff_id,
                    "prompt": prompt[:200],
                },
            )

        if executor_fn:
            try:
                res = executor_fn(prompt)
                handoff.result = res
                handoff.success = True
                handoff.completed_at = _utc_now()
                if journal:
                    journal.record(
                        kind=ExecutionEventKind.ACTION,
                        message=f"Worker '{worker_id}' completed delegated subtask",
                        metadata={"handoff_id": handoff_id, "result_preview": str(res)[:200]},
                    )
            except Exception as exc:
                handoff.success = False
                handoff.result = str(exc)
                handoff.completed_at = _utc_now()
                if journal:
                    journal.record(
                        kind=ExecutionEventKind.ERROR,
                        message=f"Worker '{worker_id}' failed: {exc}",
                        metadata={"handoff_id": handoff_id},
                    )

        return handoff
