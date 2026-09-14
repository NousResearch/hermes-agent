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

    def to_kanban_metadata(self) -> dict[str, Any]:
        return {
            "workstation": {
                "report_version": 1,
                "session_id": self.session_id,
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
