from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Callable, List, Optional
from uuid import uuid4

from workstation.contracts import ExecutionEventKind, RiskLevel, utc_now
from workstation.journal import ExecutionJournal
from workstation.kanban import WorkstationKanbanBridge

_log = logging.getLogger(__name__)


class SystemEventType(str, Enum):
    PROCESS_CRASH = "process_crash"
    BUILD_FAILURE = "build_failure"
    BUILD_SUCCESS = "build_success"
    DOWNLOAD_COMPLETED = "download_completed"
    REPO_STATE_CHANGED = "repo_state_changed"
    CONTROLLER_HEALTH_CHANGED = "controller_health_changed"
    LONG_RUNNING_JOB_COMPLETED = "long_running_job_completed"
    USER_ATTENTION_REQUIRED = "user_attention_required"


class SystemEventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class SystemEvent:
    event_id: str
    event_type: SystemEventType
    severity: SystemEventSeverity
    source: str
    title: str
    message: str
    timestamp: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    target_task_id: Optional[str] = None
    target_session_id: Optional[str] = None
    processed: bool = False
    resulting_task_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data


class SystemEventPipeline:
    """Surfaces meaningful host/system events back into the canonical Hermes task model.

    Retains provenance, reason, and evidence without introducing competing schedulers or stores.
    """

    def __init__(self, kanban_bridge: Optional[WorkstationKanbanBridge] = None) -> None:
        self.kanban_bridge = kanban_bridge or WorkstationKanbanBridge()
        self._recent_events: list[SystemEvent] = []
        self._handlers: list[Callable[[SystemEvent], None]] = []

    def register_handler(self, handler: Callable[[SystemEvent], None]) -> None:
        self._handlers.append(handler)

    def get_recent_events(self, limit: int = 50) -> list[SystemEvent]:
        return self._recent_events[-limit:]

    def ingest_event(
        self,
        event_type: SystemEventType,
        severity: SystemEventSeverity,
        source: str,
        title: str,
        message: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
        target_task_id: Optional[str] = None,
        target_session_id: Optional[str] = None,
    ) -> SystemEvent:
        event = SystemEvent(
            event_id=f"sysevt-{uuid4().hex[:8]}",
            event_type=event_type,
            severity=severity,
            source=source,
            title=title,
            message=message,
            metadata=metadata or {},
            target_task_id=target_task_id,
            target_session_id=target_session_id,
        )
        return self.process_event(event)

    def process_event(self, event: SystemEvent) -> SystemEvent:
        """Process an ingested system event according to canonical task and policy boundaries."""
        self._recent_events.append(event)

        # Notify custom subscribers/listeners
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as exc:
                _log.warning("Error in event handler %s: %s", handler, exc)

        session_id = event.target_session_id or "system-event-bus"

        # Case 1: Event enriches an existing Hermes task
        if event.target_task_id:
            try:
                journal = ExecutionJournal(event.target_task_id, session_id)
                kind = (
                    ExecutionEventKind.ERROR
                    if event.severity in (SystemEventSeverity.ERROR, SystemEventSeverity.CRITICAL)
                    else ExecutionEventKind.ACTION
                )
                risk = (
                    RiskLevel.HIGH
                    if event.severity == SystemEventSeverity.CRITICAL
                    else RiskLevel.LOW
                )
                journal.record(
                    kind=kind,
                    message=f"[System Event: {event.event_type.value}] {event.title}: {event.message}",
                    risk=risk,
                    metadata={
                        "event_id": event.event_id,
                        "source": event.source,
                        "severity": event.severity.value,
                        **event.metadata,
                    },
                )
            except Exception as exc:
                _log.warning("Could not append event to journal for task %s: %s", event.target_task_id, exc)

            event.processed = True
            return event

        # Uncorrelated system events are observations, never new work intent.
        # Subscribers may correlate/wake an existing task or request human triage.
        event.processed = True
        return event
