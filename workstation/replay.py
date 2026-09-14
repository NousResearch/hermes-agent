"""Portable, side-effect-free execution trace and replay contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from workstation.journal import ExecutionJournal


_SENSITIVE_KEY = re.compile(r"token|secret|password|cookie|credential|authorization|api[_-]?key|otp|pin", re.I)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlsplit(url)
        if parsed.username or parsed.password:
            return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
        safe_query = [(key, "[REDACTED]") if _SENSITIVE_KEY.search(key) else (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)]
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(safe_query), ""))
    except ValueError:
        return None


@dataclass(slots=True)
class PortableTrace:
    trace_id: str
    task_id: str
    session_id: str
    events: list[dict[str, Any]]
    model_id: str | None = None
    provider: str | None = None
    policy_version: str | None = None
    parent_trace_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "events": self.events,
            "model_id": self.model_id,
            "provider": self.provider,
            "policy_version": self.policy_version,
            "parent_trace_id": self.parent_trace_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortableTrace":
        return cls(
            trace_id=str(data["trace_id"]),
            task_id=str(data["task_id"]),
            session_id=str(data["session_id"]),
            events=[dict(event) for event in data.get("events", []) if isinstance(event, dict)],
            model_id=data.get("model_id"),
            provider=data.get("provider"),
            policy_version=data.get("policy_version"),
            parent_trace_id=data.get("parent_trace_id"),
            created_at=str(data.get("created_at", datetime.now(timezone.utc).isoformat())),
        )

    def save(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)

    @classmethod
    def load(cls, path: Path) -> "PortableTrace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("portable trace must be a JSON object")
        return cls.from_dict(data)

    def fork_at(self, event_index: int, *, model_id: str, provider: str) -> "PortableTrace":
        if event_index < 0 or event_index > len(self.events):
            raise IndexError(event_index)
        return PortableTrace(
            trace_id=f"trace-{uuid4().hex}",
            task_id=self.task_id,
            session_id=self.session_id,
            events=[dict(event) for event in self.events[:event_index]],
            model_id=model_id,
            provider=provider,
            policy_version=self.policy_version,
            parent_trace_id=self.trace_id,
        )


class PortableTraceBuilder:
    @staticmethod
    def from_journal(
        journal: ExecutionJournal,
        *,
        model_id: str | None = None,
        provider: str | None = None,
        policy_version: str | None = None,
    ) -> PortableTrace:
        events = []
        for event in journal.read_events():
            payload = event.to_dict()
            payload["url"] = _safe_url(payload.get("url"))
            payload["metadata"] = _redact(payload.get("metadata", {}))
            payload.pop("evidence", None)
            events.append(payload)
        return PortableTrace(
            trace_id=f"trace-{uuid4().hex}",
            task_id=journal.task_id,
            session_id=journal.session_id,
            events=events,
            model_id=model_id,
            provider=provider,
            policy_version=policy_version,
        )


@dataclass(slots=True)
class ReplayReport:
    trace_id: str
    replayed_events: int
    side_effects_performed: bool = False
    errors: list[str] = field(default_factory=list)


class ReplayEngine:
    """Replays observation callbacks only; it never invokes tools or providers."""

    def replay(self, trace: PortableTrace, observer: Callable[[dict[str, Any]], Any] | None = None) -> ReplayReport:
        report = ReplayReport(trace_id=trace.trace_id, replayed_events=0)
        for event in trace.events:
            try:
                if observer:
                    observer(dict(event))
                report.replayed_events += 1
            except Exception as exc:
                report.errors.append(str(exc))
        return report
