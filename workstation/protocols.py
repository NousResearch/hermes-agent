"""Versioned adapters for external agent/client protocols.

External protocols are deliberately translated into the canonical Workstation
event contract; they do not become task, session, or resource owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from workstation.runtime import RuntimeEvent


class ProtocolKind(str, Enum):
    A2A = "a2a"
    ACP = "acp"
    UHP = "uhp"


@dataclass(slots=True)
class ExternalProtocolAdapter:
    protocol: ProtocolKind
    supported_versions: tuple[str, ...]

    def to_runtime_event(self, envelope: Mapping[str, Any]) -> RuntimeEvent:
        version = str(envelope.get("version", ""))
        if not version or version not in self.supported_versions:
            raise ValueError(f"unsupported {self.protocol.value} version: {version or 'missing'}")
        event_type = str(envelope.get("type", "")).strip()
        task_id = str(envelope.get("task_id", "")).strip()
        session_id = str(envelope.get("session_id", "")).strip()
        payload = envelope.get("payload")
        if not event_type or not task_id or not session_id or not isinstance(payload, dict):
            raise ValueError("protocol envelope requires type, task_id, session_id and object payload")
        return RuntimeEvent(
            type=event_type,
            task_id=task_id,
            session_id=session_id,
            payload={
                "protocol": self.protocol.value,
                "version": version,
                **payload,
            },
        )
