from __future__ import annotations

import pytest

from workstation.protocols import ExternalProtocolAdapter, ProtocolKind


def test_external_protocol_maps_to_canonical_runtime_event() -> None:
    adapter = ExternalProtocolAdapter(ProtocolKind.A2A, ("1",))
    event = adapter.to_runtime_event(
        {
            "version": "1",
            "type": "worker.progress",
            "task_id": "task-1",
            "session_id": "session-1",
            "payload": {"progress": 0.5},
        }
    )
    assert event.type == "worker.progress"
    assert event.task_id == "task-1"
    assert event.payload["protocol"] == "a2a"
    assert event.payload["progress"] == 0.5


def test_external_protocol_rejects_missing_or_unknown_version() -> None:
    adapter = ExternalProtocolAdapter(ProtocolKind.ACP, ("2026-01",))
    with pytest.raises(ValueError, match="unsupported"):
        adapter.to_runtime_event({"type": "x", "task_id": "t", "session_id": "s", "payload": {}})
    with pytest.raises(ValueError, match="unsupported"):
        adapter.to_runtime_event(
            {"version": "old", "type": "x", "task_id": "t", "session_id": "s", "payload": {}}
        )
