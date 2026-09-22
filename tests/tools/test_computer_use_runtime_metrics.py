"""Tests for the content-free computer-use telemetry seam."""

from __future__ import annotations

from types import SimpleNamespace

from agent import relay_runtime
from hermes_cli.observability.shared_metrics_contract import computer_use_phase_fields
from hermes_cli.observability import relay_shared_metrics
from tools.computer_use import tool as computer_use_tool
from tools.computer_use.backend import ActionResult
from tools.computer_use.runtime_metrics import ComputerUseTelemetry


def test_computer_use_phase_fields_are_bounded_and_content_free() -> None:
    fields = computer_use_phase_fields(
        {
            "action": "type",
            "phase": "input",
            "outcome": "success",
            "backend_kind": "cua",
            "backend_cache_hit": "hit",
            "backend_rebound": "rebound",
            "capture_mode": "vision",
            "target_changed": True,
            "element_count": 101,
            "image_bytes": 64 * 1024,
            "aux_vision_used": False,
            "prompt": "private prompt",
            "session_id": "private session",
        }
    )

    assert fields == {
        "action": "type",
        "phase": "input",
        "outcome": "success",
        "backend_kind": "cua",
        "backend_cache_hit": "hit",
        "backend_rebound": "rebound",
        "capture_mode": "vision",
        "target_state": "changed",
        "element_count_bucket": "gte_11",
        "image_size_bucket": "64kb_to_256kb",
        "aux_vision": "not_used",
    }
    assert "private" not in str(fields)


def test_computer_use_phase_fields_keep_backend_rebind_phase() -> None:
    assert computer_use_phase_fields({"phase": "backend_rebind"})["phase"] == "backend_rebind"


def test_computer_use_phase_fields_reject_unbounded_values() -> None:
    fields = computer_use_phase_fields(
        {
            "action": "run arbitrary command",
            "phase": "private phase",
            "outcome": "private outcome",
            "backend_kind": "private backend",
            "capture_mode": "private mode",
            "target_changed": "yes",
            "element_count": -1,
            "image_bytes": -1,
            "aux_vision_used": None,
        }
    )

    assert fields == {
        "action": "unknown",
        "phase": "unknown",
        "outcome": "unknown",
        "backend_kind": "unknown",
        "backend_cache_hit": "unknown",
        "backend_rebound": "unknown",
        "capture_mode": "unknown",
        "target_state": "unknown",
        "element_count_bucket": "unknown",
        "image_size_bucket": "unknown",
        "aux_vision": "unknown",
    }


def test_computer_use_telemetry_closes_total_and_phase_spans() -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.events: list[tuple[str, object, object]] = []

        def start_computer_use_phase(self, event):
            handle = len(self.events)
            self.events.append(("start", handle, event))
            return handle

        def finish_computer_use_phase(self, phase, event) -> None:
            self.events.append(("finish", phase.handle, event))

    runtime = _Runtime()
    telemetry = ComputerUseTelemetry("capture", session_id="session", task_id="task")
    telemetry._runtime = runtime

    with telemetry:
        telemetry.set_backend(SimpleNamespace(), cache_hit=True, rebound=False)
        telemetry.observe_capture(
            SimpleNamespace(
                mode="som",
                elements=[1, 2],
                png_bytes_len=128,
                app="private app",
                window_title="private title",
            )
        )
        with telemetry.phase("capture"):
            telemetry.set_aux_vision(True)

    assert [event[0] for event in runtime.events] == ["start", "start", "finish", "finish"]
    assert runtime.events[1][2]["phase"] == "capture"
    assert runtime.events[1][2]["backend_cache_hit"] == "hit"
    assert runtime.events[1][2]["backend_rebound"] == "not_rebound"
    assert runtime.events[2][2]["outcome"] == "success"
    assert runtime.events[3][2]["phase"] == "total"


def test_computer_use_action_emits_operation_phases(monkeypatch) -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.events: list[tuple[str, object, object]] = []

        def start_computer_use_phase(self, event):
            handle = len(self.events)
            self.events.append(("start", handle, event))
            return handle

        def finish_computer_use_phase(self, phase, event) -> None:
            self.events.append(("finish", phase.handle, event))

    class _Backend:
        def wait(self, seconds: float) -> ActionResult:
            return ActionResult(ok=True, action="wait", message=f"waited {seconds:.2f}s")

    runtime = _Runtime()
    monkeypatch.setattr(relay_runtime, "relay_instrumentation_enabled", lambda: True)
    monkeypatch.setattr(relay_shared_metrics, "get_computer_use_runtime", lambda: runtime)
    monkeypatch.setattr(computer_use_tool, "_get_backend", lambda session_id="": _Backend())

    result = computer_use_tool.handle_computer_use(
        {"action": "wait", "seconds": 0}, session_id="session", task_id="task"
    )

    assert result.startswith('{"ok": true')
    started = [event[2] for event in runtime.events if event[0] == "start"]
    assert [event["phase"] for event in started] == [
        "total", "admission", "backend_resolve", "dispatch_lock_wait", "backend_call"
    ]
    finished = [event[2] for event in runtime.events if event[0] == "finish"]
    assert [event["phase"] for event in finished] == [
        "admission", "backend_resolve", "dispatch_lock_wait", "backend_call", "total"
    ]
    assert finished[-1]["outcome"] == "success"
