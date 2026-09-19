"""Content-free computer-use phase spans for the existing Hermes trace."""

from __future__ import annotations

import contextvars
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional


_CURRENT: contextvars.ContextVar[Optional["ComputerUseTelemetry"]] = contextvars.ContextVar(
    "computer_use_telemetry", default=None
)


@dataclass(frozen=True, slots=True)
class _PhaseHandle:
    runtime: Any
    handle: Any
    phase: str


def _observability_ids() -> tuple[str, str]:
    """Read the current tool correlation ids without exporting them."""
    try:
        from tools.approval_context import _approval_tool_call_id, _approval_turn_id

        return _approval_turn_id.get() or "", _approval_tool_call_id.get() or ""
    except Exception:
        return "", ""


def _safe_backend_kind(backend: Any) -> str:
    if backend is None:
        return "unknown"
    if type(backend).__name__ == "_NoopBackend":
        return "noop"
    return "cua" if "cua_backend" in getattr(type(backend), "__module__", "") else "other"


def _result_failed(result: Any) -> bool:
    if isinstance(result, dict):
        return result.get("ok") is False or "error" in result
    if not isinstance(result, str):
        return False
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and (payload.get("ok") is False or "error" in payload)


class ComputerUseTelemetry:
    """Open bounded phase spans when shared metrics are enabled.

    The runtime is resolved once per computer-use call. Every failure in this
    observer is swallowed so instrumentation can never change tool behavior.
    """

    def __init__(self, action: str, *, session_id: str = "", task_id: str = "") -> None:
        turn_id, tool_call_id = _observability_ids()
        self.action = action
        self.session_id = session_id or ""
        self.task_id = task_id or ""
        self.turn_id = turn_id
        self.tool_call_id = tool_call_id
        self._runtime: Any = _UNSET
        self._token: Any = None
        self._total: Optional[_PhaseHandle] = None
        self._outcome: Optional[str] = None
        self._backend_kind = "unknown"
        self._backend_cache_hit: Optional[str] = None
        self._backend_rebound: Optional[str] = None
        self._capture_mode: Optional[str] = None
        self._target_changed: Optional[bool] = None
        self._element_count: Optional[int] = None
        self._image_bytes: Optional[int] = None
        self._aux_vision_used: Optional[bool] = None
        self._last_capture: Any = None
        self._last_target: Optional[tuple[str, str]] = None

    def __enter__(self) -> "ComputerUseTelemetry":
        self._token = _CURRENT.set(self)
        self._total = self._start("total")
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> bool:
        if exc_type is not None:
            self.set_outcome("failed")
        elif self._outcome is None:
            self.set_outcome("success")
        self._finish(self._total, self._outcome or "unknown")
        if self._token is not None:
            _CURRENT.reset(self._token)
        return False

    def _runtime_instance(self) -> Any:
        if self._runtime is not _UNSET:
            return self._runtime
        self._runtime = None
        if not self.task_id:
            return None
        try:
            from agent import relay_runtime
            from hermes_cli.observability import relay_shared_metrics

            if not relay_runtime.relay_instrumentation_enabled():
                return None
            self._runtime = relay_shared_metrics.get_computer_use_runtime()
        except Exception:
            self._runtime = None
        return self._runtime

    def _event(self, phase_name: str, outcome: str) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "turn_id": self.turn_id,
            "tool_call_id": self.tool_call_id,
            "action": self.action,
            "phase": phase_name,
            "outcome": outcome,
            "backend_kind": self._backend_kind,
            "backend_cache_hit": self._backend_cache_hit,
            "backend_rebound": self._backend_rebound,
            "capture_mode": self._capture_mode,
            "target_changed": self._target_changed,
            "element_count": self._element_count,
            "image_bytes": self._image_bytes,
            "aux_vision_used": self._aux_vision_used,
        }

    def _start(self, phase_name: str) -> Optional[_PhaseHandle]:
        runtime = self._runtime_instance()
        if runtime is None:
            return None
        try:
            handle = runtime.start_computer_use_phase(self._event(phase_name, "unknown"))
            return _PhaseHandle(runtime, handle, phase_name) if handle is not None else None
        except Exception:
            return None

    def _finish(self, phase: Optional[_PhaseHandle], outcome: str) -> None:
        if phase is None:
            return
        try:
            phase.runtime.finish_computer_use_phase(phase, self._event(phase.phase, outcome))
        except Exception:
            pass

    @contextmanager
    def phase(self, phase_name: str) -> Iterator[None]:
        phase = self._start(phase_name)
        try:
            yield
        except BaseException:
            self.set_outcome("failed")
            self._finish(phase, "failed")
            raise
        else:
            self._finish(phase, self._outcome or "success")

    def set_outcome(self, outcome: str) -> None:
        if outcome in {"blocked", "failed", "success", "unavailable"}:
            self._outcome = outcome

    def set_result(self, result: Any) -> None:
        self.set_outcome("failed" if _result_failed(result) else "success")

    def set_backend(
        self, backend: Any, *, cache_hit: Optional[bool] = None, rebound: Optional[bool] = None
    ) -> None:
        self._backend_kind = _safe_backend_kind(backend)
        if cache_hit is not None:
            self._backend_cache_hit = "hit" if cache_hit else "miss"
        if rebound is not None:
            self._backend_rebound = "rebound" if rebound else "not_rebound"

    def mark_backend_rebound(self) -> None:
        self._backend_rebound = "rebound"

    def observe_capture(self, capture: Any) -> None:
        if capture is self._last_capture:
            return
        self._last_capture = capture
        self._capture_mode = getattr(capture, "mode", None)
        elements = getattr(capture, "elements", None)
        self._element_count = len(elements) if isinstance(elements, list) else None
        raw_size = getattr(capture, "png_bytes_len", None)
        self._image_bytes = raw_size if isinstance(raw_size, int) and raw_size >= 0 else None
        target = (str(getattr(capture, "app", "") or ""), str(getattr(capture, "window_title", "") or ""))
        self._target_changed = None if self._last_target is None else target != self._last_target
        self._last_target = target

    def set_aux_vision(self, used: bool) -> None:
        self._aux_vision_used = bool(used)


_UNSET = object()


def current_telemetry() -> Optional[ComputerUseTelemetry]:
    return _CURRENT.get()


@contextmanager
def phase(phase_name: str) -> Iterator[None]:
    telemetry = current_telemetry()
    if telemetry is None:
        yield
        return
    with telemetry.phase(phase_name):
        yield


def observe_backend(
    backend: Any, *, cache_hit: Optional[bool] = None, rebound: Optional[bool] = None
) -> None:
    telemetry = current_telemetry()
    if telemetry is not None:
        telemetry.set_backend(backend, cache_hit=cache_hit, rebound=rebound)


def mark_backend_rebound() -> None:
    telemetry = current_telemetry()
    if telemetry is not None:
        telemetry.mark_backend_rebound()


def observe_capture(capture: Any) -> None:
    telemetry = current_telemetry()
    if telemetry is not None:
        telemetry.observe_capture(capture)


def set_aux_vision(used: bool) -> None:
    telemetry = current_telemetry()
    if telemetry is not None:
        telemetry.set_aux_vision(used)


__all__ = [
    "ComputerUseTelemetry", "current_telemetry", "mark_backend_rebound", "observe_backend", "observe_capture",
    "phase", "set_aux_vision",
]
