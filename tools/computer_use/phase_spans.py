"""In-process phase-span recording for computer_use (RFC #112734, Phase 0A).

The *recording* half of the P0 instrumentation: ``critical_path.py`` is the analysis
half and consumes the ``Span`` objects produced here, so a fixture or a real task can
answer "where did the time go" without reconstructing it from logs.

Spans are content-free: phase names plus a fixed allowlist of scalar dimensions. No
screenshot bytes, element text, typed values, credentials, or PII ever enter a span.
Recording is gated on ``relay_instrumentation_enabled`` — outside an opted-in turn the
recorder is a no-op and the tool path is untouched. Spans never leave the process;
the relay-runtime forwarding half is #112778's lane, not this module's.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from tools.computer_use.critical_path import Span, build_task_report

# Phase vocabulary from #112734 §C. ``screen_start`` is reserved for #108914's Bot
# Screen desktop startup; ``model`` for the pre/post_api_request hooks — neither is
# wired here yet, but the report already understands them.
PHASES = (
    "total",
    "admission",
    "approval_wait",
    "screen_start",
    "backend_resolve",
    "backend_start",
    "backend_rebind",
    "dispatch_lock_wait",
    "capture",
    "input",
    "validate",
    "capture_persist",
    "element_processing",
    "aux_vision",
    "response_shape",
    "model",
)

# The only dimensions a span may carry. Anything else is dropped: telemetry must
# never become a side channel for screen content.
_DIMENSIONS = frozenset({
    "action",
    "capture_mode",
    "backend_kind",
    "backend_cache_hit",
    "backend_rebound",
    "display_changed",
    "revision_invalidated",
    "invalidation_reason",
    "element_count",
    "capture_bytes",
    "token_est",  # text token estimate for this phase's content (token_estimates.py)
    "image_token_est",  # screenshot tile-grid estimate (token_estimates.py)
    "aux_vision_used",
    "outcome",
})


def _coerce(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value)
    return text[:120]  # dimensions are labels, never content


class PhaseRecorder:
    """Records one computer_use call's phase spans; no-op unless instrumented."""

    def __init__(self, action: str = "", session_id: str = "",
                 task_id: str = "", tool_call_id: str = "") -> None:
        self._action = action
        self._session_id = session_id
        self._task_id = task_id
        self._tool_call_id = tool_call_id
        self._t0 = time.monotonic()
        self._spans: List[Span] = []
        self._open: Optional[Dict[str, Any]] = None
        self._enabled: Optional[bool] = None

    @property
    def enabled(self) -> bool:
        """True inside an opted-in instrumentation turn; cached per recorder."""
        if self._enabled is None:
            try:
                from agent import relay_runtime
                self._enabled = bool(relay_runtime.relay_instrumentation_enabled())
            except Exception:
                self._enabled = False
        return self._enabled

    def _now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time one phase; silently a no-op when instrumentation is off."""
        if not self.enabled:
            yield
            return
        attrs: Dict[str, Any] = {"action": self._action}
        span = Span(task_id=self._task_id, phase=name, start_ms=self._now_ms(),
                    end_ms=self._now_ms(), tool_call_id=self._tool_call_id,
                    session_id=self._session_id, attrs=attrs)
        prev, self._open = self._open, attrs
        try:
            yield
        except BaseException:
            attrs["outcome"] = "failed"
            raise
        else:
            attrs.setdefault("outcome", "success")
        finally:
            span.end_ms = self._now_ms()
            self._open = prev
            self._spans.append(span)

    def set(self, dimension: str, value: Any) -> None:
        """Attach a dimension to the open phase (dropped when off or unknown)."""
        if not self.enabled or dimension not in _DIMENSIONS or self._open is None:
            return
        coerced = _coerce(value)
        if coerced is not None:
            self._open[dimension] = coerced

    def note_invalidation(self, reason: Optional[str]) -> None:
        """Record a failed revision validation on the open (validate) phase."""
        if not self.enabled or self._open is None:
            return
        self._open["revision_invalidated"] = True
        if reason:
            self._open["invalidation_reason"] = _coerce(reason)

    @property
    def spans(self) -> List[Span]:
        return list(self._spans)

    def drain(self) -> List[Span]:
        """Take the recorded spans, clearing the buffer (report handoff)."""
        spans, self._spans = self._spans, []
        return spans

    def task_report(self, task_id: str = ""):
        """Critical-path report over this recorder's spans (the Phase 0A exit gate)."""
        return build_task_report(self._spans, task_id or self._task_id)


_CURRENT: contextvars.ContextVar[Optional[PhaseRecorder]] = contextvars.ContextVar(
    "computer_use_phase_recorder", default=None)


def current_recorder() -> Optional[PhaseRecorder]:
    """The ambient recorder for this computer_use call, if instrumentation is on."""
    rec = _CURRENT.get()
    return rec if rec is not None and rec.enabled else None


def _set_current_recorder(rec: Optional[PhaseRecorder]):
    return _CURRENT.set(rec)


def _reset_current_recorder(token) -> None:
    _CURRENT.reset(token)


@contextmanager
def record_call(action: str, session_id: str = "",
                task_id: str = "", tool_call_id: str = "") -> Iterator[PhaseRecorder]:
    """Ambient recorder for one ``handle_computer_use`` invocation."""
    rec = PhaseRecorder(action, session_id=session_id, task_id=task_id,
                        tool_call_id=tool_call_id)
    token = _set_current_recorder(rec)
    try:
        yield rec
    finally:
        _reset_current_recorder(token)


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Time ``name`` on the ambient recorder; no-op without one."""
    rec = current_recorder()
    if rec is None:
        yield
        return
    with rec.phase(name):
        yield


def set_dimension(dimension: str, value: Any) -> None:
    """Attach a dimension to the ambient recorder's open phase, if any."""
    rec = current_recorder()
    if rec is not None:
        rec.set(dimension, value)


def note_invalidation(reason: Optional[str]) -> None:
    """Record a revision invalidation on the ambient recorder's open phase."""
    rec = current_recorder()
    if rec is not None:
        rec.note_invalidation(reason)
