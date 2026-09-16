"""Replayable synthetic computer-use workloads (RFC #112639, P0 slice).

Deterministic benchmark harness for the P0 exit gate and the P3 runahead
benchmark: scripted workloads against a fake CUA backend on a virtual clock,
so the same seed always produces the same action sequence and timing trace.
No production paths touched; nothing here is model-facing.

The long-stall fixture models the RFC's Blender example: an action whose
completion is delayed by a seeded, controllable multi-second stall. P3's
scheduler will run against its stall window to prove avoidable-stall reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import random

# Phase vocabulary mirrors the critical-path report taxonomy (#113225) so its
# spans can be consumed there directly.
PHASES = (
    "model", "admission", "approval_wait", "backend_resolve", "backend_start",
    "dispatch_lock_wait", "input", "capture", "backend_call", "backend_rebind",
    "capture_persist", "element_processing", "aux_vision", "response_shape",
    "total",
)


@dataclass
class Span:
    """One measured phase span; same shape as the critical-path report input."""

    task_id: str
    phase: str
    start_ms: float
    end_ms: float
    tool_call_id: str = ""
    session_id: str = ""
    attrs: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class WorkloadReport:
    """Everything a replay produced: spans, windows, and aggregate metrics."""

    task_id: str
    seed: int
    spans: List[Span]
    actions: List[str]  # action sequence, in order
    stall_windows: List[Tuple[float, float]]  # deterministic long stalls
    wall_ms: float
    phase_ms: Dict[str, float]
    avoidable_stall_ms: float  # stall time where nothing else ran (serial baseline)


class VirtualClock:
    """Deterministic clock; workloads advance virtual time instead of sleeping."""

    def __init__(self) -> None:
        self.now_ms = 0.0

    def sleep_ms(self, ms: float) -> None:
        self.now_ms += max(0.0, ms)


class SynthBackend:
    """Fake CUA backend with seeded per-phase latencies; records phase spans."""

    # Seeded latency ranges per phase, in ms. Small jitter keeps traces lively
    # while the seed keeps them exactly replayable.
    LATENCY_RANGES = {
        "backend_resolve": (1.0, 3.0),
        "capture": (180.0, 340.0),
        "input": (25.0, 60.0),
        "element_processing": (8.0, 30.0),
        "aux_vision": (1500.0, 3000.0),
        "response_shape": (20.0, 50.0),
        "capture_persist": (5.0, 20.0),
    }

    def __init__(self, rng: random.Random, clock: VirtualClock, record: Callable[[Span], None],
                 task_id: str, session_id: str) -> None:
        self._rng = rng
        self._clock = clock
        self._record = record
        self._task_id = task_id
        self._session_id = session_id
        self._tool_calls = 0
        self._render_until_ms: Optional[float] = None

    def _latency(self, phase: str) -> float:
        lo, hi = self.LATENCY_RANGES[phase]
        return self._rng.uniform(lo, hi)

    def _span(self, phase: str, attrs: Optional[Dict[str, str]] = None) -> None:
        # Every backend call emits resolve + the phase + response shaping.
        self._tool_calls += 1
        call_id = f"synth-{self._tool_calls}"
        start = self._clock.now_ms
        self._clock.sleep_ms(self._latency(phase))
        self._record(Span(task_id=self._task_id, phase=phase, start_ms=start,
                          end_ms=self._clock.now_ms, tool_call_id=call_id,
                          session_id=self._session_id, attrs=attrs or {}))

    def capture(self, **attrs: str) -> None:
        self._span("backend_resolve")
        self._span("capture", dict(attrs))
        self._span("element_processing")
        self._span("capture_persist")
        self._span("response_shape")

    def click(self, target: str) -> None:
        self._span("backend_resolve")
        self._span("input", {"action": "click", "target": target})
        self._span("response_shape")

    def type_text(self, target: str, text: str) -> None:
        self._span("backend_resolve")
        self._span("input", {"action": "type", "target": target,
                             "chars": str(len(text))})
        self._span("response_shape")

    def aux_vision(self, roi: str) -> None:
        self._span("backend_resolve")
        self._span("aux_vision", {"roi": roi})
        self._span("response_shape")

    def start_render(self, stall_ms: float) -> None:
        # The Blender-like fixture: the app accepts the render, then blocks.
        self._span("backend_resolve")
        self._span("input", {"action": "start_render"})
        self._span("response_shape")
        self._render_until_ms = self._clock.now_ms + stall_ms

    def poll_render(self) -> bool:
        # One blocking poll; the stall window is emitted as backend_call time.
        assert self._render_until_ms is not None, "no render in flight"
        start = self._clock.now_ms
        remaining = self._render_until_ms - start
        if remaining > 0:
            self._tool_calls += 1
            self._clock.sleep_ms(remaining)
            self._record(Span(task_id=self._task_id, phase="backend_call",
                              start_ms=start, end_ms=self._clock.now_ms,
                              tool_call_id=f"synth-{self._tool_calls}",
                              session_id=self._session_id,
                              attrs={"action": "render_stall", "stall": "true"}))
        return True


def form_fill_workload(backend: SynthBackend, log: List[str]) -> None:
    # Scripted form fill: observe, type three fields, observe, submit.
    backend.capture(view="form")
    log.append("capture:form")
    for name, value in (("name", "Ada"), ("email", "ada@example.com"),
                        ("notes", "deterministic")):
        backend.type_text(name, value)
        log.append(f"type:{name}")
    backend.capture(view="form_filled")
    log.append("capture:form_filled")
    backend.click("submit")
    log.append("click:submit")
    backend.capture(view="done")
    log.append("capture:done")


def dialog_dismiss_workload(backend: SynthBackend, log: List[str]) -> None:
    # Modal appears; dismiss it and confirm the underlying view is back.
    backend.capture(view="modal")
    log.append("capture:modal")
    backend.click("dismiss")
    log.append("click:dismiss")
    backend.capture(view="main")
    log.append("capture:main")


def long_stall_workload(backend: SynthBackend, log: List[str],
                        stall_ms: float) -> None:
    # Blender-like: start a render, sit in the deterministic stall, then act.
    backend.capture(view="editor")
    log.append("capture:editor")
    backend.start_render(stall_ms)
    log.append("start_render")
    backend.poll_render()
    log.append("render_complete")
    backend.capture(view="render_done")
    log.append("capture:render_done")
    backend.click("export")
    log.append("click:export")


Workload = Callable[[SynthBackend, List[str]], None]


def _stall_ms_for(seed: int) -> float:
    # Deterministic multi-second stall: 4s base + seed-derived 0-2s.
    return 4000.0 + (seed * 2654435761 % 2000)


def run_workload(workload: Workload, seed: int,
                 task_id: Optional[str] = None) -> WorkloadReport:
    """Replay one workload deterministically; same seed -> identical trace."""
    task_id = task_id or f"synth-{workload.__name__}-{seed}"
    session_id = f"session-{seed}"
    rng = random.Random(seed)
    clock = VirtualClock()
    spans: List[Span] = []
    actions: List[str] = []
    backend = SynthBackend(rng, clock, spans.append, task_id, session_id)

    start_ms = clock.now_ms
    if workload is long_stall_workload:
        workload(backend, actions, _stall_ms_for(seed))
    else:
        workload(backend, actions)
    wall_ms = clock.now_ms - start_ms

    stall_windows = [(s.start_ms, s.end_ms) for s in spans
                     if s.phase == "backend_call" and s.attrs.get("stall") == "true"]
    # Serial baseline: nothing else runs during the stall, so it is all avoidable.
    avoidable_stall_ms = sum(e - s for s, e in stall_windows)

    phase_ms: Dict[str, float] = {}
    for s in spans:
        phase_ms[s.phase] = phase_ms.get(s.phase, 0.0) + s.duration_ms
    phase_ms["total"] = wall_ms

    return WorkloadReport(task_id=task_id, seed=seed, spans=spans,
                          actions=actions, stall_windows=stall_windows,
                          wall_ms=wall_ms, phase_ms=phase_ms,
                          avoidable_stall_ms=avoidable_stall_ms)
