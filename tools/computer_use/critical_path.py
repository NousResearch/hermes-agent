"""Local critical-path reconstruction for computer_use latency (RFC #112639, P0 slice).

This is the *analysis* half of the P0 instrumentation. The in-process phase-span
*recording* half is ``phase_spans.py`` (PR #112778's ``runtime_metrics.py`` is the
relay-forwarding counterpart): this module consumes phase spans keyed by the existing
observer/request/tool correlation IDs (``session_id`` / ``task_id`` / ``tool_call_id``,
as carried by the ``pre_tool_call`` / ``post_tool_call`` and ``pre_api_request`` /
``post_api_request`` lifecycle hooks) and reconstructs per-task critical paths locally,
with no Relay round trip.

Behavior-neutral by construction: nothing here touches the tool's execution path, records new
spans, or changes any schema. It only reads span lists and reports where the wall clock went.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

# Phase vocabulary from #112734 §C, shared with the recording half
# (``phase_spans.PHASES``). "model" is the whole model turn as seen by the
# existing pre/post_api_request hooks; queue/ttft/decode arrive later.
PHASES = (
    "total", "admission", "approval_wait", "screen_start", "backend_resolve",
    "backend_start", "backend_rebind", "dispatch_lock_wait", "capture", "input",
    "validate", "capture_persist", "element_processing", "aux_vision",
    "response_shape", "model",
)

# Phases that burn wall clock without doing work: the report calls these out as avoidable.
AVOIDABLE_IDLE_PHASES = frozenset({"approval_wait", "dispatch_lock_wait"})


@dataclass
class Span:
    """One measured phase span. Correlate via the IDs the lifecycle hooks already carry."""

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
class Segment:
    """One piece of the critical path: measured phase work, or idle between spans."""

    kind: str  # "phase" or "idle"
    phase: str  # phase name, or "idle" for gaps between spans
    start_ms: float
    end_ms: float
    tool_call_id: str = ""

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class TaskReport:
    task_id: str
    e2e_ms: float
    measured_ms: float  # on-path phase time; e2e - measured = unaccounted
    idle_ms: float  # gaps between spans: queueing, settle, scheduling
    avoidable_idle_ms: float  # idle + approval_wait + dispatch_lock_wait
    explained_pct: float  # measured / e2e
    phase_totals_ms: Dict[str, float]
    segments: List[Segment] = field(default_factory=list)
    phase_tokens_est: Dict[str, int] = field(default_factory=dict)  # per-phase, from span token attrs
    total_tokens_est: int = 0  # sum of phase_tokens_est


@dataclass
class AggregateReport:
    tasks: int
    e2e_p50_ms: float
    e2e_p95_ms: float
    explained_p50_pct: float
    avoidable_idle_total_ms: float
    per_task: List[TaskReport] = field(default_factory=list)
    total_tokens_est: int = 0  # sum of per-task totals


# Span attr keys that carry token estimates (ints, recorded via phase_spans dims).
_TOKEN_ATTRS = ("token_est", "image_token_est")


def _span_tokens(span: Span) -> int:
    """Token estimate a span carries: text and image estimates are distinct content."""
    total = 0
    for key in _TOKEN_ATTRS:
        value = span.attrs.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def _sweep(spans: Sequence[Span]) -> List[Segment]:
    """Reconstruct the critical path: sort by start, walk forward, never double-count.

    The computer_use tool path is serial per call, so the critical path is the union of
    span coverage. Gaps between spans become "idle" segments; overlapping spans (e.g. a
    "total" envelope over its sub-phases) add only their uncovered remainder. The "total"
    phase is an envelope, never on-path work, so it is skipped here.
    """
    ordered = sorted(
        (s for s in spans if s.phase != "total" and s.end_ms > s.start_ms),
        key=lambda s: (s.start_ms, s.end_ms),
    )
    segments: List[Segment] = []
    covered_until: float | None = None
    for s in ordered:
        if covered_until is None:
            covered_until = s.start_ms
        if s.start_ms > covered_until:
            segments.append(Segment("idle", "idle", covered_until, s.start_ms))
            covered_until = s.start_ms
        on_path = s.end_ms - covered_until
        if on_path > 0:
            segments.append(Segment("phase", s.phase, covered_until, s.end_ms, s.tool_call_id))
        covered_until = max(covered_until, s.end_ms)
    return segments


def build_task_report(spans: Sequence[Span], task_id: str) -> TaskReport:
    """Explain one task's end-to-end latency as a sum of measured phases."""
    spans = [s for s in spans if s.task_id == task_id]
    segments = _sweep(spans)
    measured = sum(s.duration_ms for s in segments if s.kind == "phase")
    idle = sum(s.duration_ms for s in segments if s.kind == "idle")
    avoidable = idle + sum(
        s.duration_ms for s in segments
        if s.kind == "phase" and s.phase in AVOIDABLE_IDLE_PHASES
    )
    phase_totals: Dict[str, float] = {}
    for s in segments:
        if s.kind == "phase":
            phase_totals[s.phase] = phase_totals.get(s.phase, 0.0) + s.duration_ms
    starts = [s.start_ms for s in spans]
    ends = [s.end_ms for s in spans]
    e2e = max(ends) - min(starts) if starts and ends else 0.0
    phase_tokens: Dict[str, int] = {}
    for s in spans:
        if s.phase == "total":  # envelope, never on-path work (same skip as _sweep)
            continue
        n = _span_tokens(s)
        if n:
            phase_tokens[s.phase] = phase_tokens.get(s.phase, 0) + n
    return TaskReport(
        task_id=task_id,
        e2e_ms=e2e,
        measured_ms=measured,
        idle_ms=idle,
        avoidable_idle_ms=avoidable,
        explained_pct=(100.0 * measured / e2e) if e2e > 0 else 0.0,
        phase_totals_ms=phase_totals,
        segments=segments,
        phase_tokens_est=phase_tokens,
        total_tokens_est=sum(phase_tokens.values()),
    )


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; deterministic on ties."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-pct * len(ordered) // 100)))
    return ordered[rank - 1]


def aggregate(reports: Sequence[TaskReport]) -> AggregateReport:
    """Exit-criterion view: p50/p95 E2E explained as a sum of measured phases."""
    reports = list(reports)
    return AggregateReport(
        tasks=len(reports),
        e2e_p50_ms=_percentile([r.e2e_ms for r in reports], 50),
        e2e_p95_ms=_percentile([r.e2e_ms for r in reports], 95),
        explained_p50_pct=_percentile([r.explained_pct for r in reports], 50),
        avoidable_idle_total_ms=sum(r.avoidable_idle_ms for r in reports),
        per_task=reports,
        total_tokens_est=sum(r.total_tokens_est for r in reports),
    )


def render(report: TaskReport | AggregateReport) -> str:
    """One-screen text summary of where the wall clock went."""
    if isinstance(report, AggregateReport):
        lines = [
            f"computer_use critical path: {report.tasks} tasks",
            f"  e2e p50={report.e2e_p50_ms:.0f}ms p95={report.e2e_p95_ms:.0f}ms",
            f"  explained p50={report.explained_p50_pct:.1f}% of e2e as measured phases",
            f"  avoidable idle total={report.avoidable_idle_total_ms:.0f}ms",
            f"  tokens_est total={report.total_tokens_est}",
        ]
        return "\n".join(lines)
    lines = [
        f"task {report.task_id}: e2e={report.e2e_ms:.0f}ms "
        f"measured={report.measured_ms:.0f}ms ({report.explained_pct:.1f}%) "
        f"idle={report.idle_ms:.0f}ms avoidable={report.avoidable_idle_ms:.0f}ms "
        f"tokens_est={report.total_tokens_est}",
    ]
    for phase, total in sorted(report.phase_totals_ms.items(), key=lambda kv: -kv[1]):
        line = f"  {phase}: {total:.0f}ms"
        if report.phase_tokens_est.get(phase):
            line += f" tokens_est={report.phase_tokens_est[phase]}"
        lines.append(line)
    return "\n".join(lines)
