"""Reconciliation cost vs the work it could replace (#112734 §D / Phase 1A exit gate).

The exit gate demands "reconciliation overhead is materially below the work it could
eventually replace". The honest split, timed per capture:

- ``build_ms``: building the new shadow state — shared substrate, needed by the
  full-observation path too. Reported, not gated.
- ``extra_ms`` (diff + delta serialize): what reconciliation ADDS. This is the
  overhead the gate judges.
- ``full_observation_ms`` (build + full serialize): the paired in-process
  microbenchmark — what the delta path rides on.
- Replaced work: the fresh-capture round-trip on the real rig (~28ms, measured
  Xvfb :99 / cua-driver 0.28.2, four-arm experiment notes, #113287). The gate is
  ``extra_p95 < 50%`` of that baseline. On 4-element trees the diff itself
  dominates the in-process comparison (reported honestly as marginal overhead);
  against the real round-trip it is two orders of magnitude down.

Results aggregate per transition and persist as JSON so cost trends accumulate
across runs instead of living in one test's stdout.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from tools.computer_use.backend import UIElement
from tools.computer_use.semantic_fixtures import (
    FixtureTransition, SemanticFixture, W, H, element_list,
)
from tools.computer_use.semantic_state import build_state
from tools.computer_use.state_diff import (
    delta_bytes, diff_states, full_observation_bytes,
)

# Fresh-capture round-trip measured on the real rig (Xvfb :99, cua-driver 0.28.2;
# four-arm experiment notes, #113287): ~28ms per capture. That round-trip is the
# "work it could eventually replace" — the honest baseline for the Phase 1A exit
# gate. Defaults to the measured 28ms; override when the rig's number moves.
_REAL_RIG_CAPTURE_BASELINE_MS = 28.0


@dataclass(frozen=True)
class ReconciliationSample:
    """One timed reconciliation against one timed full-observation baseline."""
    fixture: str
    transition: str
    elements: int
    build_ms: float  # shared substrate: build the new shadow state
    extra_ms: float  # reconciliation overhead: diff + delta serialize
    full_ser_ms: float  # in-process replaced work: full-state serialize
    reconcile_ms: float  # build + diff + delta serialize
    full_observation_ms: float  # build + full serialize
    delta_bytes: int
    full_bytes: int
    changed_element_ratio: float

    @property
    def marginal_overhead_ratio(self) -> float:
        """Extra cost reconciliation adds, as a fraction of full-observation processing."""
        if self.full_observation_ms <= 0:
            return 0.0
        return (self.reconcile_ms - self.full_observation_ms) / self.full_observation_ms

    @property
    def byte_ratio(self) -> float:
        return self.delta_bytes / self.full_bytes if self.full_bytes else 0.0


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-pct * len(ordered) // 100)))
    return ordered[rank - 1]


def time_transition(fixture: SemanticFixture, transition: FixtureTransition,
                    repeats: int = 5) -> List[ReconciliationSample]:
    """Time the shadow path vs full-observation processing, ``repeats`` trials."""
    old_els, new_els = element_list(transition.old), element_list(transition.new)
    target = f"FixtureApp/{fixture.name}"
    samples: List[ReconciliationSample] = []
    for i in range(repeats):
        old_state = build_state(old_els, revision=2 * i + 1, target=target, width=W, height=H)

        t0 = time.perf_counter()
        new_state = build_state(new_els, revision=2 * i + 2, target=target, width=W, height=H)
        t_built = time.perf_counter()
        delta = diff_states(old_state, new_state)
        d_bytes = delta_bytes(delta)
        t_extra = time.perf_counter()
        build_ms = (t_built - t0) * 1000.0
        extra_ms = (t_extra - t_built) * 1000.0  # diff + delta serialize: the overhead

        t1 = time.perf_counter()
        full_state = build_state(new_els, revision=2 * i + 2, target=target, width=W, height=H)
        t_fbuilt = time.perf_counter()
        f_bytes = full_observation_bytes(full_state)
        t_full = time.perf_counter()
        full_build_ms = (t_fbuilt - t1) * 1000.0
        full_ser_ms = (t_full - t_fbuilt) * 1000.0

        samples.append(ReconciliationSample(
            fixture=fixture.name, transition=transition.name, elements=len(new_els),
            build_ms=build_ms, extra_ms=extra_ms, full_ser_ms=full_ser_ms,
            reconcile_ms=build_ms + extra_ms,
            full_observation_ms=full_build_ms + full_ser_ms,
            delta_bytes=d_bytes, full_bytes=f_bytes,
            changed_element_ratio=delta.changed_element_ratio,
        ))
    return samples


def scaled_list_fixture(n: int = 200) -> SemanticFixture:
    """Stress fixture: an n-element list with one flag flip — proves the overhead
    holds as trees grow, not just on 4-element toys."""
    def items(focused: bool):
        return (UIElement(index=1, role="AXWindow", label="Big", bounds=(0, 0, 800, 600),
                          app="FixtureApp", pid=1, window_id=1),) + tuple(
            UIElement(index=i + 2, role="AXStaticText", label=f"item-{i}",
                      bounds=(50, 100 + i * 4, 200, 24), app="FixtureApp", pid=1,
                      window_id=1, attributes={"focused": focused and i == 0})
            for i in range(n)
        )
    from tools.computer_use.semantic_fixtures import FixtureElement, FixtureTransition as FT
    old = tuple(FixtureElement(f"big:{i}", e) for i, e in enumerate(items(False)))
    new = tuple(FixtureElement(f"big:{i}", e) for i, e in enumerate(items(True)))
    return SemanticFixture(
        name=f"scaled-list-{n}",
        description=f"{n}-element stress tree, one flag flip",
        transitions=(FT(name="flag-flip", old=old, new=new,
                        expected_binding={f.fid: f.fid for f in new}),),
    )


@dataclass
class AggregateBenchReport:
    """Per-transition p50/p95 view plus the exit-gate verdict."""
    samples: int
    extra_p50_ms: float  # reconciliation overhead (diff + delta serialize)
    extra_p95_ms: float
    reconcile_p50_ms: float
    reconcile_p95_ms: float
    full_observation_p50_ms: float
    marginal_overhead_p50: float  # (reconcile - full) / full, informational
    marginal_overhead_max: float
    byte_ratio_p50: float
    changed_element_ratio_mean: float
    # The exit gate, as a number: reconciliation overhead p95 must stay under
    # half the replaced-work baseline (the real-rig capture round-trip).
    replaced_work_baseline_ms: float = _REAL_RIG_CAPTURE_BASELINE_MS
    per_transition: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def gate_passes(self) -> bool:
        return self.extra_p95_ms < 0.5 * self.replaced_work_baseline_ms

    def render(self) -> str:
        lines = [
            f"reconciliation bench: {self.samples} samples",
            f"  overhead (diff+delta-ser) p50={self.extra_p50_ms:.2f}ms "
            f"p95={self.extra_p95_ms:.2f}ms",
            f"  reconcile p50={self.reconcile_p50_ms:.2f}ms p95={self.reconcile_p95_ms:.2f}ms",
            f"  full-observation p50={self.full_observation_p50_ms:.2f}ms",
            f"  marginal overhead p50={self.marginal_overhead_p50:.2%} "
            f"max={self.marginal_overhead_max:.2%}",
            f"  delta/full bytes p50={self.byte_ratio_p50:.2%}",
            f"  gate: overhead p95 < 50% of replaced-work baseline "
            f"({self.replaced_work_baseline_ms:.0f}ms): "
            f"{'PASS' if self.gate_passes else 'FAIL'}",
        ]
        return "\n".join(lines)


def aggregate(samples: Sequence[ReconciliationSample],
              replaced_work_baseline_ms: float = _REAL_RIG_CAPTURE_BASELINE_MS,
              ) -> AggregateBenchReport:
    samples = list(samples)
    ratios = [s.marginal_overhead_ratio for s in samples]
    per_transition: Dict[str, Dict[str, float]] = {}
    for s in samples:
        key = f"{s.fixture}/{s.transition}"
        per_transition.setdefault(key, {"extra_ms": [], "reconcile_ms": []})
        per_transition[key]["extra_ms"].append(s.extra_ms)
        per_transition[key]["reconcile_ms"].append(s.reconcile_ms)
    summary = {k: {"extra_p50_ms": _percentile(v["extra_ms"], 50),
                   "reconcile_p50_ms": _percentile(v["reconcile_ms"], 50)}
               for k, v in per_transition.items()}
    return AggregateBenchReport(
        samples=len(samples),
        extra_p50_ms=_percentile([s.extra_ms for s in samples], 50),
        extra_p95_ms=_percentile([s.extra_ms for s in samples], 95),
        reconcile_p50_ms=_percentile([s.reconcile_ms for s in samples], 50),
        reconcile_p95_ms=_percentile([s.reconcile_ms for s in samples], 95),
        full_observation_p50_ms=_percentile([s.full_observation_ms for s in samples], 50),
        marginal_overhead_p50=_percentile(ratios, 50),
        marginal_overhead_max=max(ratios) if ratios else 0.0,
        byte_ratio_p50=_percentile([s.byte_ratio for s in samples], 50),
        changed_element_ratio_mean=(sum(s.changed_element_ratio for s in samples) / len(samples)) if samples else 0.0,
        replaced_work_baseline_ms=replaced_work_baseline_ms,
        per_transition=summary,
    )


def bench_all_fixtures(fixtures: Sequence[SemanticFixture], repeats: int = 5,
                       include_scaled: bool = True,
                       replaced_work_baseline_ms: float = _REAL_RIG_CAPTURE_BASELINE_MS,
                       ) -> AggregateBenchReport:
    fixtures = list(fixtures) + ([scaled_list_fixture()] if include_scaled else [])
    samples = [s for fx in fixtures for t in fx.transitions
               for s in time_transition(fx, t, repeats=repeats)]
    return aggregate(samples, replaced_work_baseline_ms=replaced_work_baseline_ms)


def save_report(report: AggregateBenchReport, path: Path) -> Path:
    """Persist the aggregate report as JSON — the trend history §J implies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["saved_schema"] = 3
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_report(path: Path) -> AggregateBenchReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload.pop("saved_schema", None)
    return AggregateBenchReport(**payload)
