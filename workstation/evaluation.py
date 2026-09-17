"""Model-independent runtime evaluation and regression-budget contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import os
from typing import Any, Callable

from workstation.contracts import ExecutionEventKind
from workstation.journal import ExecutionJournal


@dataclass(slots=True)
class EvaluationCase:
    name: str
    input: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    case_name: str
    success: bool
    actions: int
    latency_seconds: float
    tokens: int = 0
    cost_usd: float = 0.0
    safety_ok: bool = True
    model_independent: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RegressionBudget:
    max_action_increase: int = 0
    max_latency_increase: float = 0.0
    max_token_increase: int = 0
    max_cost_increase: float = 0.0


@dataclass(slots=True)
class RegressionComparison:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    model_independent: bool = True


class EvaluationHarness:
    def __init__(self, journal: ExecutionJournal | None = None) -> None:
        self.journal = journal

    def outcome_metrics(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Local AVCR projection. Missing provenance is never production data."""
        eligible = [r for r in runs if r.get("environment") in {"production", "dogfood"}
                    and r.get("eligible_delegated") is True]
        verified = [r for r in eligible if r.get("outcome_status") == "verified_completed"
                    and r.get("acceptance_approved") is True]
        autonomous = [r for r in verified if not r.get("unplanned_human_rescues")]
        count, complete = len(eligible), len(verified)
        ratio = lambda numerator, denominator: numerator / denominator if denominator and numerator is not None else None
        known_sum = lambda key: sum(r[key] for r in eligible) if eligible and all(r.get(key) is not None for r in eligible) else None
        recoveries = [r for r in eligible if r.get("recovery_attempted") is True]
        return {
            "eligible_delegated_tasks": count, "verified_completion": complete,
            "AVCR": ratio(len(autonomous), count),
            "false_completion_rate": ratio(sum(r.get("done") is True and r not in verified for r in eligible), count),
            "zombie_running_rate": ratio(sum(r.get("running") is True and not r.get("live_handles") for r in eligible), count),
            "planned_handoff_rate": ratio(sum(bool(r.get("planned_handoffs")) for r in eligible), count),
            "unplanned_rescue_rate": ratio(sum(bool(r.get("unplanned_human_rescues")) for r in eligible), count),
            "systemic_failure_amplification": sum(r.get("systemic_failure_amplification", 0) for r in eligible),
            "tokens_per_verified_task": ratio(known_sum("tokens"), complete),
            "cost_per_verified_task": ratio(known_sum("cost_usd"), complete),
            "time_per_verified_task": ratio(known_sum("duration_seconds"), complete),
            "tool_calls_per_verified_task": ratio(known_sum("tool_calls"), complete),
            "recovery_success_rate": ratio(sum(r.get("recovery_success") is True for r in recoveries), len(recoveries)),
            "evidence_coverage": ratio(sum(r.get("verified_evidence_count", 0) for r in eligible), sum(r.get("required_evidence_count", 0) for r in eligible)),
            **{key: sum(r.get(key, 0) for r in eligible) for key in (
                "model_interventions", "tool_calls", "no_progress_calls", "duplicate_calls_blocked", "routine_candidate")},
            "routine_reuse": sum(r.get("routine_reuse", 0) for r in eligible),
            "routine_savings": sum(r.get("routine_savings", 0) for r in eligible),
        }

    def run(self, case: EvaluationCase, executor: Callable[[Any], dict[str, Any]]) -> EvaluationResult:
        started = time.monotonic()
        try:
            result = executor(case.input)
            elapsed = time.monotonic() - started
            evaluation = EvaluationResult(
                case_name=case.name,
                success=bool(result.get("success", False)),
                actions=max(0, int(result.get("actions", 0))),
                latency_seconds=float(result.get("latency_seconds", elapsed)),
                tokens=max(0, int(result.get("tokens", 0))),
                cost_usd=max(0.0, float(result.get("cost_usd", 0.0))),
                safety_ok=bool(result.get("safety_ok", True)),
                metadata=dict(result.get("metadata", {})),
            )
            evaluation.metadata.update({k: case.metadata[k] for k in (
                "environment", "build_sha", "workstation_version", "test_case_id", "evaluation_run_id",
                "task_id", "session_id") if k in case.metadata})
            evaluation.metadata.setdefault("environment", "test" if os.environ.get("HERMES_TEST_ISOLATION") else "benchmark")
            self._record(evaluation)
            return evaluation
        except Exception as exc:
            evaluation = EvaluationResult(case.name, False, 0, time.monotonic() - started, metadata={"error": str(exc)})
            self._record(evaluation)
            return evaluation

    def _record(self, result: EvaluationResult) -> None:
        if self.journal is not None:
            self.journal.record(
                ExecutionEventKind.ACTION,
                f"evaluation:{result.case_name}",
                metadata={
                    "success": result.success,
                    "actions": result.actions,
                    "latency_seconds": result.latency_seconds,
                    "tokens": result.tokens,
                    "cost_usd": result.cost_usd,
                    "safety_ok": result.safety_ok,
                    **result.metadata,
                },
            )

    def compare(
        self,
        baseline: EvaluationResult,
        candidate: EvaluationResult,
        budget: RegressionBudget,
    ) -> RegressionComparison:
        reasons: list[str] = []
        if not candidate.success:
            reasons.append("candidate failed")
        if not candidate.safety_ok:
            reasons.append("candidate safety guard failed")
        if candidate.actions - baseline.actions > budget.max_action_increase:
            reasons.append("action regression budget exceeded")
        if candidate.latency_seconds - baseline.latency_seconds > budget.max_latency_increase:
            reasons.append("latency regression budget exceeded")
        if candidate.tokens - baseline.tokens > budget.max_token_increase:
            reasons.append("token regression budget exceeded")
        if candidate.cost_usd - baseline.cost_usd > budget.max_cost_increase:
            reasons.append("cost regression budget exceeded")
        return RegressionComparison(not reasons, reasons)


@dataclass(slots=True)
class SoakResult:
    iterations: int
    failures: int
    success_rate: float
    total_actions: int
    total_latency_seconds: float
    errors: list[str] = field(default_factory=list)
    completed_iterations: int = 0
    duration_seconds: float = 0.0
    timed_out: bool = False


class SoakRunner:
    def run(
        self,
        iterations: int,
        executor: Callable[[int], dict[str, Any]],
        *,
        max_duration_seconds: float | None = None,
    ) -> SoakResult:
        if max_duration_seconds is not None and max_duration_seconds < 0:
            raise ValueError("max_duration_seconds must be non-negative")
        started = time.monotonic()
        failures = 0
        actions = 0
        latency = 0.0
        errors: list[str] = []
        completed = 0
        timed_out = False
        for index in range(max(0, iterations)):
            if max_duration_seconds is not None and time.monotonic() - started >= max_duration_seconds:
                timed_out = True
                break
            try:
                result = executor(index)
                if not bool(result.get("success", False)):
                    failures += 1
                actions += max(0, int(result.get("actions", 0)))
                latency += max(0.0, float(result.get("latency_seconds", 0.0)))
            except Exception as exc:
                failures += 1
                errors.append(str(exc))
            completed += 1
        count = max(0, iterations)
        if max_duration_seconds is not None and not timed_out and time.monotonic() - started >= max_duration_seconds:
            timed_out = completed < count
        return SoakResult(
            count,
            failures,
            (completed - failures) / completed if completed else 1.0,
            actions,
            latency,
            errors,
            completed,
            max(0.0, time.monotonic() - started),
            timed_out,
        )

    def run_for_duration(
        self,
        duration_seconds: float,
        executor: Callable[[int], dict[str, Any]],
        *,
        max_iterations: int = 1_000_000,
    ) -> SoakResult:
        """Run a bounded duration soak while retaining a hard iteration cap."""

        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        return self.run(
            max_iterations,
            executor,
            max_duration_seconds=duration_seconds,
        )
