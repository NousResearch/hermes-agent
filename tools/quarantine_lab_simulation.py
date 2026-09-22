"""Simulation-only sink and orchestration for temporary LAB approval."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tools.quarantine_dev_context import (
    DevHumanApprovalContext,
    validate_dev_approval_context,
)
from tools.quarantine_dev_human_approval import (
    DevHumanApprovalResult,
    ForegroundReviewConsole,
    recheck_dev_binding,
    record_dev_approval_state,
    request_dev_human_approval,
)

ALLOW_TO_LAB_SIMULATION = "ALLOW_TO_LAB_SIMULATION"


@dataclass(frozen=True)
class LabSimulationSinkResult:
    simulated: bool
    outcome: str
    candidate_sha256: str
    evidence_digest: str


class LabQuarantineSimulationSink:
    """No-write candidate sink: returns metadata only and never executes bytes."""

    def simulate(
        self,
        external_result: Any,
        context: DevHumanApprovalContext,
    ) -> LabSimulationSinkResult:
        checked = validate_dev_approval_context(context)
        if not checked.valid:
            return LabSimulationSinkResult(False, checked.reason, "", "")
        binding = getattr(external_result, "approval_binding", None)
        if binding is None:
            return LabSimulationSinkResult(False, "dev_approval_binding_missing", "", "")
        return LabSimulationSinkResult(
            True,
            ALLOW_TO_LAB_SIMULATION,
            str(binding.candidate_sha256),
            str(binding.evidence_digest),
        )


@dataclass(frozen=True)
class LabQuarantineSimulationResult:
    allowed: bool
    outcome: str
    reason: str
    approval: DevHumanApprovalResult


def run_lab_quarantine_simulation(
    external_result: Any,
    candidate_path: Path,
    context: DevHumanApprovalContext,
    console: ForegroundReviewConsole,
    *,
    final_confirm: Callable[[str], bool],
    source_id: str = "",
    source_version: str = "",
    install_attempt_id: str | None = None,
    audit_root: Path | None = None,
    now: datetime | None = None,
    monotonic: Any = None,
    timeout_seconds: float = 120.0,
    sink: LabQuarantineSimulationSink | None = None,
) -> LabQuarantineSimulationResult:
    approval_kwargs: dict[str, Any] = {
        "source_id": source_id,
        "source_version": source_version,
        "install_attempt_id": install_attempt_id,
        "audit_root": audit_root,
        "now": now,
        "timeout_seconds": timeout_seconds,
    }
    if monotonic is not None:
        approval_kwargs["monotonic"] = monotonic

    approval = request_dev_human_approval(
        external_result,
        Path(candidate_path),
        context,
        console,
        **approval_kwargs,
    )
    if not approval.approved:
        return LabQuarantineSimulationResult(
            False, "LAB_SIMULATION_REJECTED", approval.reason, approval
        )

    final_summary = (
        "DEVELOPMENT LAB FINAL CONFIRMATION — simulation only; "
        "candidate will not be installed or executed."
    )
    if not bool(final_confirm(final_summary)):
        record_dev_approval_state(
            external_result,
            approval,
            "final_confirmation_cancelled",
            result="DEV_FINAL_CONFIRMATION_CANCELLED",
            now=now or datetime.now(timezone.utc),
        )
        return LabQuarantineSimulationResult(
            False,
            "LAB_SIMULATION_CANCELLED",
            "DEV_FINAL_CONFIRMATION_CANCELLED",
            approval,
        )

    stable = recheck_dev_binding(
        external_result,
        Path(candidate_path),
        context,
        source_id=source_id,
        source_version=source_version,
    )
    if not stable.valid:
        record_dev_approval_state(
            external_result,
            approval,
            "binding_drift_rejected",
            result="DEV_APPROVAL_BINDING_DRIFT_REJECTED",
            now=now or datetime.now(timezone.utc),
        )
        return LabQuarantineSimulationResult(
            False,
            "LAB_SIMULATION_REJECTED",
            "DEV_APPROVAL_BINDING_DRIFT_REJECTED",
            approval,
        )

    active_sink = sink or LabQuarantineSimulationSink()
    simulated = active_sink.simulate(external_result, context)
    if not simulated.simulated or simulated.outcome != ALLOW_TO_LAB_SIMULATION:
        return LabQuarantineSimulationResult(
            False, "LAB_SIMULATION_REJECTED", simulated.outcome, approval
        )

    record_dev_approval_state(
        external_result,
        approval,
        "simulation_completed",
        result=ALLOW_TO_LAB_SIMULATION,
        now=now or datetime.now(timezone.utc),
    )
    return LabQuarantineSimulationResult(
        True,
        ALLOW_TO_LAB_SIMULATION,
        ALLOW_TO_LAB_SIMULATION,
        approval,
    )
