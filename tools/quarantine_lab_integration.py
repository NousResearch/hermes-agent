"""Explicit dependency-injected bridge from Skills Hub QUARANTINE to LAB simulation.

This module is development-only.  It has no install, execution, production-signing,
or reusable-approval capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.quarantine_dev_context import (
    DevHumanApprovalContext,
    validate_dev_approval_context,
)
from tools.quarantine_dev_human_approval import ForegroundReviewConsole
from tools.quarantine_lab_simulation import (
    ALLOW_TO_LAB_SIMULATION,
    run_lab_quarantine_simulation,
)

MAX_DEV_PROMPT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class DevLabIntegrationResult:
    handled: bool
    allowed: bool
    outcome: str
    reason: str
    candidate_sha256: str = ""
    evidence_digest: str = ""
    approval_scope_sha256: str = ""


@dataclass(frozen=True)
class DevLabQuarantineIntegration:
    """In-process LAB dependency. Never auto-constructed by Skills Hub."""

    context: DevHumanApprovalContext
    review_console: ForegroundReviewConsole
    final_confirm: Callable[[str], bool]
    audit_root: Path | None = None
    install_attempt_id: str | None = None
    timeout_seconds: float = MAX_DEV_PROMPT_TIMEOUT_SECONDS

    def simulate(
        self,
        external_result: Any,
        candidate_path: Path,
        *,
        source_id: str = "",
        source_version: str = "",
    ) -> DevLabIntegrationResult:
        checked = validate_dev_approval_context(self.context)
        if not checked.valid:
            return DevLabIntegrationResult(
                True, False, "LAB_SIMULATION_REJECTED", checked.reason
            )

        if self.timeout_seconds <= 0 or self.timeout_seconds > MAX_DEV_PROMPT_TIMEOUT_SECONDS:
            return DevLabIntegrationResult(
                True, False, "LAB_SIMULATION_REJECTED", "dev_lab_timeout_invalid"
            )

        if getattr(external_result, "decision", "") != "QUARANTINE":
            return DevLabIntegrationResult(
                True, False, "LAB_SIMULATION_REJECTED", "dev_approval_not_quarantine"
            )

        result = run_lab_quarantine_simulation(
            external_result,
            Path(candidate_path),
            self.context,
            self.review_console,
            final_confirm=self.final_confirm,
            source_id=source_id,
            source_version=source_version,
            install_attempt_id=self.install_attempt_id,
            audit_root=self.audit_root,
            timeout_seconds=self.timeout_seconds,
        )

        binding = getattr(external_result, "approval_binding", None)
        candidate_sha = str(getattr(binding, "candidate_sha256", "") or "")
        evidence_digest = str(getattr(binding, "evidence_digest", "") or "")

        allowed = bool(
            result.allowed and result.outcome == ALLOW_TO_LAB_SIMULATION
        )
        return DevLabIntegrationResult(
            handled=True,
            allowed=allowed,
            outcome=result.outcome,
            reason=result.reason,
            candidate_sha256=candidate_sha,
            evidence_digest=evidence_digest,
            approval_scope_sha256=result.approval.approval_scope_sha256,
        )
