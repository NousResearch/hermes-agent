"""Control-plane, degraded-boot and supply-chain qualification contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from workstation.runtime import CancellationToken, run_with_deadline


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ControlPlaneDecision:
    decision: str
    reason: str
    task_id: str
    destination: str
    created_at: str = field(default_factory=_utc_now)


class ControlPlaneMonitor:
    """Independent policy observation path; it has no task-execution methods."""

    def __init__(self, *, allowed_destinations: Iterable[str] = ()) -> None:
        self.allowed_destinations = {item.strip().lower() for item in allowed_destinations}
        self.incident_trace: list[dict[str, Any]] = []

    def observe_network(self, destination: str, *, task_id: str) -> ControlPlaneDecision:
        normalized = destination.strip().lower()
        allowed = normalized in self.allowed_destinations
        return self._decide(
            task_id=task_id,
            destination=destination,
            allowed=allowed,
            allowed_reason="destination is declared",
            denied_reason="undeclared external surface",
        )

    def observe_action(self, action: str, *, task_id: str, allowed_actions: Iterable[str]) -> ControlPlaneDecision:
        allowed = action.strip().lower() in {item.strip().lower() for item in allowed_actions}
        return self._decide(
            task_id=task_id,
            destination=action,
            allowed=allowed,
            allowed_reason="action is declared",
            denied_reason="undeclared action",
        )

    def observe_spend(self, cost_usd: float, *, task_id: str, budget_usd: float) -> ControlPlaneDecision:
        allowed = cost_usd >= 0 and budget_usd >= 0 and cost_usd <= budget_usd
        return self._decide(
            task_id=task_id,
            destination=f"spend:{cost_usd:.6f}",
            allowed=allowed,
            allowed_reason="spend is within budget",
            denied_reason="spend exceeds budget",
        )

    def observe_permission(
        self,
        permission: str,
        *,
        task_id: str,
        declared_permissions: Iterable[str],
    ) -> ControlPlaneDecision:
        allowed = permission.strip().lower() in {item.strip().lower() for item in declared_permissions}
        return self._decide(
            task_id=task_id,
            destination=f"permission:{permission}",
            allowed=allowed,
            allowed_reason="permission is declared",
            denied_reason="undeclared permission expansion",
        )

    def _decide(
        self,
        *,
        task_id: str,
        destination: str,
        allowed: bool,
        allowed_reason: str,
        denied_reason: str,
    ) -> ControlPlaneDecision:
        decision = ControlPlaneDecision(
            decision="allow" if allowed else "suspend",
            reason=allowed_reason if allowed else denied_reason,
            task_id=task_id,
            destination=destination,
        )
        self.incident_trace.append(
            {
                "incident_id": f"incident-{uuid4().hex}",
                "task_id": task_id,
                "destination": destination,
                "decision": decision.decision,
                "reason": decision.reason,
                "created_at": decision.created_at,
            }
        )
        return decision


@dataclass(slots=True)
class ComponentBootStatus:
    name: str
    status: str
    detail: str = ""


class ComponentBootManager:
    """Boot optional components independently and retain quarantine state."""

    def __init__(self, recovery_plane: Any | None = None) -> None:
        self._statuses: dict[str, ComponentBootStatus] = {}
        self.recovery_plane = recovery_plane

    def boot(self, checks: Mapping[str, Callable[[], Any]]) -> dict[str, ComponentBootStatus]:
        report: dict[str, ComponentBootStatus] = {}
        for name, check in checks.items():
            if self.recovery_plane is not None and self.recovery_plane.is_quarantined(name):
                report[name] = ComponentBootStatus(name, "quarantined", "disabled by Recovery Plane")
                continue
            try:
                result = check()
                ready = result is True or (isinstance(result, str) and result.lower() in {"ok", "ready", "healthy"})
                status = ComponentBootStatus(name, "ready" if ready else "quarantined", str(result))
            except Exception as exc:
                status = ComponentBootStatus(name, "quarantined", str(exc))
            report[name] = status
        self._statuses = report
        return dict(report)

    def statuses(self) -> dict[str, ComponentBootStatus]:
        return dict(self._statuses)

    def is_available(self, name: str) -> bool:
        return self._statuses.get(name, ComponentBootStatus(name, "missing")).status == "ready"


class TrustLabel(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class BoundedMCPExecutor:
    def __init__(
        self,
        *,
        max_pages: int = 10,
        max_tools: int = 100,
        max_response_bytes: int = 1_000_000,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.max_pages = max(1, max_pages)
        self.max_tools = max(1, max_tools)
        self.max_response_bytes = max(1, max_response_bytes)
        self.timeout_seconds = max(0.001, timeout_seconds)

    def discover(
        self,
        fetch_page: Callable[[int], Iterable[str]],
        *,
        pages: int,
        cancellation: CancellationToken | None = None,
    ) -> list[str]:
        if pages < 0:
            raise ValueError("page count cannot be negative")
        if pages > self.max_pages:
            raise ValueError(f"page limit exceeded: {pages} > {self.max_pages}")
        tools: list[str] = []
        for page in range(pages):
            page_tools = run_with_deadline(
                lambda page=page: list(fetch_page(page)),
                timeout_seconds=self.timeout_seconds,
                cancellation=cancellation,
            )
            for tool in page_tools:
                tools.append(str(tool))
                if len(tools) > self.max_tools:
                    raise ValueError("tool limit exceeded")
        return tools

    def execute(
        self,
        call: Callable[[], Any],
        *,
        timeout_seconds: float | None = None,
        cancellation: CancellationToken | None = None,
    ) -> Any:
        result = run_with_deadline(
            call,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            cancellation=cancellation,
        )
        encoded = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
        if len(encoded) > self.max_response_bytes:
            raise ValueError("response limit exceeded")
        return result


class QualificationStage(str, Enum):
    PROVENANCE = "source/provenance"
    STATIC_INSPECTION = "static_inspection"
    CAPABILITIES = "requested_capabilities"
    SANDBOX = "sandbox"
    BEHAVIORAL_SMOKE = "behavioral_smoke"
    SIGNATURE = "signature/reputation"


@dataclass(slots=True)
class QualificationResult:
    accepted: bool
    source: str
    trust: TrustLabel
    stages: list[QualificationStage] = field(default_factory=list)
    failed_stage: QualificationStage | None = None
    reason: str = ""


class ExtensionQualification:
    def run(
        self,
        *,
        source: str,
        requested_capabilities: Iterable[str],
        checks: Mapping[QualificationStage, Callable[[], Any]],
        trust: TrustLabel,
    ) -> QualificationResult:
        if not source.strip() or not tuple(requested_capabilities):
            return QualificationResult(False, source, trust, reason="source and capabilities are required")
        passed: list[QualificationStage] = []
        for stage in QualificationStage:
            check = checks.get(stage)
            if check is None:
                return QualificationResult(False, source, trust, passed, stage, "qualification stage missing")
            try:
                result = check()
            except Exception as exc:
                return QualificationResult(False, source, trust, passed, stage, str(exc))
            if result is not True and not (isinstance(result, str) and result.lower() in {"ok", "pass", "passed"}):
                return QualificationResult(False, source, trust, passed, stage, "qualification check failed")
            passed.append(stage)
        return QualificationResult(True, source, trust, passed, reason="all qualification stages passed")


class ReleaseQualificationGate:
    """Evidence gate for upstream updates; callbacks are intentionally read-only."""

    STAGES = ("ownership", "assets", "clean_install", "workstation_smoke", "native_smoke", "migration")

    def run(self, checks: Mapping[str, Callable[[], bool]]) -> dict[str, Any]:
        result: dict[str, Any] = {"accepted": True, "stages": []}
        for stage in self.STAGES:
            check = checks.get(stage)
            if check is None:
                result.update(accepted=False, failed_stage=stage, reason="stage missing")
                return result
            try:
                passed = bool(check())
            except Exception as exc:
                result.update(accepted=False, failed_stage=stage, reason=str(exc))
                return result
            if not passed:
                result.update(accepted=False, failed_stage=stage, reason="stage failed")
                return result
            result["stages"].append(stage)
        return result
