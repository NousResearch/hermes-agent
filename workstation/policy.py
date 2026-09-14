from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path
import re
from typing import Any, List, Optional

from workstation.contracts import RiskLevel, utc_now
from workstation.safety import APPROVAL_REQUIRED_ACTIONS, classify_action

_log = logging.getLogger(__name__)


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    SANDBOX = "sandbox"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(slots=True)
class ActionScope:
    task_id: str
    session_id: str
    capability: str  # "browser" | "filesystem" | "process" | "git" | "worker_delegation"
    action_name: str
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    workspace_root: Optional[str] = None
    human_in_the_loop: bool = False


@dataclass(slots=True)
class PolicyEvaluation:
    decision: PolicyDecision
    reason: str
    risk_level: RiskLevel
    constraints: list[str] = field(default_factory=list)
    evaluated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        data["risk_level"] = self.risk_level.value
        return data


# Denied command patterns that can cause irrecoverable system destruction
DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r"rm\s+-rf\s+(/|/\*|~|~/\*)", re.IGNORECASE),
    re.compile(r"format\s+[c-z]:", re.IGNORECASE),
    re.compile(r"del\s+/[a-z0-9\s]*[c-z]:\\", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),  # fork bomb
    re.compile(r"dd\s+if=.*of=/dev/[sh]d[a-z]", re.IGNORECASE),
    re.compile(r"mkfs\.[a-z0-9]+\s+/dev/", re.IGNORECASE),
]

# Sensitive system locations protected from arbitrary modification
SENSITIVE_PATHS = [
    r"^[a-zA-Z]:\\windows",
    r"^[a-zA-Z]:\\program files",
    r"^/etc",
    r"^/usr/bin",
    r"^/bin",
    r"^/sbin",
    r"^/boot",
    r".*[\\/]\.ssh([\\/]|$)",
    r".*[\\/]\.gnupg([\\/]|$)",
    r".*[\\/]\.aws([\\/]|$)",
]


class ScopedPolicyEngine:
    """Scoped Autonomy Policy Engine for Hermes Workstation.

    Evaluates operations across browser, host capabilities, and specialist workers,
    enforcing least privilege and auditable safety boundaries.
    """

    def __init__(self) -> None:
        self._audit_log: list[tuple[ActionScope, PolicyEvaluation]] = []

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "task_id": scope.task_id,
                "session_id": scope.session_id,
                "capability": scope.capability,
                "action": scope.action_name,
                "target": scope.target,
                "evaluation": eval_res.to_dict(),
            }
            for scope, eval_res in self._audit_log[-limit:]
        ]

    def evaluate(self, scope: ActionScope) -> PolicyEvaluation:
        # Extensions execute third-party code inside the authenticated Chromium
        # profile.  The caller supplies a manifest-derived risk assessment;
        # never trust an action name alone to downgrade it.
        if scope.capability == "browser_extension":
            raw_risk = str(scope.parameters.get("extension_risk", RiskLevel.HIGH.value)).lower()
            try:
                extension_risk = RiskLevel(raw_risk)
            except ValueError:
                extension_risk = RiskLevel.HIGH
            if scope.action_name in {"install_extension", "update_extension"} and extension_risk in {
                RiskLevel.MEDIUM,
                RiskLevel.HIGH,
                RiskLevel.CRITICAL,
            }:
                res = PolicyEvaluation(
                    decision=PolicyDecision.REQUIRE_APPROVAL,
                    reason="Extension permissions can access authenticated browser data or pages",
                    risk_level=extension_risk,
                    constraints=["require_explicit_human_confirmation", "record_manifest_permissions"],
                )
                self._record_audit(scope, res)
                return res
            if scope.action_name == "uninstall_extension":
                res = PolicyEvaluation(
                    decision=PolicyDecision.REQUIRE_APPROVAL,
                    reason="Removing an extension changes the browser capability set",
                    risk_level=max(extension_risk, RiskLevel.MEDIUM, key=lambda item: list(RiskLevel).index(item)),
                    constraints=["require_explicit_human_confirmation"],
                )
                self._record_audit(scope, res)
                return res

        # 1. Fast check: dangerous destructive command patterns -> DENY
        if scope.capability == "process" or scope.action_name in ("run_command", "terminal", "exec"):
            target_cmd = scope.target or str(scope.parameters.get("command", ""))
            for pat in DANGEROUS_COMMAND_PATTERNS:
                if pat.search(target_cmd):
                    res = PolicyEvaluation(
                        decision=PolicyDecision.DENY,
                        reason=f"Dangerous command pattern matched: '{pat.pattern}'",
                        risk_level=RiskLevel.CRITICAL,
                    )
                    self._record_audit(scope, res)
                    return res

        # 2. Check filesystem write/delete operations against sensitive OS paths
        if scope.capability == "filesystem" and scope.action_name in ("write", "delete", "modify", "remove"):
            target_path = os.path.abspath(scope.target or str(scope.parameters.get("path", "")))
            for pat in SENSITIVE_PATHS:
                if re.search(pat, target_path, re.IGNORECASE):
                    res = PolicyEvaluation(
                        decision=PolicyDecision.DENY,
                        reason=f"Operation targets protected system path: '{target_path}'",
                        risk_level=RiskLevel.CRITICAL,
                    )
                    self._record_audit(scope, res)
                    return res

        # 3. Check workspace boundary containment if workspace_root is defined
        if scope.workspace_root and scope.capability == "filesystem":
            target_path = os.path.abspath(scope.target or str(scope.parameters.get("path", "")))
            ws_root = os.path.abspath(scope.workspace_root)
            try:
                common = os.path.commonpath([ws_root, target_path])
                if common != ws_root and scope.action_name in ("write", "delete", "modify"):
                    res = PolicyEvaluation(
                        decision=PolicyDecision.REQUIRE_APPROVAL,
                        reason=f"Filesystem modification outside workspace root: {target_path}",
                        risk_level=RiskLevel.HIGH,
                        constraints=["require_explicit_human_confirmation"],
                    )
                    self._record_audit(scope, res)
                    return res
            except ValueError:
                # Different drives on Windows
                res = PolicyEvaluation(
                    decision=PolicyDecision.REQUIRE_APPROVAL,
                    reason=f"Target path is on a different drive than workspace: {target_path}",
                    risk_level=RiskLevel.HIGH,
                )
                self._record_audit(scope, res)
                return res

        # 4. Check safety boundaries for financial/destructive/account actions
        safety_dec = classify_action(scope.action_name)
        if safety_dec.approval_required:
            res = PolicyEvaluation(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                reason=f"High-impact action requires human approval: {safety_dec.reason}",
                risk_level=safety_dec.risk,
                constraints=["wait_for_human_consent"],
            )
            self._record_audit(scope, res)
            return res

        # 5. Untrusted / external network script execution -> SANDBOX
        if scope.action_name in ("run_untrusted", "download_and_exec", "sandbox_eval"):
            res = PolicyEvaluation(
                decision=PolicyDecision.SANDBOX,
                reason="Untrusted execution requires sandboxing constraints",
                risk_level=RiskLevel.MEDIUM,
                constraints=["network_isolated", "timeout_15s", "read_only_fs"],
            )
            self._record_audit(scope, res)
            return res

        # 6. Default benign / low risk action -> ALLOW
        res = PolicyEvaluation(
            decision=PolicyDecision.ALLOW,
            reason="Action conforms to scoped autonomy boundaries",
            risk_level=safety_dec.risk,
        )
        self._record_audit(scope, res)
        return res

    def _record_audit(self, scope: ActionScope, evaluation: PolicyEvaluation) -> None:
        self._audit_log.append((scope, evaluation))
