"""Browser input execution v1.

This module executes browser input proposals only through an explicitly injected
browser backend. It does not launch browsers, call paid/cloud providers, access
native Windows GUI automation, or use SendInput/pyautogui/pynput. The default
capability status is therefore conservative: the executor exists, but no live
backend is configured by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .proposals import BrowserActionProposal, dry_run_action_proposal

_APPROVAL_TOKENS = frozenset({"approved", "approve_once", "approve_session", "always_approve"})
_SUPPORTED_ACTIONS = ("click", "type_text", "key_combo", "drag", "scroll")


class BrowserInputBackend(Protocol):
    def is_available(self) -> bool: ...

    def click(self, target: dict[str, Any], *, button: str = "left") -> dict[str, Any]: ...

    def type_text(self, target: dict[str, Any] | None, text: str) -> dict[str, Any]: ...

    def key_combo(self, keys: tuple[str, ...]) -> dict[str, Any]: ...

    def drag(self, source: dict[str, Any] | None, target: dict[str, Any] | None) -> dict[str, Any]: ...

    def scroll(self, target: dict[str, Any] | None, delta_x: int = 0, delta_y: int = 0) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BrowserInputExecutionResult:
    ok: bool
    executed: bool
    action: str
    risk: str
    requires_approval: bool
    reason: str
    proposal: dict[str, Any]
    backend_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrowserInputExecutor:
    """Execute browser proposals through an injected backend after dry-run gates."""

    def __init__(self, *, backend: BrowserInputBackend) -> None:
        self.backend = backend

    def execute(
        self,
        proposal: BrowserActionProposal,
        *,
        platform: str | None = None,
        approval_token: str = "",
    ) -> BrowserInputExecutionResult:
        dry_run = dry_run_action_proposal(proposal, platform=platform)
        proposal_dict = dry_run.proposal

        if not dry_run.allowed:
            return BrowserInputExecutionResult(
                ok=False,
                executed=False,
                action=proposal.action,
                risk=dry_run.risk,
                requires_approval=dry_run.requires_approval,
                reason=dry_run.reason,
                proposal=proposal_dict,
            )

        if dry_run.requires_approval and approval_token not in _APPROVAL_TOKENS:
            return BrowserInputExecutionResult(
                ok=False,
                executed=False,
                action=proposal.action,
                risk=dry_run.risk,
                requires_approval=True,
                reason="Browser input action requires explicit approval before execution.",
                proposal=proposal_dict,
            )

        try:
            available = bool(self.backend.is_available())
        except Exception as exc:
            return BrowserInputExecutionResult(
                ok=False,
                executed=False,
                action=proposal.action,
                risk=dry_run.risk,
                requires_approval=dry_run.requires_approval,
                reason=f"Browser input backend availability check failed: {exc}",
                proposal=proposal_dict,
            )
        if not available:
            return BrowserInputExecutionResult(
                ok=False,
                executed=False,
                action=proposal.action,
                risk=dry_run.risk,
                requires_approval=dry_run.requires_approval,
                reason="Browser input backend is not available.",
                proposal=proposal_dict,
            )

        target = proposal.target.to_dict() if proposal.target is not None else None
        backend_result = self._execute_backend_action(proposal, target)
        return BrowserInputExecutionResult(
            ok=bool(backend_result.get("ok", True)),
            executed=True,
            action=proposal.action,
            risk=dry_run.risk,
            requires_approval=dry_run.requires_approval,
            reason="Browser input action executed through injected backend.",
            proposal=proposal_dict,
            backend_result=dict(backend_result),
        )

    def _execute_backend_action(
        self,
        proposal: BrowserActionProposal,
        target: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if proposal.action == "click":
            return self.backend.click(target or {}, button="left")
        if proposal.action == "type_text":
            return self.backend.type_text(target, proposal.text)
        if proposal.action == "key_combo":
            return self.backend.key_combo(tuple(proposal.keys))
        if proposal.action == "drag":
            source = proposal.target.to_dict() if proposal.target is not None else None
            destination = proposal.target.to_dict() if proposal.target is not None else None
            return self.backend.drag(source, destination)
        if proposal.action == "scroll":
            return self.backend.scroll(target, delta_x=0, delta_y=0)
        return {"ok": False, "action": proposal.action, "error": "unsupported action"}


def browser_input_execution_capability_status() -> dict[str, Any]:
    return {
        "available": False,
        "requires_injected_backend": True,
        "native_gui_mutation_allowed": False,
        "launches_browser": False,
        "uses_paid_or_cloud_provider": False,
        "supported_actions": list(_SUPPORTED_ACTIONS),
        "approval_required_for_high_risk": True,
        "sensitive_tasks_blocked": True,
        "next_step": "Wire a local/browser-tool backend explicitly; this module will not launch or auto-discover one.",
    }
