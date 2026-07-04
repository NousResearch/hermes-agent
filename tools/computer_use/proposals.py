"""Dry-run-only input action proposal schemas for computer-use/browser routing.

This module deliberately does not execute keyboard, mouse, browser, or native UI
input. It only normalizes proposed actions, classifies risk, and returns an
auditable dry-run result that a future approved executor may consume.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .capabilities import classify_action_risk

BrowserActionKind = Literal["click", "type_text", "key_combo", "drag", "scroll"]
DesktopSurface = Literal["windows_uia", "windows_input_fallback", "native_gui"]
DesktopActionKind = str

_BROWSER_RISK_ACTION = {
    "click": "browser_click",
    "type_text": "browser_type",
    "key_combo": "browser_press",
    "drag": "browser_click",
    "scroll": "browser_scroll",
}

_WINDOWS_UIA_READ_ONLY_ACTIONS = frozenset({"list_windows", "snapshot_tree", "element_capabilities"})
_WINDOWS_UIA_SEMANTIC_MUTATION_ACTIONS = frozenset(
    {"invoke", "set_value", "select", "toggle", "expand", "collapse", "scroll_into_view"}
)
_WINDOWS_INPUT_FALLBACK_ACTIONS = frozenset(
    {"click_coord", "drag_coord", "type_foreground", "key_combo_foreground"}
)


@dataclass(frozen=True)
class ElementRef:
    """Stable-ish web or UI target reference for a dry-run proposal."""

    selector: str = ""
    role: str = ""
    name: str = ""
    point: tuple[int, int] | None = None
    frame: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrowserActionProposal:
    """A browser input action proposal. It is not an executor."""

    action: BrowserActionKind
    target: ElementRef | None = None
    text: str = ""
    keys: tuple[str, ...] = ()
    origin: str = ""
    task: str = ""
    expected_result: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.target is not None:
            data["target"] = self.target.to_dict()
        return data


@dataclass(frozen=True)
class DesktopActionProposal:
    """A Windows/native desktop action proposal. It is dry-run only."""

    surface: DesktopSurface = "windows_uia"
    action: DesktopActionKind = "snapshot_tree"
    target_selector: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)
    task: str = ""
    expected_result: str = ""
    read_only: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionProposalDryRun:
    """Dry-run result for an input proposal.

    `will_execute` is always False in v0. This prevents proposal creation from
    becoming an implicit click/type/drag execution path.
    """

    surface: str
    action: str
    risk: str
    allowed: bool
    requires_approval: bool
    reason: str
    next_step: str
    will_execute: bool
    proposal: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dry_run_action_proposal(
    proposal: BrowserActionProposal | DesktopActionProposal,
    *,
    platform: str | None = None,
) -> ActionProposalDryRun:
    """Classify a proposed action without executing it."""

    platform_id = platform or sys.platform
    if isinstance(proposal, BrowserActionProposal):
        return _dry_run_browser_proposal(proposal, platform_id)
    if isinstance(proposal, DesktopActionProposal):
        return _dry_run_desktop_proposal(proposal, platform_id)
    raise TypeError(f"Unsupported proposal type: {type(proposal)!r}")


def input_proposal_capability_status() -> dict[str, Any]:
    """Return JSON-ready capability metadata for status surfaces."""

    return {
        "available": True,
        "dry_run_only": True,
        "will_execute": False,
        "schemas": ["BrowserActionProposal", "DesktopActionProposal"],
        "browser_actions": sorted(_BROWSER_RISK_ACTION),
        "desktop_read_only_actions": sorted(_WINDOWS_UIA_READ_ONLY_ACTIONS),
        "desktop_semantic_mutation_actions": sorted(_WINDOWS_UIA_SEMANTIC_MUTATION_ACTIONS),
        "coordinate_input_fallback_actions": sorted(_WINDOWS_INPUT_FALLBACK_ACTIONS),
        "coordinate_input_fallback_enabled": False,
        "next_step": "Create dry-run proposals first; route execution only through a separately approved backend.",
    }


def _dry_run_browser_proposal(proposal: BrowserActionProposal, platform_id: str) -> ActionProposalDryRun:
    risk_action = _BROWSER_RISK_ACTION.get(proposal.action, "")
    decision = classify_action_risk(
        surface="browser",
        action=risk_action,
        platform=platform_id,
        task=proposal.task,
    )
    return ActionProposalDryRun(
        surface="browser",
        action=proposal.action,
        risk=decision.risk,
        allowed=decision.allowed,
        requires_approval=decision.requires_approval,
        reason=decision.reason,
        next_step=f"Dry-run only. {decision.next_step}",
        will_execute=False,
        proposal=proposal.to_dict(),
    )


def _dry_run_desktop_proposal(proposal: DesktopActionProposal, platform_id: str) -> ActionProposalDryRun:
    normalized_surface = (proposal.surface or "").strip().lower().replace("-", "_")
    normalized_action = (proposal.action or "").strip().lower()

    if normalized_surface == "windows_input_fallback" or normalized_action in _WINDOWS_INPUT_FALLBACK_ACTIONS:
        return ActionProposalDryRun(
            surface="windows_input_fallback",
            action=normalized_action,
            risk="blocked",
            allowed=False,
            requires_approval=False,
            reason="Coordinate/global input fallback is disabled in computer-use-input-proposal-v0.",
            next_step="Use browser proposals or Windows UIA read-only/semantic proposals; do not execute SendInput/pyautogui/pynput.",
            will_execute=False,
            proposal=proposal.to_dict(),
        )

    if normalized_surface in {"windows_uia", "native_gui"} and normalized_action in _WINDOWS_UIA_READ_ONLY_ACTIONS:
        return ActionProposalDryRun(
            surface="windows_uia",
            action=normalized_action,
            risk="low",
            allowed=True,
            requires_approval=False,
            reason="Windows UIA read-only proposal; no focus, click, type, drag, or native input execution.",
            next_step="Use the Windows UIA read-only skeleton to collect metadata when available.",
            will_execute=False,
            proposal=proposal.to_dict(),
        )

    if normalized_surface == "windows_uia" and normalized_action in _WINDOWS_UIA_SEMANTIC_MUTATION_ACTIONS:
        return ActionProposalDryRun(
            surface="windows_uia",
            action=normalized_action,
            risk="high",
            allowed=True,
            requires_approval=True,
            reason="Windows UIA semantic mutation proposal only; execution is not implemented in v0.",
            next_step="Require future explicit approval token and selector revalidation before any executor is added.",
            will_execute=False,
            proposal=proposal.to_dict(),
        )

    decision = classify_action_risk(
        surface="native_gui",
        action=normalized_action,
        platform=platform_id,
        task=proposal.task,
        read_only=proposal.read_only,
    )
    return ActionProposalDryRun(
        surface="native_gui",
        action=normalized_action,
        risk=decision.risk,
        allowed=decision.allowed,
        requires_approval=decision.requires_approval,
        reason=decision.reason,
        next_step=f"Dry-run only. {decision.next_step}",
        will_execute=False,
        proposal=proposal.to_dict(),
    )
