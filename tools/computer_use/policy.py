"""Policy layer for Hermes computer use.

The policy is intentionally app/action scoped, mirroring Codex Computer Use:
read-only state calls are free; mutating UI actions need explicit approval and
can be granted once, for the session, or always for the specific app+action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ComputerUseRisk(str, Enum):
    READ = "read"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


READ_ONLY_ACTIONS = frozenset({"capture", "get_app_state", "list_apps", "wait"})
MUTATING_ACTIONS = frozenset({
    "click",
    "double_click",
    "right_click",
    "middle_click",
    "perform_secondary_action",
    "drag",
    "scroll",
    "type",
    "type_text",
    "key",
    "press_key",
    "set_value",
    "select_text",
    "focus_app",
    "launch_app",
})


@dataclass(frozen=True)
class ComputerUseRequest:
    action: str
    app: str = ""
    args: Dict[str, Any] | None = None

    @property
    def scope_key(self) -> Tuple[str, str]:
        return ((self.app or "*").strip().lower() or "*", self.action.strip().lower())


@dataclass(frozen=True)
class ComputerUsePolicyDecision:
    allowed: bool
    risk: ComputerUseRisk
    reason: str = ""
    approval_required: bool = False
    scope_key: Tuple[str, str] | None = None


class ComputerUsePolicy:
    def __init__(self) -> None:
        self.session_allowed: set[Tuple[str, str]] = set()
        self.always_allowed: set[Tuple[str, str]] = set()

    def reset_session(self) -> None:
        self.session_allowed.clear()

    def grant(self, request: ComputerUseRequest, scope: str) -> None:
        key = request.scope_key
        if scope in {"approve_session", "session"}:
            self.session_allowed.add(key)
        elif scope in {"always_approve", "approve_always", "always"}:
            self.session_allowed.add(key)
            self.always_allowed.add(key)

    def evaluate(self, request: ComputerUseRequest) -> ComputerUsePolicyDecision:
        action = request.action.strip().lower()
        key = request.scope_key
        wildcard = ("*", action)
        if action in READ_ONLY_ACTIONS:
            return ComputerUsePolicyDecision(True, ComputerUseRisk.READ, "read-only", False, key)
        if action not in MUTATING_ACTIONS:
            return ComputerUsePolicyDecision(False, ComputerUseRisk.BLOCKED, f"unknown action {action!r}", False, key)
        if key in self.session_allowed or key in self.always_allowed or wildcard in self.session_allowed or wildcard in self.always_allowed:
            return ComputerUsePolicyDecision(True, ComputerUseRisk.LOW, "approved for scope", False, key)
        risk = ComputerUseRisk.MEDIUM
        if action in {"type", "type_text", "press_key", "key", "set_value", "select_text"}:
            risk = ComputerUseRisk.HIGH
        return ComputerUsePolicyDecision(False, risk, "approval required", True, key)


def app_from_args(args: Optional[Dict[str, Any]]) -> str:
    if not args:
        return ""
    return str(args.get("app") or args.get("bundle_id") or "")
