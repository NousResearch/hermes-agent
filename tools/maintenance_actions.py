"""Non-executing maintenance-action policy validation.

This module intentionally does not run commands. It classifies whether a named
maintenance action is blocked, eligible-but-needs-current-user-approval, or
approved by policy. Execution, preflight probes, postchecks, and audit writes
belong to later layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_SHELL_WRAPPERS = {"sh", "bash", "zsh", "ksh", "fish", "python", "python3", "perl", "ruby"}
_SHELL_EVAL_FLAGS = {"-c", "-lc", "-ec", "-e"}
_UNATTENDED_CONTEXTS = {"cron", "unattended", "background", "scheduler"}


@dataclass(frozen=True)
class MaintenanceActionDecision:
    """Result of maintenance-action policy validation."""

    allowed: bool
    reason: str
    eligible: bool = False
    action_id: str = ""
    command_id: str = ""
    host_label: str = ""


def _blocked(reason: str, *, action_id: str = "", action: dict[str, Any] | None = None) -> MaintenanceActionDecision:
    action = action or {}
    return MaintenanceActionDecision(
        allowed=False,
        reason=reason,
        eligible=False,
        action_id=action_id,
        command_id=str(action.get("command_id", "") or ""),
        host_label=str(action.get("host_label", "") or ""),
    )


def _requires_approval(action_id: str, action: dict[str, Any]) -> MaintenanceActionDecision:
    return MaintenanceActionDecision(
        allowed=False,
        reason="requires_current_user_approval",
        eligible=True,
        action_id=action_id,
        command_id=str(action.get("command_id", "") or ""),
        host_label=str(action.get("host_label", "") or ""),
    )


def _approved(action_id: str, action: dict[str, Any]) -> MaintenanceActionDecision:
    return MaintenanceActionDecision(
        allowed=True,
        reason="approved",
        eligible=True,
        action_id=action_id,
        command_id=str(action.get("command_id", "") or ""),
        host_label=str(action.get("host_label", "") or ""),
    )


def _looks_like_shell_wrapping(argv: list[Any]) -> bool:
    if not argv:
        return False
    head = str(argv[0]).rsplit("/", 1)[-1]
    if head not in _SHELL_WRAPPERS:
        return False
    return any(str(part) in _SHELL_EVAL_FLAGS for part in argv[1:3])


def _valid_exact_argv(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(part, str) and part for part in value)
    )


def _valid_requested_argv(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(part, str) and part for part in value)
    )


def evaluate_maintenance_action(
    policy: Any,
    action_id: str,
    requested_argv: Any,
    *,
    invocation_context: str = "interactive",
    current_user_approved: bool = False,
) -> MaintenanceActionDecision:
    """Evaluate a named maintenance action without executing it.

    The function is deliberately conservative. It does not run preflight or
    postcheck probes; it only validates static policy gates and exact argv
    matching. Later execution code must still perform live preflight,
    current-user approval capture, command execution, postcheck verification,
    and audit logging.
    """

    if not policy:
        return _blocked("policy_absent", action_id=action_id)
    if not isinstance(policy, dict):
        return _blocked("policy_invalid", action_id=action_id)
    if not policy.get("enabled", False):
        return _blocked("policy_disabled", action_id=action_id)

    if str(invocation_context or "").lower() in _UNATTENDED_CONTEXTS:
        if policy.get("unattended_policy", "none") in (None, "none", False):
            return _blocked("unattended_forbidden", action_id=action_id)

    actions = policy.get("actions")
    if not isinstance(actions, dict):
        return _blocked("actions_invalid", action_id=action_id)

    action = actions.get(action_id)
    if action is None:
        return _blocked("unknown_action", action_id=action_id)
    if not isinstance(action, dict):
        return _blocked("action_invalid", action_id=action_id)
    if not action.get("enabled", False):
        return _blocked("action_disabled", action_id=action_id, action=action)

    expected_argv = action.get("exact_argv")
    if not _valid_exact_argv(expected_argv):
        return _blocked("exact_argv_invalid", action_id=action_id, action=action)

    if not isinstance(requested_argv, list):
        return _blocked("argv_must_be_list", action_id=action_id, action=action)
    if not _valid_requested_argv(requested_argv):
        return _blocked("argv_invalid", action_id=action_id, action=action)
    if _looks_like_shell_wrapping(requested_argv):
        return _blocked("shell_wrapping_forbidden", action_id=action_id, action=action)
    if requested_argv != expected_argv:
        return _blocked("argv_mismatch", action_id=action_id, action=action)

    if not action.get("preflight_profile"):
        return _blocked("missing_preflight_profile", action_id=action_id, action=action)
    if not action.get("postcheck_profile"):
        return _blocked("missing_postcheck_profile", action_id=action_id, action=action)

    if policy.get("require_interactive_user_approval", True) and not current_user_approved:
        return _requires_approval(action_id, action)

    return _approved(action_id, action)
