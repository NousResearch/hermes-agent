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
_UNATTENDED_FORBIDDEN_VALUES = {"", "none", "false", "off", "disabled", "forbidden"}


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


def _basename(value: Any) -> str:
    return str(value).rsplit("/", 1)[-1]


def _looks_like_shell_wrapping(argv: list[Any]) -> bool:
    if argv and _basename(argv[0]) == "env" and "-S" in argv[1:]:
        return True
    for index, part in enumerate(argv):
        if _basename(part) in _SHELL_WRAPPERS:
            return any(str(later) in _SHELL_EVAL_FLAGS for later in argv[index + 1 :])
    return False


def _valid_exact_argv(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(part, str) and part for part in value)
        and not _looks_like_shell_wrapping(value)
    )


def _valid_requested_argv(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(part, str) and part for part in value)
    )


def _unattended_policy_reason(value: Any) -> str:
    if value is None or value is False:
        return "unattended_forbidden"
    if isinstance(value, str) and value.strip().lower() in _UNATTENDED_FORBIDDEN_VALUES:
        return "unattended_forbidden"
    return "unattended_policy_invalid"


def _valid_profile_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_action_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
        return _blocked("policy_absent", action_id=str(action_id or ""))
    if not _valid_action_id(action_id):
        return _blocked("invalid_action_id")
    if not isinstance(policy, dict):
        return _blocked("policy_invalid", action_id=action_id)
    policy_enabled = policy.get("enabled")
    if policy_enabled is False:
        return _blocked("policy_disabled", action_id=action_id)
    if policy_enabled is not True:
        return _blocked("policy_enabled_invalid", action_id=action_id)

    unattended_reason = _unattended_policy_reason(policy.get("unattended_policy", "none"))
    if unattended_reason == "unattended_policy_invalid":
        return _blocked(unattended_reason, action_id=action_id)

    invocation_context_normalized = str(invocation_context or "").strip().lower()
    if invocation_context_normalized in _UNATTENDED_CONTEXTS:
        return _blocked(unattended_reason, action_id=action_id)

    actions = policy.get("actions")
    if not isinstance(actions, dict):
        return _blocked("actions_invalid", action_id=action_id)

    action = actions.get(action_id)
    if action is None:
        return _blocked("unknown_action", action_id=action_id)
    if not isinstance(action, dict):
        return _blocked("action_invalid", action_id=action_id)
    action_enabled = action.get("enabled")
    if action_enabled is False:
        return _blocked("action_disabled", action_id=action_id, action=action)
    if action_enabled is not True:
        return _blocked("action_enabled_invalid", action_id=action_id, action=action)

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

    if "preflight_profile" not in action or action.get("preflight_profile") is None:
        return _blocked("missing_preflight_profile", action_id=action_id, action=action)
    preflight_profile = action.get("preflight_profile")
    if not _valid_profile_ref(preflight_profile):
        return _blocked("invalid_preflight_profile", action_id=action_id, action=action)

    if "postcheck_profile" not in action or action.get("postcheck_profile") is None:
        return _blocked("missing_postcheck_profile", action_id=action_id, action=action)
    postcheck_profile = action.get("postcheck_profile")
    if not _valid_profile_ref(postcheck_profile):
        return _blocked("invalid_postcheck_profile", action_id=action_id, action=action)

    require_approval = policy.get("require_interactive_user_approval", True)
    if not isinstance(require_approval, bool):
        return _blocked("invalid_approval_requirement", action_id=action_id, action=action)
    if not isinstance(current_user_approved, bool):
        return _blocked("invalid_current_user_approval", action_id=action_id, action=action)
    if require_approval and current_user_approved is not True:
        return _requires_approval(action_id, action)

    return _approved(action_id, action)


def classify_maintenance_action(
    policy: Any,
    action_id: str,
    requested_argv: Any,
    *,
    invocation_context: str = "interactive",
    current_user_approved: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable dry-run classification for an action.

    This is intentionally a thin wrapper around ``evaluate_maintenance_action``.
    It does not execute commands, load live config, run preflight/postcheck
    probes, write audit logs, or integrate with the terminal tool.
    """

    decision = evaluate_maintenance_action(
        policy,
        action_id,
        requested_argv,
        invocation_context=invocation_context,
        current_user_approved=current_user_approved,
    )
    return {
        "allowed": decision.allowed,
        "eligible": decision.eligible,
        "reason": decision.reason,
        "action_id": decision.action_id,
        "command_id": decision.command_id,
        "host_label": decision.host_label,
    }
