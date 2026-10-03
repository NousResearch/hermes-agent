"""Closed machine-readable reason vocabulary for approval decisions."""

from __future__ import annotations

from enum import StrEnum


class ApprovalReason(StrEnum):
    """Stable reason codes carried beside human-facing approval text."""

    USER_DENIED = "user_denied"
    APPROVAL_TIMEOUT = "approval_timeout"
    APPROVAL_NOTIFY_FAILED = "approval_notify_failed"
    APPROVAL_CANCELLED = "approval_cancelled"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_BLOCKED = "policy_blocked"
    HARDLINE_BLOCKED = "hardline_blocked"
    USER_RULE_BLOCKED = "user_rule_blocked"
    SUDO_STDIN_BLOCKED = "sudo_stdin_blocked"
    SMART_DENIED = "smart_denied"
    SECURITY_SCANNER_UNAVAILABLE = "security_scanner_unavailable"


_OUTCOME_REASONS = {
    "denied": ApprovalReason.USER_DENIED,
    "timeout": ApprovalReason.APPROVAL_TIMEOUT,
    "notify_failed": ApprovalReason.APPROVAL_NOTIFY_FAILED,
    "cancelled": ApprovalReason.APPROVAL_CANCELLED,
    "blocked": ApprovalReason.POLICY_BLOCKED,
}


def reason_for_outcome(outcome: str) -> ApprovalReason:
    """Translate a legacy approval outcome to its stable security reason."""
    return _OUTCOME_REASONS.get(outcome, ApprovalReason.POLICY_BLOCKED)


def is_approval_reason(value: object) -> bool:
    """Return whether *value* belongs to the closed approval vocabulary."""
    try:
        ApprovalReason(value)
    except (TypeError, ValueError):
        return False
    return True
