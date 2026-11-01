"""Machine-readable reason codes for approval decisions."""

from __future__ import annotations


def test_nonconsent_builders_emit_closed_reason_codes():
    from tools import approval
    from tools.approval_reasons import ApprovalReason, is_approval_reason

    denied = approval._denied(
        "BLOCKED",
        pattern_key="rm",
        description="delete",
        outcome="denied",
    )
    blocked = approval._blocked(
        "BLOCKED",
        pattern_key="plugin_rule:network",
        description="external send",
    )

    assert denied["reason_code"] == ApprovalReason.USER_DENIED
    assert blocked["reason_code"] == ApprovalReason.POLICY_BLOCKED
    assert is_approval_reason(denied["reason_code"])
    assert is_approval_reason(blocked["reason_code"])
    assert not is_approval_reason("invented")


def test_floor_blocks_have_specific_reason_codes(monkeypatch):
    from tools import approval
    from tools.approval_reasons import ApprovalReason

    hardline = approval._floor_block("rm -rf /")
    assert hardline is not None
    assert hardline["reason_code"] == ApprovalReason.HARDLINE_BLOCKED

    monkeypatch.setattr(
        approval, "_match_user_deny_rule", lambda _command: "git push --force*"
    )
    user_deny = approval._user_deny_block("git push --force origin main")
    assert user_deny is not None
    assert user_deny["reason_code"] == ApprovalReason.USER_RULE_BLOCKED


def test_pending_and_smart_denial_codes_are_distinct(monkeypatch):
    from tools import approval
    from tools.approval_reasons import ApprovalReason

    monkeypatch.setattr(approval, "submit_pending", lambda *_args, **_kwargs: None)
    pending = approval._pending_result(
        approval._COMMAND_GATE,
        "session",
        command="rm -rf build",
        description="recursive delete",
        pattern_key="recursive-delete",
        pattern_keys=["recursive-delete"],
        body=None,
        smart_denied=False,
    )
    assert pending["reason_code"] == ApprovalReason.APPROVAL_REQUIRED

    monkeypatch.setattr(approval, "_smart_verdict", lambda *_args, **_kwargs: "deny")
    monkeypatch.setattr(approval, "_record_denial", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(approval, "_denial_breaker_addendum", lambda *_args: "")
    smart, denied = approval._smart_gate(
        approval._COMMAND_GATE,
        "dangerous",
        "dangerous command",
        "dangerous",
        ["dangerous"],
        "session",
        human_present=False,
    )
    assert denied is True
    assert smart is not None
    assert smart["reason_code"] == ApprovalReason.SMART_DENIED
