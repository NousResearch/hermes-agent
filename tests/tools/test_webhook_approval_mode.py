"""Tests for webhook approval mode (autonomous webhook runs).

Webhook-triggered agent runs have no interactive channel to answer /approve,
so dangerous commands must resolve via approvals.webhook_mode (deny|approve)
instead of hanging on submit_pending(). Mirrors cron_mode behavior.
"""
import sys
from pathlib import Path
from unittest.mock import patch as mock_patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.approval as approval


def _webhook_session():
    """Context: simulate a webhook session (platform == 'webhook', not cron)."""
    return mock_patch.multiple(
        approval,
        _get_session_platform=lambda: "webhook",
    )


def test_is_webhook_session_true_for_webhook_platform():
    with mock_patch.object(approval, "_get_session_platform", lambda: "webhook"):
        with mock_patch.object(approval, "env_var_enabled", lambda v: False):
            assert approval._is_webhook_session() is True


def test_is_webhook_session_false_when_cron():
    # Cron takes precedence even if a platform is bound.
    with mock_patch.object(approval, "_get_session_platform", lambda: "webhook"):
        with mock_patch.object(approval, "env_var_enabled",
                               lambda v: v == "HERMES_CRON_SESSION"):
            assert approval._is_webhook_session() is False


def test_is_webhook_session_false_for_telegram():
    with mock_patch.object(approval, "_get_session_platform", lambda: "telegram"):
        with mock_patch.object(approval, "env_var_enabled", lambda v: False):
            assert approval._is_webhook_session() is False


def test_webhook_mode_default_deny():
    with mock_patch("hermes_cli.config.load_config", return_value={}):
        assert approval._get_webhook_approval_mode() == "deny"


def test_webhook_mode_approve_when_configured():
    with mock_patch("hermes_cli.config.load_config",
                    return_value={"approvals": {"webhook_mode": "approve"}}):
        assert approval._get_webhook_approval_mode() == "approve"


def test_webhook_deny_blocks_dangerous_command_without_hanging():
    """A dangerous command in a webhook run (mode=deny) returns BLOCKED, not pending."""
    with mock_patch.object(approval, "_get_session_platform", lambda: "webhook"), \
         mock_patch.object(approval, "_get_webhook_approval_mode", lambda: "deny"), \
         mock_patch.object(approval, "_get_approval_mode", lambda: "manual"), \
         mock_patch.object(approval, "env_var_enabled",
                           lambda v: v in {"HERMES_GATEWAY_SESSION"}):
        # rm -rf of a non-root path: dangerous but not hardline
        result = approval.check_all_command_guards("rm -rf /tmp/somedir/data", "local")
    assert result["approved"] is False
    assert "webhook" in result["message"].lower()
    assert result.get("status") != "approval_required"
    assert "approval_pending" not in result


def test_webhook_approve_allows_dangerous_command():
    with mock_patch.object(approval, "_get_session_platform", lambda: "webhook"), \
         mock_patch.object(approval, "_get_webhook_approval_mode", lambda: "approve"), \
         mock_patch.object(approval, "_get_approval_mode", lambda: "manual"), \
         mock_patch.object(approval, "env_var_enabled",
                           lambda v: v in {"HERMES_GATEWAY_SESSION"}):
        result = approval.check_all_command_guards("rm -rf /tmp/somedir/data", "local")
    assert result["approved"] is True
    assert result.get("webhook_approved") is True


def test_webhook_deny_still_blocks_hardline():
    """webhook_mode=approve must NOT bypass the hardline floor (rm -rf /)."""
    with mock_patch.object(approval, "_get_session_platform", lambda: "webhook"), \
         mock_patch.object(approval, "_get_webhook_approval_mode", lambda: "approve"), \
         mock_patch.object(approval, "_get_approval_mode", lambda: "manual"):
        result = approval.check_all_command_guards("rm -rf /", "local")
    assert result["approved"] is False
    assert "hardline" in result["message"].lower()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as e:
                fails += 1
                print(f"[FAIL] {name}: {e!r}")
    print("ALL PASS" if not fails else f"{fails} FAILURE(S)")
    sys.exit(1 if fails else 0)
