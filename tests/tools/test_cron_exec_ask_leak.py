"""Regression tests for HERMES_EXEC_ASK leaking into cron approval path.

The gateway sets ``HERMES_EXEC_ASK=1`` process-wide at startup
(``gateway/run.py``).  Cron jobs run inside the gateway process, so they
inherit ``HERMES_EXEC_ASK``.  In ``check_all_command_guards()``, the
``cron_mode`` check was nested inside ``if not is_ask:``, which was always
``False`` when ``HERMES_EXEC_ASK`` was set — meaning ``cron_mode: deny`` was
completely bypassed for cron jobs running inside the gateway (the standard
deployment).  Dangerous commands entered the gateway approval path, which
blocks waiting for a user response that never comes — 300s timeout.

The existing tests in ``test_cron_approval_mode.py`` all do
``monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)``, which artificially
removes the env var and masks the bug.  These tests set it to ``"1"``,
matching production.
"""

import pytest

import tools.approval as approval_module
from tools.approval import (
    check_all_command_guards,
    check_dangerous_command,
)


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")


class TestCronWithExecAskLeak:
    """Cron jobs inherit HERMES_EXEC_ASK from the gateway process.

    These tests reproduce the production environment: both
    HERMES_CRON_SESSION and HERMES_EXEC_ASK are set simultaneously.
    """

    def test_dangerous_command_blocked_when_exec_ask_is_set(self, monkeypatch):
        """cron_mode=deny must block dangerous commands even when
        HERMES_EXEC_ASK=1 is present (the real gateway+cron scenario).

        Before the fix, HERMES_EXEC_ASK caused is_ask=True, which skipped
        the cron_mode branch and routed to the gateway approval path
        (submit_pending → 300s timeout, no listener).
        """
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")  # leaked from gateway
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"], (
                "Dangerous command must be blocked in cron+deny mode even "
                "when HERMES_EXEC_ASK is set (gateway leak scenario)"
            )
            assert "BLOCKED" in result["message"]
            assert result.get("status") != "approval_required"

    def test_safe_command_allowed_when_exec_ask_is_set(self, monkeypatch):
        """Safe commands still pass through in cron mode with HERMES_EXEC_ASK."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_dangerous_command("echo hello", "local")
            assert result["approved"]

    def test_combined_guard_blocks_with_exec_ask_set(self, monkeypatch):
        """check_all_command_guards must also honor cron_mode over is_ask."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]
            assert result.get("status") != "approval_required"

    def test_combined_guard_allows_safe_with_exec_ask_set(self, monkeypatch):
        """check_all_command_guards allows safe commands with HERMES_EXEC_ASK."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_all_command_guards("echo hello", "local")
            assert result["approved"]

    def test_cron_approve_mode_passthrough_with_exec_ask(self, monkeypatch):
        """cron_mode=approve still allows dangerous commands when HERMES_EXEC_ASK is set."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="approve"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert result["approved"]

    def test_git_operations_allowed_in_cron_deny_with_exec_ask(self, monkeypatch):
        """Non-dangerous git operations (commit, push, log) must pass through
        in cron+deny mode even with HERMES_EXEC_ASK set.  This was the
        user-visible symptom: ``git commit`` and ``git push`` (non-force)
        triggered 300s timeouts because they entered the gateway approval
        path instead of being auto-approved as safe commands.
        """
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            for cmd in [
                "git commit -S -m 'test message'",
                "git push",
                "git add VISION.md",
                "git log --oneline",
            ]:
                result = check_all_command_guards(cmd, "local")
                assert result["approved"], (
                    f"Safe git command must be allowed in cron mode: {cmd}"
                )
