"""Tests for approvals.autonomous_delegation_mode.

When a gateway turn is driven by another bot (an in-process fleet delegation)
rather than an interactive human, a dangerous non-allowlisted command would
otherwise post an approval card and hang until the consent timeout (~60s)
before failing — nobody is present to answer it. set_delegated_autonomous
marks such a turn; approvals.autonomous_delegation_mode (default deny, like
cron_mode) then fails it fast via the same _Unattended machinery cron and
single-query already use.
"""

import pytest

import tools.approval as approval_module
from tools.approval import check_all_command_guards, check_dangerous_command
from tools.approval_context import (
    _get_autonomous_delegation_approval_mode,
    _is_autonomous_delegation_context,
    reset_delegated_autonomous,
    set_delegated_autonomous,
)


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")


# ---------------------------------------------------------------------------
# _get_autonomous_delegation_approval_mode() config parsing — delegates to the
# same _binary_approval_mode() helper cron_mode/single_query_mode use, so this
# only needs to prove the delegation, not re-prove every parsing edge case.
# ---------------------------------------------------------------------------

class TestAutonomousDelegationModeParsing:
    def test_default_is_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {}}):
            assert _get_autonomous_delegation_approval_mode() == "deny"

    def test_explicit_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"approvals": {"autonomous_delegation_mode": "approve"}},
        ):
            assert _get_autonomous_delegation_approval_mode() == "approve"

    def test_config_load_failure_defaults_to_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("broken")):
            assert _get_autonomous_delegation_approval_mode() == "deny"


# ---------------------------------------------------------------------------
# ContextVar set/reset/detect
# ---------------------------------------------------------------------------

class TestAutonomousDelegationContextVar:
    def test_defaults_false(self):
        assert _is_autonomous_delegation_context() is False

    def test_set_and_reset_round_trip(self):
        token = set_delegated_autonomous(True)
        try:
            assert _is_autonomous_delegation_context() is True
        finally:
            reset_delegated_autonomous(token)
        assert _is_autonomous_delegation_context() is False

    def test_set_false_is_a_no_op_marker(self):
        token = set_delegated_autonomous(False)
        try:
            assert _is_autonomous_delegation_context() is False
        finally:
            reset_delegated_autonomous(token)

    def test_nested_set_reset_restores_the_outer_value(self):
        """A nested set/reset (e.g. two turns overlapping on the same task, or a test
        exercising the helpers directly) must restore exactly the enclosing value, not
        clobber it to the global default — same contract as reset_current_session_key,
        called from the identical _run_conversation_with_approval `finally` block."""
        outer_token = set_delegated_autonomous(True)
        try:
            assert _is_autonomous_delegation_context() is True
            inner_token = set_delegated_autonomous(False)
            try:
                assert _is_autonomous_delegation_context() is False
            finally:
                reset_delegated_autonomous(inner_token)
            assert _is_autonomous_delegation_context() is True
        finally:
            reset_delegated_autonomous(outer_token)
        assert _is_autonomous_delegation_context() is False


# ---------------------------------------------------------------------------
# check_dangerous_command() / check_all_command_guards() under delegation
# ---------------------------------------------------------------------------

class TestAutonomousDelegationDenyMode:
    def test_dangerous_command_blocked_when_delegated_and_deny(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        token = set_delegated_autonomous(True)
        try:
            with mock_patch(
                "tools.approval_context._get_autonomous_delegation_approval_mode",
                return_value="deny",
            ):
                result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        finally:
            reset_delegated_autonomous(token)

        assert not result["approved"]
        assert "BLOCKED" in result["message"]
        assert "autonomous_delegation_mode" in result["message"]

    def test_safe_command_allowed_when_delegated_and_deny(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)

        from unittest.mock import patch as mock_patch
        token = set_delegated_autonomous(True)
        try:
            with mock_patch(
                "tools.approval_context._get_autonomous_delegation_approval_mode",
                return_value="deny",
            ):
                result = check_dangerous_command("ls -la", "local")
        finally:
            reset_delegated_autonomous(token)

        assert result["approved"]

    def test_combined_guard_blocked_when_delegated_and_deny(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        token = set_delegated_autonomous(True)
        try:
            with mock_patch(
                "tools.approval_context._get_autonomous_delegation_approval_mode",
                return_value="deny",
            ):
                result = check_all_command_guards("rm -rf /tmp/stuff", "local")
        finally:
            reset_delegated_autonomous(token)

        assert not result["approved"]
        assert "BLOCKED" in result["message"]


class TestAutonomousDelegationApproveMode:
    def test_dangerous_command_allowed_when_delegated_and_approve(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)

        from unittest.mock import patch as mock_patch
        token = set_delegated_autonomous(True)
        try:
            with mock_patch(
                "tools.approval_context._get_autonomous_delegation_approval_mode",
                return_value="approve",
            ):
                result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        finally:
            reset_delegated_autonomous(token)

        assert result["approved"]


class TestAutonomousDelegationDoesNotAffectHumanTurns:
    def test_dangerous_command_still_prompts_when_not_delegated(self, monkeypatch):
        """The default (unset) contextvar must not change any existing behaviour: a
        dangerous command outside cron/single-query/unattended still goes through the
        normal interactive approval flow, not an instant deny."""
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        assert _is_autonomous_delegation_context() is False
        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        # Non-interactive, non-gateway, non-cron, non-delegated -> the existing
        # "no presence at all" auto-approve path (same as today, unaffected).
        assert result["approved"]
