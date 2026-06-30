"""Tests for approvals.kanban_mode — configurable approval behavior for
kanban-dispatched worker subprocesses.

Mirrors test_cron_approval_mode.py: kanban workers are the same kind of
unattended-execution context cron jobs are (no user present to approve), and
previously fell through to the bare non-interactive auto-approve branch
because no approval-context flag was ever set in their environment.
"""

import pytest

import tools.approval as approval_module
from tools.approval import (
    _get_kanban_approval_mode,
    check_all_command_guards,
    check_dangerous_command,
    detect_dangerous_command,
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


# ---------------------------------------------------------------------------
# _get_kanban_approval_mode() config parsing
# ---------------------------------------------------------------------------

class TestKanbanApprovalModeParsing:
    def test_default_is_deny(self):
        """When no config is set, kanban_mode defaults to 'deny'."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {}}):
            assert _get_kanban_approval_mode() == "deny"

    def test_explicit_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"kanban_mode": "deny"}}):
            assert _get_kanban_approval_mode() == "deny"

    def test_explicit_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"kanban_mode": "approve"}}):
            assert _get_kanban_approval_mode() == "approve"

    def test_off_maps_to_approve(self):
        """'off' is an alias for 'approve' (matches --yolo semantics)."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"kanban_mode": "off"}}):
            assert _get_kanban_approval_mode() == "approve"

    def test_unknown_value_defaults_to_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"kanban_mode": "maybe"}}):
            assert _get_kanban_approval_mode() == "deny"

    def test_config_load_failure_defaults_to_deny(self):
        """If config loading fails entirely, default to deny (safe)."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", side_effect=RuntimeError("config broken")):
            assert _get_kanban_approval_mode() == "deny"

    def test_yaml_boolean_false_maps_to_deny(self):
        """YAML 1.1 parses bare 'off' as False. Ensure it maps to deny."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"kanban_mode": False}}):
            assert _get_kanban_approval_mode() == "deny"


# ---------------------------------------------------------------------------
# check_dangerous_command() with kanban worker session
# ---------------------------------------------------------------------------

class TestKanbanDenyMode:
    """When HERMES_KANBAN_SESSION is set and kanban_mode=deny, dangerous commands are blocked."""

    def test_dangerous_command_blocked_in_kanban_deny_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]
            assert "kanban_mode" in result["message"]

    def test_safe_command_allowed_in_kanban_deny_mode(self, monkeypatch):
        """Non-dangerous commands still work even with kanban_mode=deny."""
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="deny"):
            result = check_dangerous_command("ls -la", "local")
            assert result["approved"]

    def test_world_writable_chmod_blocked(self, monkeypatch):
        """The exact pattern from the live repro: chmod 777 on a kanban worker."""
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="deny"):
            result = check_dangerous_command("chmod 777 /tmp/some-file.txt", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]

    def test_block_message_includes_description(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "dangerous" in result["message"].lower() or "delete" in result["message"].lower()


class TestKanbanApproveMode:
    """When HERMES_KANBAN_SESSION is set and kanban_mode=approve, dangerous commands pass through."""

    def test_dangerous_command_allowed_in_kanban_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="approve"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert result["approved"]


# ---------------------------------------------------------------------------
# check_all_command_guards() with kanban worker session
# ---------------------------------------------------------------------------

class TestKanbanDenyModeAllGuards:
    """The combined guard function also respects kanban_mode."""

    def test_dangerous_command_blocked_in_combined_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="deny"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]

    def test_safe_command_allowed_in_combined_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="deny"):
            result = check_all_command_guards("echo hello", "local")
            assert result["approved"]

    def test_combined_guard_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="approve"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert result["approved"]


# ---------------------------------------------------------------------------
# Edge cases: kanban mode interaction with other approval mechanisms
# ---------------------------------------------------------------------------

class TestKanbanModeInteractions:
    """Kanban mode should NOT interfere with other approval bypass mechanisms,
    and must not regress the pre-existing cron/CLI/gateway/non-interactive
    behavior this gap was originally found alongside.
    """

    def test_container_env_still_auto_approves(self, monkeypatch):
        """Docker/sandbox environments bypass approvals regardless of kanban_mode."""
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_kanban_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /", "docker")
            assert result["approved"]

    def test_yolo_overrides_kanban_deny(self, monkeypatch):
        """--yolo still bypasses kanban_mode=deny for dangerous (non-hardline) commands."""
        monkeypatch.setenv("HERMES_KANBAN_SESSION", "1")
        monkeypatch.setenv("HERMES_YOLO_MODE", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

        from unittest.mock import patch as mock_patch
        import tools.approval
        with (
            mock_patch.object(tools.approval, "_YOLO_MODE_FROZEN", True),
            mock_patch("tools.approval._get_kanban_approval_mode", return_value="deny"),
        ):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert result["approved"]

    def test_non_kanban_non_interactive_still_auto_approves(self, monkeypatch):
        """Non-kanban, non-cron, non-interactive sessions (e.g. scripted usage)
        still auto-approve -- this fix narrows the gap to kanban workers only,
        it does not change the pre-existing scripted-usage contract.
        """
        monkeypatch.delenv("HERMES_KANBAN_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert result["approved"]

    def test_cron_session_unaffected_by_kanban_changes(self, monkeypatch):
        """Sanity check: adding the kanban branch must not disturb the
        pre-existing, already-tested cron_mode behavior.
        """
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_KANBAN_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "cron_mode" in result["message"]
