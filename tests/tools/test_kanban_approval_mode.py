"""Tests for approvals.kanban_mode — approval behavior for kanban dispatcher workers (#129818)."""

from unittest.mock import patch as mock_patch

import pytest

import tools.approval as approval_module
from tools import approval_context
from tools.approval import (
    check_all_command_guards, check_dangerous_command, check_execute_code_guard, request_tool_approval,
)
from tools.approval_context import _get_kanban_approval_mode


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")


@pytest.fixture()
def kanban_env(monkeypatch):
    """A dispatcher-spawned worker: HERMES_KANBAN_TASK set, no human anywhere."""
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "T-42")
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    return monkeypatch


# ---------------------------------------------------------------------------
# _get_kanban_approval_mode() config parsing
# ---------------------------------------------------------------------------

class TestKanbanApprovalModeParsing:
    def test_default_is_deny(self):
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {}}):
            assert _get_kanban_approval_mode() == "deny"

    def test_explicit_approve(self):
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "approve"}}):
            assert _get_kanban_approval_mode() == "approve"

    def test_off_maps_to_approve(self):
        """'off' is an alias for 'approve' (matches --yolo semantics)."""
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "off"}}):
            assert _get_kanban_approval_mode() == "approve"

    def test_unknown_value_defaults_to_deny(self):
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "maybe"}}):
            assert _get_kanban_approval_mode() == "deny"

    def test_config_load_failure_defaults_to_deny(self):
        with mock_patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("config broken")):
            assert _get_kanban_approval_mode() == "deny"


# ---------------------------------------------------------------------------
# Context detection
# ---------------------------------------------------------------------------

class TestKanbanContextDetection:
    def test_kanban_task_env_marks_context(self, kanban_env):
        assert approval_module._is_kanban_approval_context() is True

    def test_blank_task_id_is_not_kanban(self, kanban_env):
        kanban_env.setenv("HERMES_KANBAN_TASK", "   ")
        assert approval_module._is_kanban_approval_context() is False

    def test_absent_task_id_is_not_kanban(self, kanban_env):
        kanban_env.delenv("HERMES_KANBAN_TASK")
        assert approval_module._is_kanban_approval_context() is False

    def test_kanban_is_never_a_gateway_approval_context(self, kanban_env):
        """A leaked gateway marker must not turn a worker into an answerable surface."""
        kanban_env.setenv("HERMES_GATEWAY_SESSION", "1")
        assert approval_module._is_gateway_approval_context() is False

    def test_cron_beats_kanban(self, kanban_env):
        """A cron-scheduled kanban worker resolves as cron (evaluation order)."""
        kanban_env.setenv("HERMES_CRON_SESSION", "1")
        names = [ctx.name for ctx in approval_module._unattended_contexts()]
        assert "cron" in names and "kanban" not in names

    def test_kanban_beats_unattended_platform(self, kanban_env):
        from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
        reset_session_vars()
        tokens = set_session_vars(platform="webhook")
        try:
            names = [ctx.name for ctx in approval_module._unattended_contexts()]
        finally:
            clear_session_vars(tokens)
            reset_session_vars()
        assert names == ["kanban"]


# ---------------------------------------------------------------------------
# Gate behavior — deny (default) blocks instead of silently auto-approving
# ---------------------------------------------------------------------------

class TestKanbanDenyMode:
    def test_dangerous_command_blocked(self, kanban_env):
        kanban_env.setattr(approval_context, "_get_kanban_approval_mode", lambda: "deny")
        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert result["approved"] is False
        assert "kanban" in (result.get("message") or "")

    def test_all_command_guards_blocked(self, kanban_env):
        kanban_env.setattr(approval_context, "_get_kanban_approval_mode", lambda: "deny")
        result = check_all_command_guards("rm -rf /tmp/stuff", "local")
        assert result["approved"] is False

    def test_execute_code_blocked(self, kanban_env):
        kanban_env.setattr(approval_context, "_get_kanban_approval_mode", lambda: "deny")
        result = check_execute_code_guard("import os", "local")
        assert result["approved"] is False
        assert result["outcome"] == "blocked"

    def test_plugin_tool_approval_fails_closed(self, kanban_env):
        kanban_env.setattr(approval_context, "_get_kanban_approval_mode", lambda: "deny")
        result = request_tool_approval("some_tool", "plugin rule")
        assert result["approved"] is False

    def test_default_config_blocks_headless_auto_approve(self, kanban_env):
        """Regression for the headline bug: with no config at all, the worker must NOT
        fall through to 'AUTO-APPROVED ... in non-interactive non-gateway context'."""
        kanban_env.setattr(approval_context, "_get_kanban_approval_mode", _get_kanban_approval_mode)
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {}}):
            result = check_dangerous_command("rm -rf /tmp/some-test-dir", "local")
        assert result["approved"] is False


# ---------------------------------------------------------------------------
# Gate behavior — explicit approve keeps trusted workers working
# ---------------------------------------------------------------------------

class TestKanbanApproveMode:
    def test_dangerous_command_auto_approved(self, kanban_env):
        kanban_env.setattr(approval_context, "_get_kanban_approval_mode", lambda: "approve")
        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert result["approved"] is True

    def test_execute_code_allowed(self, kanban_env):
        kanban_env.setattr(approval_context, "_get_kanban_approval_mode", lambda: "approve")
        result = check_execute_code_guard("import os", "local")
        assert result["approved"] is True

    def test_hardline_floor_still_blocks_under_approve(self, kanban_env):
        """kanban_mode: approve relaxes the approval layer only — the unconditional floors
        (hardline commands, user deny rules) are evaluated before any mode bypass."""
        kanban_env.setattr(approval_context, "_get_kanban_approval_mode", lambda: "approve")
        result = check_dangerous_command("rm -rf /", "local")
        assert result["approved"] is False
