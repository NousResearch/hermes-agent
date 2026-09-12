"""Tests for approvals.kanban_mode — configurable approval behavior for
kanban dispatcher workers.

Background (#106993): the kanban dispatcher spawns workers as
``hermes ... chat -q "work kanban task <id>"`` and also sets
``HERMES_KANBAN_TASK`` + ``HERMES_SESSION_SOURCE=kanban``. The ``-q`` flag
sets ``HERMES_SINGLE_QUERY_SESSION``, so workers previously hit the
single-query unattended approval context. With default
``approvals.single_query_mode: deny``, ``check_execute_code_guard`` blocked
``execute_code`` on every card — even when the profile granted
``code_execution``. Kanban workers and ad-hoc ``chat -q`` are different
trust objects; ``approvals.kanban_mode`` (default ``deny``) governs the
kanban path and wins over ``single_query_mode`` when kanban markers are
present.
"""

import pytest

import tools.approval as approval_module
from tools import approval_context
from gateway.session_context import reset_session_vars
from tools.approval import check_all_command_guards, check_dangerous_command


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")
    reset_session_vars()
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")
    reset_session_vars()


def _get_kanban_approval_mode():
    from tools.approval_context import _get_kanban_approval_mode as _impl
    return _impl()


# ---------------------------------------------------------------------------
# _get_kanban_approval_mode() config parsing
# ---------------------------------------------------------------------------

class TestKanbanApprovalModeParsing:
    def test_default_is_deny(self):
        """When no config is set, kanban_mode defaults to 'deny'."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {}}):
            assert _get_kanban_approval_mode() == "deny"

    def test_explicit_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "deny"}}):
            assert _get_kanban_approval_mode() == "deny"

    def test_explicit_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "approve"}}):
            assert _get_kanban_approval_mode() == "approve"

    def test_off_maps_to_approve(self):
        """'off' is an alias for 'approve' (matches --yolo semantics)."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "off"}}):
            assert _get_kanban_approval_mode() == "approve"

    def test_allow_maps_to_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "allow"}}):
            assert _get_kanban_approval_mode() == "approve"

    def test_yes_maps_to_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "yes"}}):
            assert _get_kanban_approval_mode() == "approve"

    def test_unknown_value_defaults_to_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"kanban_mode": "maybe"}}):
            assert _get_kanban_approval_mode() == "deny"

    def test_config_load_failure_defaults_to_deny(self):
        """If config loading fails entirely, default to deny (safe)."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("config broken")):
            assert _get_kanban_approval_mode() == "deny"


# ---------------------------------------------------------------------------
# Kanban context detection
# ---------------------------------------------------------------------------

class TestKanbanContextDetection:
    def test_kanban_task_env_marks_context(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
        assert approval_context._is_kanban_approval_context() is True

    def test_env_var_unset_is_not_kanban(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
        assert approval_context._is_kanban_approval_context() is False

    def test_session_source_kanban_marks_context(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
        assert approval_context._is_kanban_approval_context() is True


# ---------------------------------------------------------------------------
# Bug repro (#106993): kanban + -q must not be governed by single_query_mode
# ---------------------------------------------------------------------------

class TestKanbanWinsOverSingleQuery:
    """Real dispatcher spawn sets BOTH -q and kanban env. Kanban context must
    win so ``kanban_mode: approve`` can unlock execute_code while
    ``single_query_mode`` stays deny for ad-hoc ``chat -q``."""

    def test_bug_repro_kanban_approve_beats_single_query_deny(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)

        from unittest.mock import patch as mock_patch
        with mock_patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"approvals": {"single_query_mode": "deny", "kanban_mode": "approve"}},
        ):
            result = approval_module.check_execute_code_guard("import os", "local")
        assert result["approved"] is True

    def test_control_dual_env_kanban_deny_blocks(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)

        from unittest.mock import patch as mock_patch
        with mock_patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"approvals": {"single_query_mode": "deny", "kanban_mode": "deny"}},
        ):
            result = approval_module.check_execute_code_guard("import os", "local")
        assert result["approved"] is False
        assert result.get("outcome") == "blocked"
        assert "BLOCKED" in result["message"]
        assert "kanban_mode" in result["message"] or "kanban" in result["message"].lower()

    def test_control_adhoc_q_still_uses_single_query_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)

        from unittest.mock import patch as mock_patch
        with mock_patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"approvals": {"single_query_mode": "deny", "kanban_mode": "approve"}},
        ):
            result = approval_module.check_execute_code_guard("import os", "local")
        assert result["approved"] is False
        assert "single_query_mode" in result["message"]


# ---------------------------------------------------------------------------
# Optional: dangerous terminal commands follow the same kanban context
# ---------------------------------------------------------------------------

class TestKanbanCommandGuards:
    def test_dangerous_command_blocked_in_kanban_deny(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)

        from unittest.mock import patch as mock_patch
        with mock_patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"approvals": {"single_query_mode": "deny", "kanban_mode": "deny"}},
        ):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
        assert not result["approved"]
        assert "BLOCKED" in result["message"]
        assert "kanban_mode" in result["message"]

    def test_dangerous_command_allowed_in_kanban_approve(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)

        from unittest.mock import patch as mock_patch
        with mock_patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"approvals": {"single_query_mode": "deny", "kanban_mode": "approve"}},
        ):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert result["approved"]
