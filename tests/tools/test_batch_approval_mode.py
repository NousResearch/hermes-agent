"""Tests for approvals.batch_mode — configurable approval behavior for batch runs.

batch_runner.py exports HERMES_BATCH_SESSION before constructing the agent; the
approval gate must resolve dangerous commands from ``approvals.batch_mode``
(default deny) instead of falling through to the legacy auto-approve path.
Mirrors tests/tools/test_cron_approval_mode.py.
"""

import pytest

import tools.approval as approval_module
from tools import approval_context
from tools.approval import check_all_command_guards, check_dangerous_command


def _batch_mode():
    # Attribute access at call time so the file collects (and the deny legs run
    # and fail behaviorally) on a base that lacks the batch context entirely.
    return approval_context._get_batch_approval_mode()


@pytest.fixture(autouse=True)
def _clear_approval_state():
    import os
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")
    # test_batch_runner_exports_marker calls the real _export_batch_approval_context(),
    # which writes os.environ directly (monkeypatch cannot restore that) — without this
    # pop the batch marker leaks into every later test file in the same pytest process.
    os.environ.pop("HERMES_BATCH_SESSION", None)


# ---------------------------------------------------------------------------
# _get_batch_approval_mode() config parsing
# ---------------------------------------------------------------------------

class TestBatchApprovalModeParsing:
    def test_default_is_deny(self):
        """When no config is set, batch_mode defaults to 'deny'."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {}}):
            assert _batch_mode() == "deny"

    def test_explicit_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"batch_mode": "approve"}}):
            assert _batch_mode() == "approve"

    def test_off_means_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"batch_mode": "off"}}):
            assert _batch_mode() == "approve"

    def test_uppercase_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"batch_mode": "APPROVE"}}):
            assert _batch_mode() == "approve"

    def test_invalid_value_falls_back_to_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"batch_mode": "maybe"}}):
            assert _batch_mode() == "deny"

    def test_bool_false_falls_back_to_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"batch_mode": False}}):
            assert _batch_mode() == "deny"


# ---------------------------------------------------------------------------
# check_dangerous_command() with batch session
# ---------------------------------------------------------------------------

class TestBatchDenyMode:
    """When HERMES_BATCH_SESSION is set and batch_mode=deny, dangerous commands are blocked."""

    def test_batch_marker_dangerous_command_resolves_from_batch_mode_not_legacy(self, monkeypatch):
        """The bug pin: with the marker exported (as batch_runner does), a dangerous
        command must resolve from approvals.batch_mode (default deny) instead of
        falling through to the legacy auto-approve. No mock: on a base without the
        batch context this fails with approved=True — the reported defect."""
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert not result["approved"]
        assert "batch_mode" in result["message"]

    def test_dangerous_command_blocked_in_batch_deny_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_batch_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]
            assert "batch_mode" in result["message"]

    def test_safe_command_allowed_in_batch_deny_mode(self, monkeypatch):
        """Non-dangerous commands still work even with batch_mode=deny."""
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_batch_approval_mode", return_value="deny"):
            result = check_dangerous_command("ls -la", "local")
            assert result["approved"]


class TestBatchApproveMode:
    """When HERMES_BATCH_SESSION is set and batch_mode=approve, dangerous commands pass through."""

    def test_dangerous_command_allowed_in_batch_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_batch_approval_mode", return_value="approve"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert result["approved"]


# ---------------------------------------------------------------------------
# check_all_command_guards() with batch session
# ---------------------------------------------------------------------------

class TestBatchDenyModeAllGuards:
    """The combined guard function also respects batch_mode."""

    def test_dangerous_command_blocked_in_combined_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_batch_approval_mode", return_value="deny"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]

    def test_safe_command_allowed_in_combined_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_batch_approval_mode", return_value="deny"):
            result = check_all_command_guards("echo hello", "local")
            assert result["approved"]

    def test_combined_guard_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_batch_approval_mode", return_value="approve"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert result["approved"]


# ---------------------------------------------------------------------------
# execute_code guard with batch session
# ---------------------------------------------------------------------------

class TestBatchExecuteCodeGuard:
    """A batch session must not silently auto-approve arbitrary code via the legacy path."""

    def test_execute_code_blocked_in_batch_deny_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_batch_approval_mode", return_value="deny"):
            result = approval_module.check_execute_code_guard("import os", "local")
            assert result["approved"] is False
            assert result["outcome"] == "blocked"

    def test_execute_code_allowed_in_batch_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_batch_approval_mode", return_value="approve"):
            result = approval_module.check_execute_code_guard("import os", "local")
            assert result["approved"] is True


# ---------------------------------------------------------------------------
# Evaluation order: batch resolves before cron when both markers are active
# ---------------------------------------------------------------------------

class TestBatchBeforeCronOrder:
    """batch_runner is the innermost runner: when both markers leak into one
    process, the batch context resolves first (pinned by message text)."""

    def test_batch_context_wins_over_cron_in_deny_message(self, monkeypatch):
        monkeypatch.setenv("HERMES_BATCH_SESSION", "1")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with (
            mock_patch("tools.approval_context._get_batch_approval_mode", return_value="deny"),
            mock_patch("tools.approval_context._get_cron_approval_mode", return_value="deny"),
        ):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "batch_mode" in result["message"]


# ---------------------------------------------------------------------------
# batch_runner wiring: the runner must mark its own process
# ---------------------------------------------------------------------------

class TestBatchRunnerWiring:
    def test_batch_runner_exports_marker(self, monkeypatch):
        """batch_runner marks its process so the gate consults approvals.batch_mode."""
        import os

        import batch_runner
        monkeypatch.delenv("HERMES_BATCH_SESSION", raising=False)
        batch_runner._export_batch_approval_context()
        assert os.environ.get("HERMES_BATCH_SESSION") == "1"

    def test_non_batch_blank_context_keeps_legacy_approved(self, monkeypatch):
        """No batch marker (unknown context) keeps the legacy auto-approve contract.

        Global fail-closed reversal of unknown contexts is report #3 of the
        upstream issue and is deliberately out of scope here.
        """
        monkeypatch.delenv("HERMES_BATCH_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert result["approved"]
