"""Approval on /v1/runs turns: the client answers through the approval notifier."""

import threading

import pytest

import tools.approval as approval_module
from tools import approval_context


class TestRunsApprovalBridge:
    """/v1/runs registers an approval notifier, so flagged actions wait for the client."""

    SESSION_KEY = "test-runs-session"

    @pytest.fixture(autouse=True)
    def _runs_session(self, monkeypatch):
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
        monkeypatch.setattr(
            approval_context, "_get_approval_config",
            lambda: {"mode": "manual", "timeout": 5},
        )
        for name in ("HERMES_CRON_SESSION", "HERMES_GATEWAY_SESSION", "HERMES_INTERACTIVE"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")  # set by start_gateway()
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")
        monkeypatch.setenv("HERMES_SESSION_KEY", self.SESSION_KEY)
        yield
        approval_module.unregister_gateway_notify(self.SESSION_KEY)

    def _answer_with(self, choice):
        """Register a notifier that answers like POST /v1/runs/{id}/approval."""
        events = []

        def notify(data):
            events.append(data)
            threading.Timer(
                0.05, approval_module.resolve_gateway_approval, (self.SESSION_KEY, choice)
            ).start()

        approval_module.register_gateway_notify(self.SESSION_KEY, notify)
        return events

    def test_execute_code_waits_then_approves_once(self):
        events = self._answer_with("once")
        result = approval_module.check_execute_code_guard("import os", "local")
        assert len(events) == 1
        assert result["approved"] is True

    def test_execute_code_waits_then_blocks_on_deny(self):
        events = self._answer_with("deny")
        result = approval_module.check_execute_code_guard("import os", "local")
        assert len(events) == 1
        assert result["approved"] is False
        assert "denied by user" in result["message"]

    def test_execute_code_without_notifier_denies_instantly(self):
        result = approval_module.check_execute_code_guard("import os", "local")
        assert result["approved"] is False
        assert "approvals.unattended_mode" in result["message"]

    def test_shell_command_still_waits(self):
        events = self._answer_with("once")
        result = approval_module.check_all_command_guards("rm -rf /tmp/hermes-test-x", "local")
        assert len(events) == 1
        assert result["approved"] is True

    def test_execute_code_times_out_to_deny(self, monkeypatch):
        monkeypatch.setattr(
            approval_context, "_get_approval_config",
            lambda: {"mode": "manual", "timeout": 0.05},
        )
        events = []
        approval_module.register_gateway_notify(self.SESSION_KEY, events.append)
        result = approval_module.check_execute_code_guard("import os", "local")
        assert len(events) == 1
        assert result["approved"] is False
        assert "timed out" in result["message"]

    def test_webhook_with_notifier_still_denies(self, monkeypatch):
        """webhook turns register a notifier but no client can answer it."""
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "webhook")
        events = self._answer_with("once")
        result = approval_module.check_execute_code_guard("import os", "local")
        assert events == []
        assert result["approved"] is False
        assert "approvals.unattended_mode" in result["message"]

    def test_cron_with_notifier_still_denies(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        events = self._answer_with("once")
        result = approval_module.check_execute_code_guard("import os", "local")
        assert events == []
        assert result["approved"] is False
        assert "approvals.cron_mode" in result["message"]
