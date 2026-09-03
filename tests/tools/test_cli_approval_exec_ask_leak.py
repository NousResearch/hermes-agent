"""Regression: interactive CLI must not lose the Dangerous Command panel.

When ``HERMES_EXEC_ASK`` leaks into a classic CLI process (historically via
``import gateway.run`` setting the flag at module import), the ask/gateway
branch used to return ``pending_approval`` immediately with no notify
listener and skip the CLI approval callback. Users saw tools "auto-block"
with no Approve/Deny UI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools.approval import check_all_command_guards, check_execute_code_guard
from tools.terminal_tool import set_approval_callback


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_approval_env(monkeypatch):
    for key in (
        "HERMES_EXEC_ASK",
        "HERMES_GATEWAY_SESSION",
        "HERMES_SESSION_PLATFORM",
        "HERMES_CRON_SESSION",
        "HERMES_YOLO_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        approval_module,
        "_get_approval_mode",
        lambda: "manual",
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
    )
    approval_module._session_approved.clear()
    approval_module._permanent_approved.clear()
    approval_module._pending.clear()
    # The consecutive-denial breaker is process-global; a tally left behind by
    # another test would leak its escalated addendum into these assertions.
    approval_module._denial_tally.clear()
    set_approval_callback(None)
    yield
    approval_module._denial_tally.clear()
    set_approval_callback(None)


class TestCliApprovalSurvivesExecAskLeak:
    def test_cli_callback_used_when_exec_ask_set_without_notifier(self, monkeypatch):
        """Ask-mode with a CLI callback must prompt locally, not pending_approval."""
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        calls = []

        def _cb(command, description, **kwargs):
            calls.append((command, description))
            return "once"

        set_approval_callback(_cb)
        result = check_all_command_guards("rm -rf /tmp/testdir", "local")

        assert calls, "CLI approval callback was never invoked"
        assert result.get("status") != "pending_approval"
        assert result.get("approval_pending") is not True
        assert result.get("approved") is True
        assert result.get("user_approved") is True

    def test_pending_approval_still_used_without_cli_callback(self, monkeypatch):
        """Headless ask-mode without a CLI callback keeps the pending fallback."""
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        set_approval_callback(None)

        result = check_all_command_guards("rm -rf /tmp/testdir", "local")

        assert result.get("approved") is False
        assert result.get("status") == "pending_approval"
        assert result.get("approval_pending") is True


class TestExecuteCodeGuardCliApprovalSurvivesExecAskLeak:
    """check_execute_code_guard (the whole-script gate) had its own,
    unfixed copy of the same notify_cb-less short-circuit — sibling of
    check_all_command_guards, same leak, same missing CLI fall-through."""

    def test_cli_callback_used_when_exec_ask_set_without_notifier(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        calls = []

        def _cb(command, description, **kwargs):
            calls.append((command, description))
            return "once"

        set_approval_callback(_cb)
        result = check_execute_code_guard("print('hi')", "local")

        assert calls, "CLI approval callback was never invoked"
        assert result.get("status") != "pending_approval"
        assert result.get("approval_pending") is not True
        assert result.get("approved") is True
        assert result.get("user_approved") is True

    def test_cli_callback_deny_blocks_execution(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        set_approval_callback(lambda command, description, **kwargs: "deny")

        result = check_execute_code_guard("print('hi')", "local")

        assert result.get("approved") is False
        assert result.get("outcome") == "denied"
        assert result.get("status") != "pending_approval"

    def test_cli_callback_timeout_blocks_execution(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        set_approval_callback(lambda command, description, **kwargs: "timeout")

        result = check_execute_code_guard("print('hi')", "local")

        assert result.get("approved") is False
        assert result.get("outcome") == "timeout"

    def test_cli_callback_session_choice_persists_approval(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        set_approval_callback(lambda command, description, **kwargs: "session")

        first = check_execute_code_guard("print('hi')", "local")
        assert first.get("approved") is True

        # A second call in the same session must short-circuit on the
        # session-approval cache without prompting again.
        set_approval_callback(None)
        second = check_execute_code_guard("print('again')", "local")
        assert second.get("approved") is True
        assert second.get("status") != "pending_approval"

    def test_pending_approval_still_used_without_cli_callback(self, monkeypatch):
        """Headless ask-mode without a CLI callback keeps the pending fallback."""
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        set_approval_callback(None)

        result = check_execute_code_guard("print('hi')", "local")

        assert result.get("approved") is False
        assert result.get("status") == "pending_approval"
        assert result.get("approval_pending") is True

    def test_cli_callback_used_for_platform_marker_leak_without_exec_ask(
        self, monkeypatch
    ):
        """The other half of the leak: a session platform marker, no ask-mode.

        ``_is_gateway_approval_context()`` is true whenever
        ``HERMES_SESSION_PLATFORM`` is set, so the whole-script gate is
        reached with ``HERMES_EXEC_ASK`` entirely absent. That path must
        show the CLI panel too, not a silent pending approval.
        """
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        calls = []

        def _cb(command, description, **kwargs):
            calls.append((command, description))
            return "once"

        set_approval_callback(_cb)
        result = check_execute_code_guard("print('marker')", "local")

        assert calls, "CLI approval callback was never invoked"
        assert result.get("approved") is True
        assert result.get("status") != "pending_approval"
        assert result.get("approval_pending") is not True


class TestExecuteCodeGuardCliDenialBreakerParity:
    """The CLI fall-through must match its sibling guards' breaker semantics.

    The consecutive-denial breaker counts *guardian LLM* DENY verdicts, so a
    deliberate human deny must not advance the tally; and once the tally has
    tripped, the escalated addendum belongs on the timeout message too (the
    same function's gateway arm and ``check_all_command_guards``' CLI tail
    both include it).
    """

    def test_human_deny_does_not_advance_the_guardian_breaker(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        set_approval_callback(lambda command, description, **kwargs: "deny")

        for _ in range(3):
            result = check_execute_code_guard("print('hi')", "local")
            assert result.get("outcome") == "denied"

        assert not approval_module._denial_tally, (
            "a human deny must not advance the guardian-verdict breaker "
            "(check_all_command_guards does not)"
        )

    def test_timeout_message_carries_breaker_addendum_once_tripped(
        self, monkeypatch
    ):
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        session_key = approval_module.get_current_session_key()
        threshold = approval_module._get_denial_breaker_threshold()
        for _ in range(threshold):
            approval_module._record_denial(session_key)
        expected = approval_module._denial_breaker_addendum(session_key)
        assert expected, "breaker should be tripped for this fixture"

        set_approval_callback(lambda command, description, **kwargs: "timeout")
        result = check_execute_code_guard("print('hi')", "local")

        assert result.get("outcome") == "timeout"
        assert expected in (result.get("message") or "")


class TestGatewayRunImportDoesNotSetExecAsk:
    def test_importing_gateway_run_does_not_set_exec_ask(self, tmp_path):
        """Incidental imports must not poison CLI ask-mode process-wide."""
        script = r"""
import os, sys
os.environ.pop("HERMES_EXEC_ASK", None)
sys.path.insert(0, %r)
# Avoid starting the gateway; only import the module for _gateway_runner_ref
# style side imports.
import gateway.run  # noqa: F401
print("EXEC_ASK=" + repr(os.environ.get("HERMES_EXEC_ASK")))
""" % (str(REPO_ROOT),)
        hermes_home = tmp_path / "import-test-home"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "HERMES_HOME": str(hermes_home),
            },
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert "EXEC_ASK=None" in proc.stdout, proc.stdout + proc.stderr
