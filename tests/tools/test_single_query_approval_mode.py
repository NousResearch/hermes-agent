"""Tests for approvals.single_query_mode — configurable approval behavior for
single-query (-q) sessions.

Background (#86878): ``hermes chat -q "..."`` runs one turn and exits. cli.py
exports ``HERMES_INTERACTIVE=1`` (needed for interactive sudo password
prompts), which previously made ``_is_interactive_cli()`` report True in the
approval gate. A -q run has NO user waiting to answer approval prompts, so a
dangerous command just waited the full timeout (300s) then failed closed — and
the agent was effectively forced to work around the block (often silently
auto-approving via ``execute_code``, which auto-approves in non-gateway mode).
``approvals.single_query_mode`` (default ``deny``, mirror of cron_mode) makes
that decision deterministic and explicit.
"""

import pytest

import tools.approval as approval_module
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.approval import (
    _get_single_query_approval_mode,
    check_all_command_guards,
    check_dangerous_command,
    detect_dangerous_command,
)


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


# ---------------------------------------------------------------------------
# _get_single_query_approval_mode() config parsing
# ---------------------------------------------------------------------------

class TestSingleQueryApprovalModeParsing:
    def test_default_is_deny(self):
        """When no config is set, single_query_mode defaults to 'deny'."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {}}):
            assert _get_single_query_approval_mode() == "deny"

    def test_explicit_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"single_query_mode": "deny"}}):
            assert _get_single_query_approval_mode() == "deny"

    def test_explicit_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"single_query_mode": "approve"}}):
            assert _get_single_query_approval_mode() == "approve"

    def test_off_maps_to_approve(self):
        """'off' is an alias for 'approve' (matches --yolo semantics)."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"single_query_mode": "off"}}):
            assert _get_single_query_approval_mode() == "approve"

    def test_allow_maps_to_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"single_query_mode": "allow"}}):
            assert _get_single_query_approval_mode() == "approve"

    def test_yes_maps_to_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"single_query_mode": "yes"}}):
            assert _get_single_query_approval_mode() == "approve"

    def test_case_insensitive(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"single_query_mode": "APPROVE"}}):
            assert _get_single_query_approval_mode() == "approve"

    def test_unknown_value_defaults_to_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"single_query_mode": "maybe"}}):
            assert _get_single_query_approval_mode() == "deny"

    def test_config_load_failure_defaults_to_deny(self):
        """If config loading fails entirely, default to deny (safe)."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("config broken")):
            assert _get_single_query_approval_mode() == "deny"

    def test_yaml_boolean_false_maps_to_deny(self):
        """YAML 1.1 parses bare 'off' as False. Ensure it maps to deny."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config_readonly", return_value={"approvals": {"single_query_mode": False}}):
            # str(False) = "False", which is not in the approve set, so deny
            assert _get_single_query_approval_mode() == "deny"


# ---------------------------------------------------------------------------
# Single-query context detection
# ---------------------------------------------------------------------------

class TestSingleQueryContextDetection:
    def test_env_var_marks_single_query(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        assert approval_module._is_single_query_approval_context() is True

    def test_env_var_unset_is_not_single_query(self, monkeypatch):
        monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
        assert approval_module._is_single_query_approval_context() is False

    def test_env_var_false_is_not_single_query(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "0")
        assert approval_module._is_single_query_approval_context() is False

    def test_blank_session_context_masks_leaked_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        tokens = set_session_vars(cron_session="")
        try:
            # Session context engaged: get_session_env returns "" because the
            # single-query var lives outside _VAR_MAP — but the legacy env
            # fallback route in _is_single_query_approval_context still sees it.
            assert approval_module._is_single_query_approval_context() is True
        finally:
            clear_session_vars(tokens)


# ---------------------------------------------------------------------------
# check_dangerous_command() with a single-query session
# ---------------------------------------------------------------------------

class TestSingleQueryDenyMode:
    """When HERMES_SINGLE_QUERY_SESSION is set and single_query_mode=deny,
    dangerous commands are blocked deterministically instead of waiting a full
    approval timeout for a user who is not there."""

    def test_dangerous_command_blocked_in_single_query_deny_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]
            assert "single_query_mode" in result["message"]

    def test_safe_command_allowed_in_single_query_deny_mode(self, monkeypatch):
        """Non-dangerous commands still work even with single_query_mode=deny."""
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"):
            result = check_dangerous_command("ls -la", "local")
            assert result["approved"]

    def test_block_message_includes_description(self, monkeypatch):
        """The block message should mention what pattern was matched."""
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "dangerous" in result["message"].lower() or "delete" in result["message"].lower()


class TestSingleQueryApproveMode:
    """When HERMES_SINGLE_QUERY_SESSION is set and single_query_mode=approve,
    dangerous commands pass through — no prompt, no timeout wait."""

    def test_dangerous_command_allowed_in_single_query_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="approve"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert result["approved"]


# ---------------------------------------------------------------------------
# check_all_command_guards() with a single-query session
# ---------------------------------------------------------------------------

class TestSingleQueryDenyModeAllGuards:
    """The combined guard function also respects single_query_mode."""

    def test_dangerous_command_blocked_in_combined_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]
            assert "single_query_mode" in result["message"]

    def test_safe_command_allowed_in_combined_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"):
            result = check_all_command_guards("echo hello", "local")
            assert result["approved"]

    def test_combined_guard_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="approve"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert result["approved"]

    def test_tirith_content_threat_blocked_in_single_query_deny(self, monkeypatch):
        """Content-level threats caught only by tirith (not the regex patterns)
        are blocked in single-query-deny mode — the same regression #22070 fixed
        for cron must not resurface for -q."""
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        fake_tirith = {
            "action": "block",
            "findings": [{"severity": "HIGH", "title": "Homograph URL",
                          "description": "URL contains Cyrillic lookalike chars"}],
            "summary": "homograph url",
        }
        with (
            mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"),
            mock_patch("tools.approval.detect_dangerous_command",
                       return_value=(False, None, None)),
            mock_patch("tools.tirith_security.check_command_security",
                       return_value=fake_tirith),
        ):
            result = check_all_command_guards("curl http://xn--e1afmkfd.example/x", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]


# ---------------------------------------------------------------------------
# check_execute_code_guard(): the -q escape hatch is closed
# ---------------------------------------------------------------------------

class TestSingleQueryExecuteCode:
    """execute_code auto-approves in plain non-gateway mode. A -q run must not
    silently auto-approve arbitrary code — it goes through single_query_mode."""

    def test_execute_code_blocked_in_single_query_deny(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"):
            result = approval_module.check_execute_code_guard("import os", "local")
            assert not result["approved"]
            assert result["outcome"] == "blocked"
            assert "single_query_mode" in result["message"]

    def test_execute_code_allowed_in_single_query_approve(self, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="approve"):
            result = approval_module.check_execute_code_guard("import os", "local")
            assert result["approved"]

    def test_headless_execute_code_still_auto_approves_outside_single_query(self, monkeypatch):
        """Without the single-query marker, headless execute_code keeps its
        documented auto-approve contract (no behavior change outside -q)."""
        monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        result = approval_module.check_execute_code_guard("import os", "local")
        assert result["approved"]


# ---------------------------------------------------------------------------
# Edge cases: single-query mode interaction with other mechanisms
# ---------------------------------------------------------------------------

class TestSingleQueryModeInteractions:
    """Single-query mode should NOT interfere with other approval mechanisms."""

    def test_container_env_still_auto_approves(self, monkeypatch):
        """Docker/sandbox environments bypass approvals regardless of mode."""
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /", "docker")
            assert result["approved"]

    def test_yolo_overrides_single_query_deny(self, monkeypatch):
        """--yolo still bypasses single_query_mode=deny for dangerous (non-hardline)
        commands."""
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_YOLO_MODE", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)

        # _YOLO_MODE_FROZEN is frozen at module import time (security: prevents
        # prompt injection from runtime-setting HERMES_YOLO_MODE). Patch the
        # module attribute directly to simulate process-startup with
        # HERMES_YOLO_MODE=1.
        from unittest.mock import patch as mock_patch
        with (
            mock_patch.object(approval_module, "_YOLO_MODE_FROZEN", True),
            mock_patch("tools.approval._get_single_query_approval_mode", return_value="deny"),
        ):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert result["approved"]

    def test_hardline_block_still_fires(self, monkeypatch):
        """Hardline commands are blocked even under single_query_mode=approve."""
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_single_query_approval_mode", return_value="approve"):
            result = check_all_command_guards("rm -rf /", "local")
            assert not result["approved"]