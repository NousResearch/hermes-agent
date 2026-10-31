"""Tests for tools.approval.request_tool_approval — the plugin pre_tool_call
``{"action": "approve"}`` escalation into the human-approval gate.

These verify that a plugin-driven approval reuses the SAME machinery as a
Tier-2 dangerous-command match: session/permanent allowlist, the CLI prompt,
the gateway submit_pending path, cron_mode, and fail-closed timeouts.
"""

import pytest

from tools import approval
from tools import approval_prompt
import tools.approval_context as tools_approval_context
from tools import approval_context
from tools.approval import request_tool_approval


@pytest.fixture(autouse=True)
def _isolate_approval_state(monkeypatch):
    """Give each test a clean session key and empty allowlists."""
    monkeypatch.setattr(
        approval, "get_current_session_key",
        lambda default="default": "test-session",
    )
    monkeypatch.setattr(
        tools_approval_context, "get_current_session_key",
        lambda default="default": "test-session",
    )
    # Empty session + permanent approval stores so nothing pre-approves.
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    # Not a yolo session (the shared gate checks this first).
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
    # No thread-registered CLI callback by default.
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback", lambda: None, raising=False
    )
    yield


class TestRequestToolApproval:
    def test_session_cached_approval_short_circuits(self, monkeypatch):
        monkeypatch.setattr(approval, "is_approved", lambda sk, pk: True)
        # Should NOT prompt at all.
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("should not prompt when already approved"),
        )
        monkeypatch.setattr(
            approval_prompt, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("should not prompt when already approved"),
        )
        res = request_tool_approval("write_file", "sensitive path", rule_key="ssh")
        assert res == {"approved": True, "message": None}

    def test_cli_approve_once(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "once")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: "once")
        res = request_tool_approval("write_file", "writing ~/.ssh/authorized_keys")
        assert res["approved"] is True

    def test_cli_deny_blocks(self, monkeypatch):
        from hermes_cli import lifecycle

        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: "deny")
        events = []
        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda hook_name, **kwargs: events.append((hook_name, kwargs)) or [],
        )
        tokens = approval_context.set_current_observability_context(
            turn_id="turn-1",
            tool_call_id="call-1",
        )
        try:
            res = request_tool_approval("terminal", "curl PUT to external API")
        finally:
            approval_context.reset_current_observability_context(tokens)
        assert res["approved"] is False
        assert "denied" in res["message"].lower()
        assert res["pattern_key"].startswith("plugin_rule:")
        assert [name for name, _ in events] == [
            "pre_approval_request",
            "post_approval_response",
        ]
        assert all(event["turn_id"] == "turn-1" for _, event in events)
        assert all(event["tool_call_id"] == "call-1" for _, event in events)
        assert events[-1][1]["choice"] == "deny"

    def test_cli_session_persists_session_only(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "session")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: "session")
        calls = {"session": [], "permanent": []}
        monkeypatch.setattr(approval, "approve_session",
                            lambda sk, pk: calls["session"].append(pk))
        monkeypatch.setattr(approval, "approve_permanent",
                            lambda pk: calls["permanent"].append(pk))
        monkeypatch.setattr(approval, "save_permanent_allowlist", lambda x: None)
        res = request_tool_approval("write_file", "reason", rule_key="ssh-writes")
        assert res["approved"] is True
        assert calls["session"] == ["plugin_rule:ssh-writes"]
        assert calls["permanent"] == []  # session != always


    def test_cron_deny_mode_blocks(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(tools_approval_context, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny")
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is False
        assert "cron" in res["message"].lower()

    def test_cron_approve_mode_allows(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(tools_approval_context, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is True


    def test_distinct_reasons_get_distinct_keys(self, monkeypatch):
        """Two different reasons on the SAME tool must not share an [a]lways
        allowlist entry (Finding 3: tool_name alone was too coarse)."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: "deny")
        k1 = request_tool_approval("write_file", "write to ~/.ssh")["pattern_key"]
        k2 = request_tool_approval("write_file", "send an email")["pattern_key"]
        assert k1 != k2

    def test_explicit_rule_key_overrides_derivation(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: "deny")
        res = request_tool_approval("terminal", "any", rule_key="my-rule")
        assert res["pattern_key"] == "plugin_rule:my-rule"

    def test_no_human_non_cron_fails_closed(self, monkeypatch):
        """Non-interactive, non-gateway, NON-cron context blocks (fail-closed)
        — a plugin-flagged action never runs ungated without a human."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_cron_approval_context", lambda: False)
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is False
        assert "no interactive user or gateway" in res["message"].lower()

    def test_api_server_exec_ask_uses_approval_bridge(self, monkeypatch):
        """Plugin escalation reaches the /v1/runs approval bridge when ask mode is active."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_cron_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: False)
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")

        notified = []

        def approve(data):
            notified.append(data)
            assert approval.resolve_gateway_approval(
                "test-session", "once", request_id=data["request_id"]
            ) == 1

        approval.register_gateway_notify("test-session", approve)
        try:
            res = request_tool_approval("home_lock", "unlock the front door", rule_key="unlock")
        finally:
            approval.unregister_gateway_notify("test-session")

        assert res["approved"] is True
        assert len(notified) == 1
        assert notified[0]["pattern_key"] == "plugin_rule:unlock"

    def test_api_server_without_exec_ask_remains_fail_closed(self, monkeypatch):
        """An api_server call without an active approval bridge must not run ungated."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_cron_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_single_query_approval_context", lambda: False)
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        monkeypatch.setattr(approval_context, "_get_unattended_approval_mode", lambda: "deny")
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")

        res = request_tool_approval("home_lock", "unlock the front door", rule_key="unlock")

        assert res["approved"] is False
        assert "unattended platform" in res["message"].lower()

    def test_yolo_session_bypasses_gate(self, monkeypatch):
        """A --yolo session skips the plugin approval gate (parity with the
        dangerous-command path, via the shared _run_approval_gate)."""
        monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: True)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("yolo must not prompt"),
        )
        monkeypatch.setattr(
            approval_prompt, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("yolo must not prompt"),
        )
        res = request_tool_approval("terminal", "curl PUT", rule_key="ext")
        assert res == {"approved": True, "message": None}

    def test_approvals_mode_off_bypasses_gate(self, monkeypatch):
        """``approvals.mode: off`` (the Desktop "Approvals: off" toggle) must bypass the shared gate
        exactly like the shell guards do — otherwise computer_use / plugin-rule / SSH-config-write
        approvals keep prompting a user who turned approvals off."""
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off")
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("approvals.mode=off must not prompt"),
        )
        monkeypatch.setattr(
            approval_prompt, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("approvals.mode=off must not prompt"),
        )
        res = request_tool_approval("computer_use", "click", rule_key="cua")
        assert res == {"approved": True, "message": None}


class TestDenialIsNotABan:
    """A denial is one call the user refused, not consent gone for the session.

    ``request_tool_approval`` documents the plugin path as the SAME human gate as a
    Tier-2 dangerous command, but the action gate's denial never carried the closing
    sentence the command gate has ("stop the current workflow and wait for the user
    to respond"), so the text told the model the user could not decide again and a
    later explicit request was answered with a refusal and no new prompt.
    """

    def test_cli_denial_tells_the_model_the_user_can_still_decide(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: "deny")
        res = request_tool_approval(
            "example_write_tool", "creates an order", rule_key="example_write_tool"
        )
        assert res["approved"] is False
        assert "wait for the user to respond" in res["message"]

    def test_gateway_denial_closes_like_a_command_denial(self):
        action = approval._ACTION_GATE.gateway_refused.format(
            reason="denied by user", reason_addendum="", timeout_addendum="", breaker=""
        )
        command = approval._COMMAND_GATE.gateway_refused.format(
            reason="denied by user", reason_addendum="", timeout_addendum="", breaker=""
        )
        closing = "Stop the current workflow and wait for the user to respond"
        assert action.startswith("BLOCKED: Action denied by user.")
        assert closing in command
        assert closing in action

    def test_every_human_denial_surface_carries_the_closing_sentence(self):
        """The execute_code gate denies the same way a command does.

        Its ``cli_denied`` read "Do NOT retry — the user has explicitly
        rejected it." and stopped, so a single refusal of one script read as
        consent gone: the model would not call ``execute_code`` again even
        when the user asked for it. Its ``gateway_refused`` and both
        ``transport_denied`` templates carried the same omission.
        """
        closing = "Stop the current workflow and wait for the user to respond"
        rendered = {
            "command/cli_denied": approval._COMMAND_GATE.cli_denied.format(
                description="rm -rf build", breaker=""),
            "command/transport_denied": approval._COMMAND_GATE.transport_denied.format(breaker=""),
            "execute_code/cli_denied": approval._EXECUTE_CODE_GATE.cli_denied.format(
                description="script", breaker=""),
            "execute_code/transport_denied": approval._EXECUTE_CODE_GATE.transport_denied.format(breaker=""),
            "execute_code/gateway_refused": approval._EXECUTE_CODE_GATE.gateway_refused.format(
                reason="denied by user", reason_addendum="", timeout_addendum="", breaker=""),
            "action/cli_denied": approval._ACTION_GATE.cli_denied.format(
                description="write", breaker=""),
            "action/gateway_refused": approval._ACTION_GATE.gateway_refused.format(
                reason="denied by user", reason_addendum="", timeout_addendum="", breaker=""),
        }
        assert [name for name, text in rendered.items() if closing not in text] == []
        assert [name for name, text in rendered.items() if text.count(closing) != 1] == []

    def test_execute_code_cli_denial_keeps_its_own_wording(self):
        """One sentence, one place: the append does not restate the refusal twice."""
        rendered = approval._EXECUTE_CODE_GATE.cli_denied.format(description="script", breaker="")
        assert rendered.startswith("BLOCKED: User denied execute_code script execution")
        assert rendered.count("Do NOT retry") == 1
        assert rendered.count("The user has NOT consented") == 0
