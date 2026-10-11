"""Acceptance tests for issue #132507 — the plugin pre_tool_call escalation's
unattended deny message must not teach the agent to disarm the approval gate.

When a plugin's ``pre_tool_call`` hook escalates with ``{"action": "approve"}``
for an action the plugin reserved for a human decision, and nobody is available
to approve (single_query / cron / unattended platform), the gate fails closed
with a block message. The generic message reads "Find an alternative approach
…" plus "To allow … set approvals.<mode>: approve in config.yaml". In an
unattended run the agent treats advice as executable instruction: it would edit
its own config to auto-approve a decision the plugin flagged as needing a human
(a behavior vector), or route around a block whose semantics are "this decision
needs a human", not "take a different route".

Contract under test (design doc 契约规约, predicates P1-P4):

- P1/P2/P3: the plugin-path unattended deny message (single_query / cron /
  unattended platform ctx) carries the plugin ``rule_key``, contains NO
  ``approvals.<mode>: approve`` self-service switch guidance, contains NO
  "Find an alternative" reroute guidance, and stays fail-closed
  (``approved is False``).
- P4: the dangerous-command path keeps its existing cron deny wording — the
  config.yaml switch guidance STAYS there (those switches are the user's
  budgeted approval posture); the fix must touch only the plugin route.

These drive the real gate through the public entry points (``request_tool_approval``
for the plugin escalation, ``check_dangerous_command`` for the dangerous path);
no source-text inspection.
"""

import pytest

import tools.approval as approval
import tools.approval_prompt as approval_prompt
import tools.approval_context as tools_approval_context
from tools import approval_context
from gateway.session_context import reset_session_vars
from tools.approval import check_dangerous_command, request_tool_approval


@pytest.fixture(autouse=True)
def _isolate_approval_state(monkeypatch):
    """Clean session key, empty allowlists, no yolo, no CLI callback — the
    autouse fixture convention from tests/tools/test_request_tool_approval.py,
    plus the session-vars reset the approval-mode suites use."""
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
    reset_session_vars()
    yield
    reset_session_vars()


def _refuse_to_prompt(monkeypatch):
    """An unattended deny decision must be reached without prompting anyone;
    a prompt attempt fails the test instead of stalling on a timeout."""
    monkeypatch.setattr(
        approval, "prompt_dangerous_approval",
        lambda *a, **k: pytest.fail("unattended deny must not prompt"),
    )
    monkeypatch.setattr(
        approval_prompt, "prompt_dangerous_approval",
        lambda *a, **k: pytest.fail("unattended deny must not prompt"),
    )


def _assert_plugin_deny_contract(res, rule_key, ctx_guidance_key):
    """The shared plugin-path unattended deny contract.

    rule_key         — the plugin rule whose human-approval flag caused the
                       escalation; the message must carry it (plugin context,
                       not a generic wall of text).
    ctx_guidance_key — the verbatim self-service switch this unattended ctx
                       used to advertise before the fix (P1/P3:
                       approvals.single_query_mode, P2: approvals.cron_mode).
                       The contract bans ANY approvals.<mode>: approve switch
                       guidance, so the sibling keys are asserted too.
    """
    assert res["approved"] is False, "fail-closed semantics must not change"
    message = res["message"]
    assert message, "unattended deny must produce a block message"
    assert rule_key in message, (
        "deny message must carry the plugin rule_key context, got: %r" % message
    )
    assert ctx_guidance_key not in message, (
        "deny message must not advertise approvals.<mode>: approve as a way to "
        "run this action, got: %r" % message
    )
    assert "approvals.single_query_mode" not in message
    assert "approvals.cron_mode" not in message
    assert "config.yaml" not in message, (
        "deny message must not point the agent at its own config to unblock, "
        "got: %r" % message
    )
    assert "edit the plugin" not in message and "pre_tool_call rules" not in message, (
        "deny message must not teach the agent any policy-editing route (it would "
        "execute the advice and un-escalate itself), got: %r" % message
    )
    assert "ask the user" in message, (
        "issue #132507 contract: the message must direct the agent to a human "
        "instead of any self-service path, got: %r" % message
    )
    assert "Find an alternative" not in message, (
        "escalation means this decision needs a human, not a different route, "
        "got: %r" % message
    )


class TestPluginEscalationUnattendedDenyMessage:
    """Plugin escalation (request_tool_approval) into each unattended ctx."""

    def test_single_query_deny_message_has_no_self_approve_guidance(self, monkeypatch):
        # P1: plugin escalate + approvals.single_query_mode=deny.
        # Env shape mirrors the proven -q recipe (HERMES_INTERACTIVE=1 is what
        # cli.py exports in single-query runs).
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_cron_approval_context", lambda: False)
        monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: "deny")
        _refuse_to_prompt(monkeypatch)

        res = request_tool_approval(
            "email_send", "send the weekly report", rule_key="bulk-email"
        )

        _assert_plugin_deny_contract(res, "bulk-email", "approvals.single_query_mode")

    def test_cron_deny_message_has_no_self_approve_guidance(self, monkeypatch):
        # P2: plugin escalate + approvals.cron_mode=deny. Patch shape mirrors
        # tests/tools/test_request_tool_approval.py::test_cron_deny_mode_blocks.
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(tools_approval_context, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "deny")
        _refuse_to_prompt(monkeypatch)

        res = request_tool_approval("terminal", "smtp send", rule_key="smtp-relay")

        _assert_plugin_deny_contract(res, "smtp-relay", "approvals.cron_mode")

    def test_unattended_platform_deny_message_has_no_self_approve_guidance(self, monkeypatch):
        # P3: plugin escalate + unattended platform ctx deny. Patch shape
        # mirrors test_api_server_without_exec_ask_remains_fail_closed.
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
        _refuse_to_prompt(monkeypatch)

        res = request_tool_approval(
            "home_lock", "unlock the front door", rule_key="front-door"
        )

        _assert_plugin_deny_contract(res, "front-door", "approvals.single_query_mode")


class TestDangerousCommandPathWordingUnchanged:
    """P4 — regression guard: the dangerous-command path keeps its existing
    cron deny wording, config.yaml switch guidance included. Proves the fix
    touches ONLY the plugin escalation route."""

    def test_cron_deny_keeps_existing_switch_guidance(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval_context._get_cron_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")

        assert result["approved"] is False
        assert "cron_mode" in result["message"], (
            "dangerous-path cron deny wording must keep the cron_mode switch, "
            "got: %r" % result["message"]
        )
        assert "config.yaml" in result["message"], (
            "dangerous-path cron deny must keep its config.yaml switch "
            "guidance (wording unchanged by this fix), got: %r" % result["message"]
        )
