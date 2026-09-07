"""Tests for tools.approval_exact_action — the non-bypassable, single-use,
final-args-bound approval used by the ``require_exact_action`` pre_tool_call
directive.

Mirrors the style of tests/tools/test_request_tool_approval.py, which covers the
same surfaces (CLI/gateway presence, yolo, cron/unattended, fail-closed) for the
generic ``approve`` gate — these tests exist to show the exact-action gate behaves
DIFFERENTLY where it must: no yolo bypass, no allowlist short-circuit, single-use
receipts bound to exact args/session/tool-call.
"""

import time

import pytest

import tools.approval as approval
import tools.approval_prompt as approval_prompt
import tools.approval_context as tools_approval_context
from tools import approval_context
from tools.approval_exact_action import (
    ExactActionApprovalError,
    canonicalize_action,
    consume_exact_action_approval,
    request_exact_action_approval,
    _receipts,
)


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch):
    monkeypatch.setattr(approval, "get_current_session_key", lambda default="default": "test-session")
    monkeypatch.setattr(tools_approval_context, "get_current_session_key", lambda default="default": "test-session")
    monkeypatch.setattr("tools.approval_exact_action.get_current_session_key",
                        lambda default="default": "test-session")
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None, raising=False)
    _receipts.clear()
    yield
    _receipts.clear()


def _make_cli(monkeypatch, choice: str):
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: choice)
    monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *a, **k: choice)


class TestCanonicalizeAction:
    def test_same_tool_and_args_produce_same_digest(self):
        a = canonicalize_action("send_email", {"to": "a@example.com", "body": "hi"})
        b = canonicalize_action("send_email", {"body": "hi", "to": "a@example.com"})  # key order differs
        assert a == b

    def test_different_args_produce_different_digest(self):
        a = canonicalize_action("send_email", {"to": "a@example.com"})
        b = canonicalize_action("send_email", {"to": "b@example.com"})
        assert a != b

    def test_different_tool_same_args_produce_different_digest(self):
        a = canonicalize_action("send_email", {"x": 1})
        b = canonicalize_action("delete_email", {"x": 1})
        assert a != b

    @pytest.mark.parametrize("tool_name,args", [("", {}), (None, {}), ("t", "not-a-mapping"), ("t", None)])
    def test_malformed_input_raises(self, tool_name, args):
        with pytest.raises(ValueError):
            canonicalize_action(tool_name, args)


class TestRequestExactActionApproval:
    def test_missing_tool_call_id_fails_closed(self, monkeypatch):
        _make_cli(monkeypatch, "once")
        res = request_exact_action_approval("send_email", "send an email", {"to": "a@example.com"},
                                            tool_call_id="")
        assert res["approved"] is False
        assert "tool-call id" in res["message"].lower()

    def test_cli_approve_once_mints_consumable_receipt(self, monkeypatch):
        _make_cli(monkeypatch, "once")
        args = {"to": "a@example.com", "body": "hi"}
        res = request_exact_action_approval("send_email", "send an email", args, tool_call_id="call-1")
        assert res["approved"] is True

        tokens = approval_context.set_current_observability_context(tool_call_id="call-1")
        try:
            consume_exact_action_approval("send_email", args)  # must not raise
        finally:
            approval_context.reset_current_observability_context(tokens)

    def test_cli_deny_blocks_and_mints_nothing(self, monkeypatch):
        _make_cli(monkeypatch, "deny")
        args = {"to": "a@example.com"}
        res = request_exact_action_approval("send_email", "send an email", args, tool_call_id="call-2")
        assert res["approved"] is False
        tokens = approval_context.set_current_observability_context(tool_call_id="call-2")
        try:
            with pytest.raises(ExactActionApprovalError):
                consume_exact_action_approval("send_email", args)
        finally:
            approval_context.reset_current_observability_context(tokens)

    def test_yolo_session_does_not_bypass_gate(self, monkeypatch):
        """The core guarantee: unlike request_tool_approval, --yolo/session-yolo
        must NOT skip the human prompt for an exact-action approval."""
        monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: True)
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True, raising=False)
        _make_cli(monkeypatch, "deny")  # if yolo were consulted, this would never be reached
        res = request_exact_action_approval("send_email", "send an email", {"to": "a@example.com"},
                                            tool_call_id="call-yolo")
        assert res["approved"] is False
        assert "denied" in res["message"].lower()

    def test_approvals_mode_off_does_not_bypass_gate(self, monkeypatch):
        monkeypatch.setattr(tools_approval_context, "_get_approval_mode", lambda: "off")
        _make_cli(monkeypatch, "deny")
        res = request_exact_action_approval("send_email", "send an email", {"to": "a@example.com"},
                                            tool_call_id="call-off")
        assert res["approved"] is False
        assert "denied" in res["message"].lower()

    def test_prior_allowlist_entry_does_not_bypass_gate(self, monkeypatch):
        """Even if is_approved() would say yes (e.g. a stale allowlist key from
        somewhere else), the exact-action gate never consults it — it always
        prompts fresh."""
        monkeypatch.setattr(approval, "is_approved", lambda sk, pk: True)
        _make_cli(monkeypatch, "deny")
        res = request_exact_action_approval("send_email", "send an email", {"to": "a@example.com"},
                                            tool_call_id="call-allowlist")
        assert res["approved"] is False

    def test_no_human_fails_closed_unconditionally(self, monkeypatch):
        """Unlike request_tool_approval's cron_mode/single_query_mode/unattended_mode
        'approve' escape hatches, exact-action has none: no human present always blocks."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(tools_approval_context, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(tools_approval_context, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: "approve")
        res = request_exact_action_approval("send_email", "send an email", {"to": "a@example.com"},
                                            tool_call_id="call-cron")
        assert res["approved"] is False
        assert "no interactive user" in res["message"].lower()

    def test_gate_exception_fails_closed(self, monkeypatch):
        _make_cli(monkeypatch, "once")
        monkeypatch.setattr("tools.approval_exact_action._human_decision",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        res = request_exact_action_approval("send_email", "send an email", {"to": "a@example.com"},
                                            tool_call_id="call-boom")
        assert res["approved"] is False
        assert "gate failed" in res["message"].lower()

    def test_secret_looking_value_is_redacted_in_display(self, monkeypatch):
        captured = {}

        def fake_human_decision(spec, *, command, **kwargs):
            captured["command"] = command
            return {"approved": True, "message": None}

        _make_cli(monkeypatch, "once")
        monkeypatch.setattr("tools.approval_exact_action._human_decision", fake_human_decision)
        secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        request_exact_action_approval("send_email", "send an email",
                                      {"to": "a@example.com", "authorization": f"Bearer {secret}"},
                                      tool_call_id="call-secret")
        assert secret not in captured["command"]


class TestConsumeExactActionApproval:
    def _mint(self, monkeypatch, tool_name, args, tool_call_id, ttl_seconds=120):
        _make_cli(monkeypatch, "once")
        res = request_exact_action_approval(tool_name, "do it", args, tool_call_id=tool_call_id,
                                            ttl_seconds=ttl_seconds)
        assert res["approved"] is True

    def test_no_receipt_raises(self, monkeypatch):
        tokens = approval_context.set_current_observability_context(tool_call_id="never-minted")
        try:
            with pytest.raises(ExactActionApprovalError):
                consume_exact_action_approval("send_email", {"to": "a@example.com"})
        finally:
            approval_context.reset_current_observability_context(tokens)

    def test_missing_tool_call_context_raises(self):
        with pytest.raises(ExactActionApprovalError):
            consume_exact_action_approval("send_email", {"to": "a@example.com"})

    def test_changed_args_raise(self, monkeypatch):
        args = {"to": "a@example.com", "body": "hi"}
        self._mint(monkeypatch, "send_email", args, "call-3")
        tokens = approval_context.set_current_observability_context(tool_call_id="call-3")
        try:
            with pytest.raises(ExactActionApprovalError):
                consume_exact_action_approval("send_email", {"to": "a@example.com", "body": "changed"})
        finally:
            approval_context.reset_current_observability_context(tokens)

    def test_changed_tool_name_raises(self, monkeypatch):
        args = {"to": "a@example.com"}
        self._mint(monkeypatch, "send_email", args, "call-4")
        tokens = approval_context.set_current_observability_context(tool_call_id="call-4")
        try:
            with pytest.raises(ExactActionApprovalError):
                consume_exact_action_approval("delete_email", args)
        finally:
            approval_context.reset_current_observability_context(tokens)

    def test_wrong_tool_call_id_raises(self, monkeypatch):
        args = {"to": "a@example.com"}
        self._mint(monkeypatch, "send_email", args, "call-5")
        tokens = approval_context.set_current_observability_context(tool_call_id="call-other")
        try:
            with pytest.raises(ExactActionApprovalError):
                consume_exact_action_approval("send_email", args)
        finally:
            approval_context.reset_current_observability_context(tokens)

    def test_wrong_session_raises(self, monkeypatch):
        args = {"to": "a@example.com"}
        self._mint(monkeypatch, "send_email", args, "call-6")
        monkeypatch.setattr("tools.approval_exact_action.get_current_session_key",
                            lambda default="default": "other-session")
        tokens = approval_context.set_current_observability_context(tool_call_id="call-6")
        try:
            with pytest.raises(ExactActionApprovalError):
                consume_exact_action_approval("send_email", args)
        finally:
            approval_context.reset_current_observability_context(tokens)

    def test_double_consume_raises_on_second_attempt(self, monkeypatch):
        args = {"to": "a@example.com"}
        self._mint(monkeypatch, "send_email", args, "call-7")
        tokens = approval_context.set_current_observability_context(tool_call_id="call-7")
        try:
            consume_exact_action_approval("send_email", args)  # first: succeeds
            with pytest.raises(ExactActionApprovalError):
                consume_exact_action_approval("send_email", args)  # second: fails closed
        finally:
            approval_context.reset_current_observability_context(tokens)

    def test_expired_receipt_raises(self, monkeypatch):
        import tools.approval_exact_action as aea

        args = {"to": "a@example.com"}
        self._mint(monkeypatch, "send_email", args, "call-8", ttl_seconds=1)
        real_time = time.time
        monkeypatch.setattr(aea.time, "time", lambda: real_time() + 5)
        tokens = approval_context.set_current_observability_context(tool_call_id="call-8")
        try:
            with pytest.raises(ExactActionApprovalError):
                consume_exact_action_approval("send_email", args)
        finally:
            approval_context.reset_current_observability_context(tokens)
