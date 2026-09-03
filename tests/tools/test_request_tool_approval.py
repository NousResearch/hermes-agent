"""Tests for tools.approval.request_tool_approval — the plugin pre_tool_call
``{"action": "approve"}`` escalation into the human-approval gate.

These verify that a plugin-driven approval reuses the SAME machinery as a
Tier-2 dangerous-command match: session/permanent allowlist, the CLI prompt,
the gateway submit_pending path, cron_mode, and fail-closed timeouts.
"""

import pytest

import tools.approval as approval
from tools.approval import (
    build_plugin_tool_approval_rule_key,
    consume_current_native_tool_approval_grant,
    request_tool_approval,
)


def assert_native_grant(grant, **expected):
    assert grant is not None
    assert {key: grant[key] for key in expected} == expected
    assert grant["grant_id"].startswith("native-")
    assert grant["approved_at"] < grant["expires_at"]
    assert grant["expires_at"] - grant["approved_at"] <= 30


@pytest.fixture(autouse=True)
def _isolate_approval_state(monkeypatch):
    """Give each test a clean session key and empty allowlists."""
    approval.clear_current_native_tool_approval_grant()
    monkeypatch.setattr(
        approval,
        "get_current_session_key",
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
    approval.clear_current_native_tool_approval_grant()


class TestRequestToolApproval:
    def test_session_cached_approval_short_circuits(self, monkeypatch):
        monkeypatch.setattr(approval, "is_approved", lambda sk, pk: True)
        # Should NOT prompt at all.
        monkeypatch.setattr(
            approval,
            "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("should not prompt when already approved"),
        )
        res = request_tool_approval("write_file", "sensitive path", rule_key="ssh")
        assert res == {"approved": True, "message": None}

    def test_cli_approve_once(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "once"
        )
        res = request_tool_approval("write_file", "writing ~/.ssh/authorized_keys")
        assert res["approved"] is True
        assert res["approval_choice"] == "once"

    def test_cli_deny_blocks(self, monkeypatch):
        from hermes_cli import lifecycle

        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "deny"
        )
        events = []
        monkeypatch.setattr(
            lifecycle,
            "invoke_hook",
            lambda hook_name, **kwargs: events.append((hook_name, kwargs)) or [],
        )
        tokens = approval.set_current_observability_context(
            turn_id="turn-1",
            tool_call_id="call-1",
        )
        try:
            res = request_tool_approval("terminal", "curl PUT to external API")
        finally:
            approval.reset_current_observability_context(tokens)
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
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "session"
        )
        calls = {"session": [], "permanent": []}
        monkeypatch.setattr(
            approval, "approve_session", lambda sk, pk: calls["session"].append(pk)
        )
        monkeypatch.setattr(
            approval, "approve_permanent", lambda pk: calls["permanent"].append(pk)
        )
        monkeypatch.setattr(approval, "save_permanent_allowlist", lambda x: None)
        res = request_tool_approval("write_file", "reason", rule_key="ssh-writes")
        assert res["approved"] is True
        assert "native_approval_grant" not in res
        assert calls["session"] == ["plugin_rule:ssh-writes"]
        assert calls["permanent"] == []  # session != always

    def test_cron_deny_mode_blocks(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "deny")
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is False
        assert "cron" in res["message"].lower()

    def test_cron_approve_mode_allows(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: True)
        monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "approve")
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is True

    def test_distinct_reasons_get_distinct_keys(self, monkeypatch):
        """Two different reasons on the SAME tool must not share an [a]lways
        allowlist entry (Finding 3: tool_name alone was too coarse)."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "deny"
        )
        k1 = request_tool_approval("write_file", "write to ~/.ssh")["pattern_key"]
        k2 = request_tool_approval("write_file", "send an email")["pattern_key"]
        assert k1 != k2

    def test_explicit_rule_key_overrides_derivation(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "deny"
        )
        res = request_tool_approval("terminal", "any", rule_key="my-rule")
        assert res["pattern_key"] == "plugin_rule:my-rule"

    def test_no_human_non_cron_fails_closed(self, monkeypatch):
        """Non-interactive, non-gateway, NON-cron context blocks (fail-closed)
        — a plugin-flagged action never runs ungated without a human."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "_is_cron_approval_context", lambda: False)
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is False
        assert "no interactive user or gateway" in res["message"].lower()

    def test_yolo_session_bypasses_gate(self, monkeypatch):
        """A --yolo session skips the plugin approval gate (parity with the
        dangerous-command path, via the shared _run_approval_gate)."""
        monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: True)
        monkeypatch.setattr(
            approval,
            "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("yolo must not prompt"),
        )
        res = request_tool_approval("terminal", "curl PUT", rule_key="ext")
        assert res == {"approved": True, "message": None}

    def test_once_approval_issues_matching_one_use_native_grant(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "once"
        )
        tokens = approval.set_current_observability_context(
            turn_id="turn-1",
            tool_call_id="call-1",
            session_id="session-1",
        )
        try:
            res = request_tool_approval(
                "write_file",
                "writing ~/.ssh/authorized_keys",
                rule_key="ssh-writes",
            )
            assert res["approved"] is True
            assert_native_grant(
                res["native_approval_grant"],
                tool_name="write_file",
                session_id="session-1",
                tool_call_id="call-1",
                turn_id="turn-1",
                rule_key="ssh-writes",
            )
            assert (
                consume_current_native_tool_approval_grant(
                    tool_name="write_file",
                    rule_key="ssh-writes",
                )
                == res["native_approval_grant"]
            )
            assert (
                consume_current_native_tool_approval_grant(
                    tool_name="write_file",
                    rule_key="ssh-writes",
                )
                is None
            )
        finally:
            approval.reset_current_observability_context(tokens)

    @pytest.mark.parametrize("choice", ["deny", "timeout", "session", "always"])
    def test_non_once_outcomes_leave_no_native_grant(self, monkeypatch, choice):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: choice
        )
        tokens = approval.set_current_observability_context(
            turn_id="turn-2",
            tool_call_id="call-2",
            session_id="session-2",
        )
        try:
            approval._issue_current_native_tool_approval_grant(
                tool_name="stale_tool",
                rule_key="stale-rule",
            )
            res = request_tool_approval(
                "write_file",
                "writing ~/.ssh/authorized_keys",
                rule_key="ssh-writes",
            )
            if choice in {"session", "always"}:
                assert res["approved"] is True
            else:
                assert res["approved"] is False
            assert (
                consume_current_native_tool_approval_grant(
                    tool_name="write_file",
                    rule_key="ssh-writes",
                )
                is None
            )
        finally:
            approval.reset_current_observability_context(tokens)

    def test_mismatched_consumer_cannot_spend_grant(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "once"
        )
        tokens = approval.set_current_observability_context(
            turn_id="turn-3",
            tool_call_id="call-3",
            session_id="session-3",
        )
        try:
            request_tool_approval(
                "write_file",
                "writing ~/.ssh/authorized_keys",
                rule_key="ssh-writes",
            )
            assert (
                consume_current_native_tool_approval_grant(
                    tool_name="write_file",
                    rule_key="wrong-rule",
                )
                is None
            )
            assert (
                consume_current_native_tool_approval_grant(
                    tool_name="terminal",
                    rule_key="ssh-writes",
                )
                is None
            )
            grant = consume_current_native_tool_approval_grant(
                tool_name="write_file",
                rule_key="ssh-writes",
            )
            assert_native_grant(
                grant,
                tool_name="write_file",
                session_id="session-3",
                tool_call_id="call-3",
                turn_id="turn-3",
                rule_key="ssh-writes",
            )
        finally:
            approval.reset_current_observability_context(tokens)

    def test_grant_is_context_isolated_from_other_threads(self, monkeypatch):
        import threading

        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "once"
        )
        tokens = approval.set_current_observability_context(
            turn_id="turn-4",
            tool_call_id="call-4",
            session_id="session-4",
        )
        foreign = {}
        try:
            request_tool_approval(
                "write_file",
                "writing ~/.ssh/authorized_keys",
                rule_key="ssh-writes",
            )

            def worker():
                foreign["grant"] = consume_current_native_tool_approval_grant(
                    tool_name="write_file",
                    rule_key="ssh-writes",
                    session_id="session-4",
                    tool_call_id="call-4",
                    turn_id="turn-4",
                )

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=2)
            assert foreign["grant"] is None
            grant = consume_current_native_tool_approval_grant(
                tool_name="write_file",
                rule_key="ssh-writes",
            )
            assert_native_grant(
                grant,
                tool_name="write_file",
                session_id="session-4",
                tool_call_id="call-4",
                turn_id="turn-4",
                rule_key="ssh-writes",
            )
        finally:
            approval.reset_current_observability_context(tokens)

    def test_once_only_plugin_approval_hides_session_and_permanent_scopes(
        self, monkeypatch
    ):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        captured = {}

        def fake_prompt(*_args, **kwargs):
            captured.update(kwargs)
            return "once"

        monkeypatch.setattr(approval, "prompt_dangerous_approval", fake_prompt)
        res = request_tool_approval(
            "write_file",
            "protected write",
            rule_key="one-shot",
            allow_session=False,
            allow_permanent=False,
        )
        assert res["approved"] is True
        assert captured["allow_session"] is False
        assert captured["allow_permanent"] is False
        assert res["approval_choice"] == "once"

    @pytest.mark.parametrize("choice", ["session", "always"])
    def test_once_only_plugin_approval_rejects_broader_scopes(
        self, monkeypatch, choice
    ):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: choice
        )
        res = request_tool_approval(
            "write_file",
            "protected write",
            rule_key="one-shot",
            allow_session=False,
            allow_permanent=False,
        )
        assert res["approved"] is False
        assert res["outcome"] == "invalid_scope"
        assert "one-time approval" in res["message"]

    def test_derived_rule_key_matches_consumer_expectation(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval", lambda *a, **k: "once"
        )
        rule_key = build_plugin_tool_approval_rule_key(
            "write_file",
            "derived reason",
        )
        tokens = approval.set_current_observability_context(
            turn_id="turn-5",
            tool_call_id="call-5",
            session_id="session-5",
        )
        try:
            request_tool_approval("write_file", "derived reason")
            grant = consume_current_native_tool_approval_grant(
                tool_name="write_file",
                rule_key=rule_key,
            )
            assert_native_grant(
                grant,
                tool_name="write_file",
                session_id="session-5",
                tool_call_id="call-5",
                turn_id="turn-5",
                rule_key=rule_key,
            )
        finally:
            approval.reset_current_observability_context(tokens)
