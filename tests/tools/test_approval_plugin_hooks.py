"""Tests for pre_approval_request / post_approval_response plugin hooks.

These hooks fire in tools/approval.py::check_all_command_guards whenever a
dangerous command needs user approval. They are observer-only (return values
ignored) and must fire on BOTH the CLI-interactive path and the async gateway
path, so external tools like macOS notifiers can be alerted regardless of
which surface the user is on.
"""
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools.approval import (
    check_all_command_guards,
    check_execute_code_guard,
    set_current_session_key,
    clear_session,
)


@pytest.fixture
def isolated_session(monkeypatch, tmp_path):
    """Give each test a fresh session_key, clean approval-state, and isolated
    HERMES_HOME so the real user's command_allowlist doesn't leak in."""
    import tools.approval as _am

    session_key = "test:session:approval_hooks"
    token = set_current_session_key(session_key)
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    # Make sure we don't skip guards via yolo / approvals.mode=off
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    # Isolate from the real user's permanent allowlist + session state
    _saved_permanent = _am._permanent_approved.copy()
    _saved_session = {k: v.copy() for k, v in _am._session_approved.items()}
    _am._permanent_approved.clear()
    _am._session_approved.clear()
    try:
        yield session_key
    finally:
        _am._permanent_approved.update(_saved_permanent)
        _am._session_approved.update(_saved_session)
        try:
            _am._approval_session_key.reset(token)
        except Exception:
            pass
        clear_session(session_key)


class TestCliPathFiresHooks:
    """CLI-interactive approval path: HERMES_INTERACTIVE is set, the
    prompt_dangerous_approval() result decides the outcome."""

    def test_pre_and_post_fire_with_expected_kwargs(
        self, isolated_session, monkeypatch
    ):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        # approvals.mode=manual so we actually reach the prompt site
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        captured = []

        def fake_invoke_hook(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        # Force the user to "approve once" via the approval_callback contract
        def cb(command, description, *, allow_permanent=True):
            return "once"

        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_all_command_guards(
                "rm -rf /tmp/test-hook", "local", approval_callback=cb,
            )

        assert result["approved"] is True

        hook_names = [c[0] for c in captured]
        assert "pre_approval_request" in hook_names
        assert "post_approval_response" in hook_names

        pre_kwargs = next(kw for name, kw in captured if name == "pre_approval_request")
        assert pre_kwargs["command"] == "rm -rf /tmp/test-hook"
        assert pre_kwargs["surface"] == "cli"
        assert pre_kwargs["session_key"] == isolated_session
        assert isinstance(pre_kwargs["pattern_keys"], list)
        assert pre_kwargs["pattern_key"]  # non-empty primary pattern
        assert pre_kwargs["description"]

        post_kwargs = next(kw for name, kw in captured if name == "post_approval_response")
        assert post_kwargs["choice"] == "once"
        assert post_kwargs["surface"] == "cli"
        assert post_kwargs["command"] == "rm -rf /tmp/test-hook"

    def test_deny_reported_to_post_hook(self, isolated_session, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        captured = []

        def fake_invoke_hook(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        def cb(command, description, *, allow_permanent=True):
            return "deny"

        with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
            result = check_all_command_guards(
                "rm -rf /tmp/test-deny", "local", approval_callback=cb,
            )

        assert result["approved"] is False
        post_kwargs = next(kw for name, kw in captured if name == "post_approval_response")
        assert post_kwargs["choice"] == "deny"

    def test_plugin_hook_crash_does_not_break_approval(
        self, isolated_session, monkeypatch
    ):
        """A crashing plugin must never prevent the approval flow from
        reaching the user. Hooks are observer-only and safety-critical
        behavior must be preserved."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        def boom(hook_name, **kwargs):
            raise RuntimeError("plugin crashed")

        def cb(command, description, *, allow_permanent=True):
            return "once"

        with patch("hermes_cli.plugins.invoke_hook", side_effect=boom):
            result = check_all_command_guards(
                "rm -rf /tmp/test-crash", "local", approval_callback=cb,
            )

        # User's approval was still honored despite the plugin crashing
        assert result["approved"] is True


class TestGatewayPathFiresHooks:
    """Async gateway approval path: HERMES_GATEWAY_SESSION is set and a
    gateway notify callback is registered. The agent thread blocks on the
    approval event until resolve_gateway_approval() is called from another
    thread."""


class TestSmartModeFiresHooks:
    def _configure(self, monkeypatch, verdict):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda *_: verdict)
        monkeypatch.setattr(
            "tools.tirith_security.check_command_security",
            lambda _: {"action": "allow", "findings": [], "summary": ""},
        )

    @pytest.mark.parametrize(
        ("guard", "value", "verdict", "approved", "choice", "pattern_key"),
        [
            (check_all_command_guards, "rm -rf /tmp/smart-hook", "approve", True, "smart_approve", None),
            (check_all_command_guards, "rm -rf /tmp/smart-hook", "deny", False, "smart_deny", None),
            (check_execute_code_guard, "print('smart hook')", "approve", True, "smart_approve", "execute_code"),
            (check_execute_code_guard, "print('smart hook')", "deny", False, "smart_deny", "execute_code"),
        ],
    )
    def test_smart_verdict_fires_redacted_pre_and_post_hooks(
        self, isolated_session, monkeypatch, guard, value, verdict, approved, choice, pattern_key
    ):
        self._configure(monkeypatch, verdict)
        secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        value = f'{value} # Authorization: Bearer {secret}'
        captured = []

        with patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda name, **kwargs: captured.append((name, kwargs)),
        ):
            result = guard(value, "local")

        assert result["approved"] is approved
        assert result[f"smart_{'approved' if approved else 'denied'}"] is True
        assert [name for name, _ in captured] == [
            "pre_approval_request",
            "post_approval_response",
        ]
        pre, post = (kwargs for _, kwargs in captured)
        assert pre["surface"] == post["surface"] == "smart"
        assert post["choice"] == choice
        assert post["decided_by"] == "aux_llm"
        assert pre["session_key"] == post["session_key"] == isolated_session
        assert secret not in pre["command"]
        assert secret not in post["command"]
        assert pre["pattern_keys"]
        assert pre["pattern_key"] == post["pattern_key"]
        if pattern_key is not None:
            assert pre["pattern_key"] == pattern_key
            assert pre["pattern_keys"] == [pattern_key]

    @pytest.mark.parametrize("guard,value", [
        (check_all_command_guards, "rm -rf /tmp/smart-order"),
        (check_execute_code_guard, "print('smart order')"),
    ])
    def test_pre_hook_fires_before_aux_llm_decision(
        self, isolated_session, monkeypatch, guard, value
    ):
        self._configure(monkeypatch, "approve")
        events = []

        def decide(*_):
            events.append("smart_approve")
            return "approve"

        monkeypatch.setattr(approval_module, "_smart_approve", decide)
        with patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda name, **kwargs: events.append(name),
        ):
            result = guard(value, "local")

        assert result["approved"] is True
        assert events == [
            "pre_approval_request",
            "smart_approve",
            "post_approval_response",
        ]

    @pytest.mark.parametrize("guard,value", [
        (check_all_command_guards, "rm -rf /tmp/smart-force-redaction"),
        (check_execute_code_guard, "print('smart force redaction')"),
    ])
    def test_smart_observer_redaction_is_forced_when_config_disables_redaction(
        self, isolated_session, monkeypatch, guard, value
    ):
        self._configure(monkeypatch, "approve")
        force_values = []

        def redact(text, *, force=False):
            force_values.append(force)
            return f"redacted:{text}"

        with (
            patch("agent.redact.redact_sensitive_text", side_effect=redact),
            patch("hermes_cli.plugins.invoke_hook"),
        ):
            result = guard(value, "local")

        assert result["approved"] is True
        assert force_values == [True, True]

    @pytest.mark.parametrize("guard,value", [
        (check_all_command_guards, "rm -rf /tmp/smart-hook-crash"),
        (check_execute_code_guard, "print('smart hook crash')"),
    ])
    @pytest.mark.parametrize("verdict,approved", [("approve", True), ("deny", False)])
    def test_observer_exception_never_changes_smart_verdict(
        self, isolated_session, monkeypatch, guard, value, verdict, approved
    ):
        self._configure(monkeypatch, verdict)
        with patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=RuntimeError("observer failed"),
        ):
            result = guard(value, "local")
        assert result["approved"] is approved

    @pytest.mark.parametrize("guard,value", [
        (check_all_command_guards, "rm -rf /tmp/smart-redactor-crash"),
        (check_execute_code_guard, "print('smart redactor crash')"),
    ])
    @pytest.mark.parametrize("verdict,approved", [("approve", True), ("deny", False)])
    def test_redactor_exception_never_changes_smart_verdict_or_leaks_payload(
        self, isolated_session, monkeypatch, guard, value, verdict, approved
    ):
        self._configure(monkeypatch, verdict)
        captured = []

        def fail_observer_redaction(text, *, force=False):
            if force:
                raise RuntimeError("observer redactor failed")
            return text

        with (
            patch("agent.redact.redact_sensitive_text", side_effect=fail_observer_redaction),
            patch(
                "hermes_cli.plugins.invoke_hook",
                side_effect=lambda name, **kwargs: captured.append((name, kwargs)),
            ),
        ):
            result = guard(value, "local")
        assert result["approved"] is approved
        assert captured == []

    @pytest.mark.parametrize("guard,first_value,second_value", [
        (
            check_all_command_guards,
            "rm -rf /tmp/first-smart-command",
            "rm -rf /tmp/second-smart-command",
        ),
        (
            check_execute_code_guard,
            "print('first smart script')",
            "print('second smart script')",
        ),
    ])
    def test_smart_approval_is_per_command(
        self, isolated_session, monkeypatch, guard, first_value, second_value
    ):
        verdicts = iter(("approve", "deny"))
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval_module, "_smart_approve", lambda *_: next(verdicts))
        monkeypatch.setattr(
            "tools.tirith_security.check_command_security",
            lambda _: {"action": "allow", "findings": [], "summary": ""},
        )
        captured = []
        with patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda name, **kwargs: captured.append((name, kwargs)),
        ):
            first = guard(first_value, "local")
            second = guard(second_value, "local")

        assert first["approved"] is True
        assert second["approved"] is False
        assert [kwargs["choice"] for name, kwargs in captured if name == "post_approval_response"] == [
            "smart_approve",
            "smart_deny",
        ]


