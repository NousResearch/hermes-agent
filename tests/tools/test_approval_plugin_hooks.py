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
    _GATEWAY_SELF_TERMINATION_DESCRIPTIONS,
    check_all_command_guards,
    clear_session,
    register_gateway_notify,
    resolve_gateway_approval,
    set_current_session_key,
    unregister_gateway_notify,
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

class TestGatewaySelfTerminationBlock:
    """Gateway self-termination hard-block (issue #18693).

    Commands that kill the gateway process itself (pkill hermes,
    killall hermes, hermes gateway stop/restart, hermes update, etc.)
    are hard-blocked inside gateway sessions to prevent an infinite
    approval -> crash -> restart -> approval loop.
    """

    def test_gateway_blocks_pkill_hermes(self, isolated_session, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        result = check_all_command_guards("pkill -f hermes", "local")

        assert result["approved"] is False
        assert "Self-termination blocked" in result["message"]

    def test_gateway_blocks_hermes_gateway_stop(self, isolated_session, monkeypatch):
        """hermes gateway stop restarts the gateway, killing running agents."""
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        result = check_all_command_guards(
            "hermes gateway stop --replace", "local"
        )

        assert result["approved"] is False
        assert "Self-termination blocked" in result["message"]

    def test_cli_does_not_block_pkill(self, isolated_session, monkeypatch):
        """CLI path: self-termination goes through normal approval, not hard-block."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        def cb(command, description, *, allow_permanent=True):
            return "once"

        result = check_all_command_guards(
            "pkill -f hermes", "local", approval_callback=cb,
        )

        assert result["approved"] is True

    def test_non_interactive_passes_through(self, isolated_session, monkeypatch):
        """Non-CLI, non-gateway -> guard never reached, approved by fallthrough."""
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

        result = check_all_command_guards("pkill -f hermes", "local")

        assert result["approved"] is True

    def test_all_descriptions_in_dangerous_patterns(self):
        """Invariant: every self-termination description exists in DANGEROUS_PATTERNS."""
        dangerous_descs = {desc for _, desc in approval_module.DANGEROUS_PATTERNS}
        for desc in _GATEWAY_SELF_TERMINATION_DESCRIPTIONS:
            assert desc in dangerous_descs, (
                f"{desc!r} not in DANGEROUS_PATTERNS — fix the frozenset or the pattern list"
            )
