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
    register_gateway_notify,
    unregister_gateway_notify,
    resolve_gateway_approval,
    set_current_session_key,
    clear_session,
)


@pytest.fixture
def isolated_session(monkeypatch):
    """Give each test a fresh session_key and clean approval-state."""
    session_key = "test:session:approval_hooks"
    token = set_current_session_key(session_key)
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    # Make sure we don't skip guards via yolo / approvals.mode=off
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    try:
        yield session_key
    finally:
        try:
            approval_module._approval_session_key.reset(token)
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

    def test_pre_and_post_fire_on_gateway_surface(
        self, isolated_session, monkeypatch
    ):
        import threading

        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
        # Short gateway_timeout so a buggy test fails fast instead of hanging
        monkeypatch.setattr(
            approval_module, "_get_approval_config", lambda: {"gateway_timeout": 10}
        )

        captured = []

        def fake_invoke_hook(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        notify_seen = threading.Event()

        def notify_cb(approval_data):
            notify_seen.set()

        register_gateway_notify(isolated_session, notify_cb)
        result_holder = {}

        def run_guard():
            with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
                result_holder["result"] = check_all_command_guards(
                    "rm -rf /tmp/test-gateway-hook", "local",
                )

        t = threading.Thread(target=run_guard, daemon=True)
        t.start()

        # Wait for the gateway callback to see the approval request
        assert notify_seen.wait(timeout=5), "Gateway notify never fired"

        # User approves from the "other thread" (simulating /approve command)
        resolve_gateway_approval(isolated_session, "once")

        t.join(timeout=5)
        assert not t.is_alive(), "Agent thread never unblocked"
        unregister_gateway_notify(isolated_session)

        assert result_holder["result"]["approved"] is True

        hook_names = [c[0] for c in captured]
        assert "pre_approval_request" in hook_names
        assert "post_approval_response" in hook_names

        pre_kwargs = next(kw for name, kw in captured if name == "pre_approval_request")
        assert pre_kwargs["surface"] == "gateway"
        assert pre_kwargs["command"] == "rm -rf /tmp/test-gateway-hook"

        post_kwargs = next(kw for name, kw in captured if name == "post_approval_response")
        assert post_kwargs["surface"] == "gateway"
        assert post_kwargs["choice"] == "once"

    def test_timeout_reports_timeout_choice(self, isolated_session, monkeypatch):
        import threading

        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
        monkeypatch.setattr(
            approval_module, "_get_approval_config", lambda: {"gateway_timeout": 1}
        )

        captured = []

        def fake_invoke_hook(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        notify_seen = threading.Event()

        def notify_cb(approval_data):
            notify_seen.set()

        register_gateway_notify(isolated_session, notify_cb)
        result_holder = {}

        def run_guard():
            with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
                result_holder["result"] = check_all_command_guards(
                    "rm -rf /tmp/test-gateway-timeout", "local",
                )

        t = threading.Thread(target=run_guard, daemon=True)
        t.start()
        assert notify_seen.wait(timeout=5)
        # Deliberately do NOT resolve -- let it time out
        t.join(timeout=5)
        assert not t.is_alive()
        unregister_gateway_notify(isolated_session)

        assert result_holder["result"]["approved"] is False

        post_kwargs = next(kw for name, kw in captured if name == "post_approval_response")
        assert post_kwargs["choice"] == "timeout"
