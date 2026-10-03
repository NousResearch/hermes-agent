"""Distinguish approval timeout / callback error / unreachable channel from user denial (#130272).

When a dangerous-command approval times out or cannot be shown, the agent must
NOT be told "User denied this command" — no prompt was shown and no user
decision was made. The three non-human outcomes are distinct from an
affirmative human denial in both the prompt return value and the tool message.
"""

import threading

import pytest

from tools import approval as mod
from tools import approval_context
from tools.approval_prompt import prompt_dangerous_approval


@pytest.fixture
def cli_env(monkeypatch):
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    for k in ("HERMES_CRON_SESSION", "HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "test-130272")
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "timeout": 60})
    monkeypatch.setattr(approval_context, "_fire_approval_hook", lambda name, **kw: None)
    yield
    mod._session_approved.clear()
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()


def test_prompt_timeout_is_distinct_from_deny():
    """The raw input() path returns 'timeout', not 'deny', on expiry."""
    import builtins
    from unittest.mock import patch as _patch
    import time

    def _hang(_prompt=""):
        time.sleep(10)
        return ""

    with _patch.object(builtins, "input", _hang):
        result = prompt_dangerous_approval("rm -rf /var/data", "recursive delete", timeout_seconds=0.05)
    assert str(result) == "timeout"
    assert str(result) != "deny"


def test_prompt_callback_error_is_distinct_from_deny():
    """A raising callback returns 'callback_error' with a cause, not 'deny'."""

    def _broken(command, description, **kwargs):
        raise TypeError("boom")

    result = prompt_dangerous_approval("rm -rf /", "test", approval_callback=_broken)
    assert str(result) == "callback_error"
    assert str(result) != "deny"
    assert "approval callback failed" in str(getattr(result, "cause", ""))


def test_prompt_no_channel_is_distinct_from_deny(monkeypatch):
    """The prompt_toolkit fail-closed guard returns 'no_channel', not 'deny'."""
    import prompt_toolkit.application.current as ptc

    monkeypatch.setattr(ptc, "get_app_or_none", lambda: object())
    result = []

    def run():
        result.append(prompt_dangerous_approval("rm -rf /", "test", timeout_seconds=5, approval_callback=None))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "fail-closed guard hung instead of failing closed"
    assert [str(r) for r in result] == ["no_channel"]
    assert "no approval callback" in str(getattr(result[0], "cause", ""))


def test_gate_timeout_message_says_no_decision(cli_env):
    """Timeout: agent hears about the elapsed wait, never about a denial."""
    result = mod.check_all_command_guards(
        "rm -rf /var/data", "local", approval_callback=lambda *a, **k: "timeout")
    assert result["approved"] is False
    assert result.get("outcome") == "timeout"
    assert result.get("user_consent") is False
    msg = result["message"]
    assert "approval timed out after 60s with no response" in msg
    assert "No user decision was made" in msg
    assert "user denied" not in msg.lower()
    assert "denied by user" not in msg.lower()


def test_gate_callback_error_message_says_no_decision(cli_env):
    """Callback failure: fail-closed, but reported as undelivered — not denied."""

    def _broken(command, description, **kwargs):
        raise TypeError("boom")

    result = mod.check_all_command_guards("rm -rf /var/data", "local", approval_callback=_broken)
    assert result["approved"] is False
    assert result.get("outcome") == "callback_error"
    assert result.get("user_consent") is False
    msg = result["message"]
    assert "approval could not be requested (callback error)" in msg
    assert "No user decision was made" in msg
    assert "User denied this command" not in msg
    assert "denied by user" not in msg.lower()


def test_gate_no_channel_message_says_no_decision(cli_env, monkeypatch):
    """Fail-closed guard: no channel reachable, still blocked, never a denial."""
    import prompt_toolkit.application.current as ptc

    monkeypatch.setattr(ptc, "get_app_or_none", lambda: object())
    result = mod.check_all_command_guards("rm -rf /var/data", "local", approval_callback=None)
    assert result["approved"] is False
    assert result.get("outcome") == "no_channel"
    assert result.get("user_consent") is False
    msg = result["message"]
    assert "no approval channel is reachable from this thread" in msg
    assert "No user decision was made" in msg
    assert "User denied this command" not in msg
    assert "denied by user" not in msg.lower()


def test_gate_human_deny_keeps_denial_wording(cli_env):
    """Explicit human denial keeps the existing user-denied message."""
    result = mod.check_all_command_guards(
        "rm -rf /var/data", "local", approval_callback=lambda *a, **k: "deny")
    assert result["approved"] is False
    assert result.get("outcome") == "denied"
    assert "User denied this command" in result["message"]


def test_legacy_cancelled_with_callback_cause_maps_to_callback_error(cli_env):
    """Backward compat: an older 'cancelled' carrying a callback-failure cause
    still produces the accurate callback-error message, not a generic denial."""
    from tools.approval_prompt import Unanswered

    legacy = Unanswered("the approval callback failed: TypeError")
    result = mod.check_all_command_guards(
        "rm -rf /var/data", "local", approval_callback=lambda *a, **k: legacy)
    assert result.get("outcome") == "callback_error"
    assert "approval could not be requested (callback error)" in result["message"]
    assert "No user decision was made" in result["message"]


def test_legacy_cancelled_with_no_channel_cause_maps_to_no_channel(cli_env):
    """Backward compat: an older 'cancelled' carrying the guard cause still
    produces the no-channel message."""
    from tools.approval_prompt import Unanswered

    legacy = Unanswered("no approval callback is registered on this thread")
    result = mod.check_all_command_guards(
        "rm -rf /var/data", "local", approval_callback=lambda *a, **k: legacy)
    assert result.get("outcome") == "no_channel"
    assert "no approval channel is reachable from this thread" in result["message"]
