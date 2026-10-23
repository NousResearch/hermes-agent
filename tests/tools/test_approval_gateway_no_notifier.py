"""An attended gateway session whose turn-scoped approval notifier is gone must fail closed.

#133514: a background ``delegate_task`` child outlives the parent turn, and the parent's
``finally`` unregisters the session's gateway notify callback. A later approval request from
that child used to return ``pending_approval`` — an answerable-looking status — while nothing
could deliver a prompt (``/approve`` resolves only blocking gateway entries; the ``_pending``
fallback store has no reader). The gate must deny with the delivery-failure outcome instead.
"""

import pytest

import tools.approval as approval
import tools.approval_context as approval_context
from tools.approval import check_dangerous_command, request_tool_approval


SESSION = "agent:main:telegram:dm:synthetic-133514"


@pytest.fixture(autouse=True)
def _isolated_session(monkeypatch):
    """Bind one synthetic session key and keep every ambient approval marker out."""
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setattr(approval_context, "get_current_session_key", lambda default="": SESSION)
    monkeypatch.setattr(approval, "get_current_session_key", lambda default="": SESSION)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None, raising=False)
    # No notifier registered for SESSION — the parent turn already unregistered it.
    monkeypatch.setattr(approval, "_gateway_notify_cb", lambda session_key: None)
    with approval._lock:
        approval._pending.pop(SESSION, None)
        for entry in approval._gateway_queues.pop(SESSION, []):
            entry.event.set()
    yield
    with approval._lock:
        approval._pending.pop(SESSION, None)
        approval._gateway_queues.pop(SESSION, None)


def _attended_gateway_session(monkeypatch):
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"approvals": {"mode": "manual"}},
    )


class TestGatewayNoNotifierFailsClosed:
    def test_command_gate_denies_instead_of_pending(self, monkeypatch):
        _attended_gateway_session(monkeypatch)
        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert result["approved"] is False
        assert result.get("status") != "pending_approval"
        assert result.get("outcome") == "notify_failed"
        # Not an answerable-looking "asking the user" lie: the delivery channel is gone.
        assert "Asking the user" not in result.get("message", "")
        assert "no approval channel" in result["message"]

    def test_no_dead_end_pending_entry_is_stored(self, monkeypatch):
        _attended_gateway_session(monkeypatch)
        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert result["approved"] is False
        assert SESSION not in approval._pending
        assert approval.has_blocking_approval(SESSION) is False

    def test_action_gate_denies_instead_of_approval_required(self, monkeypatch):
        _attended_gateway_session(monkeypatch)
        result = request_tool_approval("home_lock", "unlock the front door", rule_key="unlock")
        assert result["approved"] is False
        assert result.get("status") != "approval_required"
        assert result.get("outcome") == "notify_failed"
        assert "no approval channel" in result["message"]


class TestAskBridgeKeepsPendingFallback:
    def test_ask_leak_without_notifier_still_pends(self, monkeypatch):
        """The ask bridge without a notifier (HERMES_EXEC_ASK leaking past its bridge)
        is the one surface still served by the pending fallback — unchanged by #133514."""
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"approvals": {"mode": "manual"}},
        )
        monkeypatch.setenv("HERMES_EXEC_ASK", "1")
        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        # check_dangerous_command routes through the action gate, whose pending
        # fallback shape is approval_required (the command gate's is pending_approval).
        assert result.get("status") == "approval_required"
        assert "Asking the user" in result.get("message", "")
        assert SESSION in approval._pending


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
