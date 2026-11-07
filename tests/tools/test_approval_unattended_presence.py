"""Unattended-platform runs resolve flagged commands from ``approvals.unattended_mode`` unless a
client can actually answer the card. The gateway exports ``HERMES_EXEC_ASK=1`` to every session
it runs, so without this a webhook run posted its card to the route's delivery chat (where
``/approve`` resolves a different session) and a non-streaming api_server chat parked on a
``pending_approval`` its client never sees; both hung for the full approval timeout (#100532)."""

import pytest

import tools.approval as approval_mod
from tools import approval_context

SESSION = "test-unattended-presence"


@pytest.fixture
def unattended_session(monkeypatch):
    monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    for var in ("HERMES_CRON_SESSION", "HERMES_GATEWAY_SESSION", "HERMES_INTERACTIVE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION)
    monkeypatch.setattr(approval_mod, "_human_decision",
                        lambda *a, **k: pytest.fail("unanswerable run reached the human approval gate"))
    approval_mod._session_approved.pop(SESSION, None)
    yield lambda platform: monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    approval_mod.unregister_gateway_notify(SESSION)


@pytest.mark.parametrize("unattended_mode", ["deny", "approve"])
@pytest.mark.parametrize("platform", ["webhook", "msgraph_webhook"])
def test_webhook_resolves_from_unattended_mode_even_with_turn_runner_notifier(
        unattended_session, monkeypatch, platform, unattended_mode):
    # The generic TurnRunner lane registers a notifier on every turn; these adapters still cannot answer it.
    unattended_session(platform)
    monkeypatch.setattr(approval_context, "_get_unattended_approval_mode", lambda: unattended_mode)
    approval_mod.register_gateway_notify(SESSION, lambda data: pytest.fail("card sent to a platform nobody answers"))
    result = approval_mod.check_all_command_guards("sudo systemctl restart nginx", "local")
    assert result["approved"] is (unattended_mode == "approve")
    if unattended_mode == "deny":
        assert "approvals.unattended_mode" in result["message"]


def test_api_server_asks_only_when_a_client_bridge_is_registered(unattended_session):
    unattended_session("api_server")
    result = approval_mod.check_all_command_guards("sudo systemctl restart nginx", "local")
    assert result["approved"] is False and "approvals.unattended_mode" in result["message"]

    # /v1/runs and streaming chat completions register a notifier: the approval bridge keeps the card.
    approval_mod.register_gateway_notify(SESSION, lambda data: None)
    assert approval_mod._presence()[3] is True
