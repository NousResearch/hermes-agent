"""``approvals.<context>_mode: ask`` — an unattended run (cron / -q / webhook) pauses on the operator's
SELECTED plugin approval transport instead of resolving from config.

Inspired by Perplexity Computer's Automations ("pause for review before consequential actions").
Invariants:
  * every unattended gate (shell command, execute_code, plugin-escalated action) reaches the selected
    transport under ``ask`` with ``request.surface`` naming the context, and the answer decides;
  * ``ask`` with no transport selected is ``deny`` — nothing waits, nothing auto-approves.
"""

import pytest

import tools.approval as approval_module
from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools import approval_context, approval_prompt


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    reset_session_vars()
    for var in ("HERMES_CRON_SESSION", "HERMES_GATEWAY_SESSION", "HERMES_INTERACTIVE", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_module, "_command_matches_permanent_allowlist", lambda command: False)
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    reset_session_vars()


def _select_transport(monkeypatch, answer: str):
    """Register a transport named ``phone`` and select it; returns the requests it saw."""
    manager = PluginManager()
    seen = []
    manifest = PluginManifest(name="phone-plugin", version="1.0.0", description="fixture", source="user",
                              key="phone-plugin")
    PluginContext(manifest, manager).register_approval_transport(
        "phone", lambda request: seen.append(request) or request.respond(answer))
    monkeypatch.setattr(approval_prompt, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("phone", None))
    return seen


def _cron_mode(monkeypatch, mode: str):
    monkeypatch.setattr(approval_context, "_get_cron_approval_mode", lambda: mode)


def test_ask_routes_every_unattended_gate_through_the_selected_transport(monkeypatch):
    seen = _select_transport(monkeypatch, "once")
    _cron_mode(monkeypatch, "ask")
    tokens = set_session_vars(cron_session="1")
    try:
        command = approval_module.check_all_command_guards("rm -rf ./build", "local")
        code = approval_module.check_execute_code_guard("import os", "local")
        action = approval_module.request_tool_approval("send_email", "plugin wants a confirmation")
    finally:
        clear_session_vars(tokens)

    assert [r["approved"] for r in (command, code, action)] == [True, True, True]
    assert command["user_approved"] is True
    assert [r.surface for r in seen] == ["cron", "cron", "cron"]
    assert seen[1].pattern_key == "execute_code"
    assert seen[2].pattern_key.startswith("plugin_rule:send_email:")

    # A transport deny is a hard halt with the consent contract, not a bare block.
    seen = _select_transport(monkeypatch, "deny")
    tokens = set_session_vars(cron_session="1")
    try:
        denied = approval_module.check_all_command_guards("rm -rf ./build", "local")
    finally:
        clear_session_vars(tokens)
    assert denied["approved"] is False
    assert denied["outcome"] == "denied"
    assert denied["user_consent"] is False
    assert len(seen) == 1


def test_ask_without_a_selected_transport_is_deny_and_session_answers_persist(monkeypatch):
    monkeypatch.setattr(approval_context, "_get_approval_transport_config", lambda: ("builtin", None))
    _cron_mode(monkeypatch, "ask")
    tokens = set_session_vars(cron_session="1")
    try:
        command = approval_module.check_all_command_guards("rm -rf ./build", "local")
        code = approval_module.check_execute_code_guard("import os", "local")
    finally:
        clear_session_vars(tokens)
    assert command["approved"] is False
    assert "approvals.cron_mode" in command["message"]
    assert code["approved"] is False and code["outcome"] == "blocked"

    # "Approve for session" on the phone answers the next identical prompt without a second push.
    seen = _select_transport(monkeypatch, "session")
    tokens = set_session_vars(cron_session="1")
    try:
        first = approval_module.check_execute_code_guard("import os", "local")
        second = approval_module.check_execute_code_guard("import sys", "local")
    finally:
        clear_session_vars(tokens)
    assert first["approved"] is True and second["approved"] is True
    assert len(seen) == 1
