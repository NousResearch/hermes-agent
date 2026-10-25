"""Localized MCP approval cards keep the current trust and spend-limit contracts."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import hermes_yaml as yaml
import pytest

from agent import i18n
from gateway.session_context import clear_session_vars, set_session_vars
from tools import approval, mcp_tool, mcp_tool_costs
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_registration as registration
from tools.approval_context import reset_current_session_key, set_current_session_key


@pytest.mark.parametrize("trust", ["untrusted", "full"])
def test_localized_cards_keep_remembered_trust_and_per_call_budget_consent(tmp_path, monkeypatch, trust):
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    (home / "config.yaml").write_text("display:\n  language: de\n", encoding="utf-8")
    overlay = {
        "approval.elicitation.mcp_untrusted_head": "Anfrage {tool} auf {server}.",
        "approval.elicitation.mcp_write_warning": "Schreibzugriff.",
        "approval.elicitation.mcp_untrusted_remember_description": "Merken: {tool} auf {server}.",
        "approval.elicitation.mcp_spend_over": "Budget überschritten.",
        "approval.elicitation.mcp_spend_command": "Budget überschritten: {tool} auf {server}.",
        "approval.elicitation.mcp_arguments": "\n\nArgumente:\n{arguments}",
        "approval.elicitation.mcp_cost_about": "Kosten etwa {amount}",
    }
    (home / "locales" / "de.yaml").write_text(yaml.safe_dump(overlay, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_permanent_approved_by_home", {})
    monkeypatch.setattr(approval, "_permanent_baseline_by_home", {})
    i18n.reset_language_cache()

    session = MagicMock()
    session.call_tool = AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], isError=False, structuredContent=None))
    cards = []
    chat = "i18n-budget-chat"

    def answer(card):
        cards.append(dict(card))
        choice = "session" if trust == "untrusted" and len(cards) == 1 else "deny"
        approval.resolve_gateway_approval(chat, choice, request_id=card["request_id"])

    def run_on_loop(coro_or_factory, timeout=30):
        async def run():
            for server in mcp_tool._servers.values():
                if server._rpc_lock is None:
                    server._rpc_lock = asyncio.Lock()
            return await (coro_or_factory() if callable(coro_or_factory) else coro_or_factory)
        return asyncio.run(run())

    scope = set_session_vars(platform="telegram", session_key=chat)
    token = set_current_session_key(chat)
    approval.register_gateway_notify(chat, answer)
    try:
        with patch.dict(mcp_tool._servers, {"srv": SimpleNamespace(session=session, _rpc_lock=None)}, clear=True), \
             patch.dict(mcp_tool._server_trust_levels, {}, clear=True), \
             patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True), \
             patch.dict(mcp_tool._server_error_counts, {}, clear=True), \
             patch.dict(mcp_tool._server_cost_policies, {}, clear=True), \
             patch.dict(mcp_tool_costs._spent, {}, clear=True), \
             patch.dict(mcp_tool_costs._ask_locks, {}, clear=True), \
             patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=run_on_loop):
            registration._record_tool_trust_metadata("srv", {
                "trust": trust, "tool_costs": {"write": 40}, "spend_limit": 100,
            }, [SimpleNamespace(name="write", annotations=None)])
            call = handlers._make_tool_handler("srv", "write", 30)
            assert json.loads(call({"text": "hello"})) == {"result": "ok"}
            assert json.loads(call({"text": "again"})) == {"result": "ok"}
            refused = json.loads(call({"text": "over budget"}))
            assert "was NOT run" in refused["error"]  # the model-facing refusal stays canonical
        assert session.call_tool.await_count == 2
        assert len(cards) == (2 if trust == "untrusted" else 1)
        assert "Budget überschritten" in cards[-1]["command"]
        assert "Kosten etwa 40 credits" in cards[-1]["command"]
        assert cards[-1]["allow_session"] is False
        assert cards[-1]["allow_permanent"] is False
        if trust == "untrusted":
            assert "Anfrage write auf srv." in cards[0]["command"]
            assert 'Argumente:\n' in cards[0]["command"]
            assert cards[0]["allow_session"] is True
            assert cards[0]["allow_permanent"] is True
    finally:
        approval.unregister_gateway_notify(chat)
        reset_current_session_key(token)
        clear_session_vars(scope)
        i18n.reset_language_cache()
