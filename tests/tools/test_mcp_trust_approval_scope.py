"""MCP trust-gate approval scopes: what a prompt offers is what the answer does.

The ``trust: untrusted`` gate offers Once / Session / Always and honours them per (server, tool)
through the real gateway approval queue; consent prompts with no identity to remember
(elicitation, sampling, payment fill) offer only Once / Deny on every surface.
"""

from unittest.mock import patch

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.config import load_config
from tools import approval, approval_prompt, mcp_tool
from tools import mcp_tool_handlers as handlers
from tools.approval_context import reset_current_session_key, set_current_session_key


@pytest.fixture(autouse=True)
def _isolated_approval_state(monkeypatch):
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_permanent_approved_by_home", {})
    monkeypatch.setattr(approval, "_permanent_baseline_by_home", {})
    with patch.dict(mcp_tool._server_trust_levels, {"srv": "untrusted"}, clear=True), \
         patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True), \
         patch.dict(mcp_tool._server_cost_policies, {}, clear=True):
        yield


class _Chat:
    """A messaging chat answering every approval card it is sent with ``choice``."""

    def __init__(self, session_key: str, choice: str):
        self.session_key, self.choice, self.cards = session_key, choice, []

    def __enter__(self):
        self._vars = set_session_vars(platform="telegram", session_key=self.session_key)
        self._key = set_current_session_key(self.session_key)
        approval.register_gateway_notify(self.session_key, self._answer)
        return self

    def __exit__(self, *exc):
        approval.unregister_gateway_notify(self.session_key)
        reset_current_session_key(self._key)
        clear_session_vars(self._vars)

    def _answer(self, data: dict) -> None:
        self.cards.append(dict(data))
        approval.resolve_gateway_approval(self.session_key, self.choice, request_id=data["request_id"])


@pytest.mark.parametrize("choice", ["session", "always"])
def test_session_and_always_answers_stop_the_next_prompt_for_that_tool_only(choice):
    with _Chat("tg:chat-a", choice) as chat:
        assert handlers._trust_gate_check("srv", "write") is None
        assert handlers._trust_gate_check("srv", "write") is None
        assert handlers._trust_gate_check("srv", "other") is None  # a sibling tool still asks
    assert len(chat.cards) == 2
    assert chat.cards[0]["allow_session"] is True and chat.cards[0]["allow_permanent"] is True

    with _Chat("tg:chat-b", "deny") as other:
        blocked = handlers._trust_gate_check("srv", "write")
    allowlist = load_config().get("command_allowlist") or []
    if choice == "session":  # another chat is not covered, and nothing reached config
        assert blocked is not None and len(other.cards) == 1
        assert "mcp:srv/write" not in allowlist
    else:
        assert blocked is None and other.cards == []
        assert "mcp:srv/write" in allowlist


def test_consent_without_an_identity_offers_and_records_only_once(monkeypatch):
    """Elicitation-style consent: the card offers Once / Deny, the CLI panel is asked for once/deny,
    and a stale broader answer approves this call without remembering anything."""
    with _Chat("tg:chat-c", "session") as chat:
        assert approval_prompt.request_elicitation_consent("Use the camera?", "server asks") == "accept"
    assert chat.cards[0]["allow_session"] is False and chat.cards[0]["allow_permanent"] is False
    assert approval._session_approved == {}

    asked = []
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback",
                        lambda: lambda *a, **kw: asked.append(kw) or "always")
    monkeypatch.setattr(approval_prompt._ctx, "_no_user_can_answer", lambda: False)
    assert approval_prompt.request_elicitation_consent("Fill card?", "pay") == "accept"
    assert asked[0]["allow_session"] is False and asked[0]["allow_permanent"] is False
    assert approval._session_approved == {} and approval._permanent_approved == set()
