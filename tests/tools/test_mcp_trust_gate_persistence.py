"""Persistence of MCP trust-gate approval scopes (#109818).

A gateway ``session``/``always`` answer to a trust-gate prompt must be remembered
under the per-(server, tool) key the gate consults on the next call; a permanent
``always`` entry must skip the prompt entirely, including in a fresh session.
Server-initiated elicitations (no ``persist_key``) keep the pre-existing per-call
behaviour. All approval state is mocked or sandboxed; nothing touches real config.
"""

import pytest
from unittest.mock import patch

from tools import approval as approval_mod
from tools.approval_prompt import request_elicitation_consent


@pytest.fixture
def clean_approval_state(monkeypatch, tmp_path):
    monkeypatch.setattr(approval_mod, "save_permanent_allowlist", lambda patterns: None)

    def _reset():
        approval_mod.load_permanent(set())
        with approval_mod._lock:
            approval_mod._session_approved.clear()

    _reset()
    yield
    _reset()


def _gateway(choice, captured=None):
    """Patch the gateway surface so decisions resolve with *choice* without a real gateway."""
    def _decision(session_key, notify_cb, data, *, surface="gateway"):
        if captured is not None:
            captured.update(data)
        return {"resolved": True, "choice": choice}

    return patch("tools.approval_gateway_wait._await_gateway_decision", side_effect=_decision)


def _gateway_context():
    return [
        patch("tools.approval_context._is_gateway_approval_context", return_value=True),
        patch("tools.approval._gateway_notify_cb", return_value=lambda **kw: None),
    ]


class TestGatewayChoicePersistence:
    def test_always_persists_permanent_entry(self, clean_approval_state):
        with _gateway_context()[0], _gateway_context()[1], _gateway("always"):
            out = request_elicitation_consent("run tool?", "desc", persist_key="mcp:srv/tool")
        assert out == "accept"
        # A *fresh* session is approved: the entry survived as permanent.
        assert approval_mod.is_approved("fresh-session", "mcp:srv/tool")

    def test_session_choice_scoped_to_answering_session(self, clean_approval_state):
        with _gateway_context()[0], _gateway_context()[1], _gateway("session"):
            out = request_elicitation_consent("run tool?", "desc", persist_key="mcp:srv/tool")
        assert out == "accept"
        # The answering session ("default") remembers; a different one does not.
        assert approval_mod.is_approved("default", "mcp:srv/tool")
        assert not approval_mod.is_approved("other-session", "mcp:srv/tool")

    def test_once_answer_persists_nothing(self, clean_approval_state):
        with _gateway_context()[0], _gateway_context()[1], _gateway("once"):
            out = request_elicitation_consent("run tool?", "desc", persist_key="mcp:srv/tool")
        assert out == "accept"
        assert not approval_mod.is_approved("default", "mcp:srv/tool")
        assert not approval_mod.is_approved("fresh-session", "mcp:srv/tool")

    def test_without_persist_key_always_stays_per_call(self, clean_approval_state):
        """Server-initiated elicitations have no (server, tool) identity to key a standing
        approval on, so they must keep the pre-existing per-call behaviour."""
        with _gateway_context()[0], _gateway_context()[1], _gateway("always"):
            out = request_elicitation_consent("confirm?", "desc")
        assert out == "accept"
        assert not approval_mod.is_approved("default", "mcp_elicitation")
        assert not approval_mod.is_approved("fresh-session", "mcp:srv/tool")

    def test_persistence_failure_never_downgrades_an_accepted_call(self, clean_approval_state, monkeypatch):
        def _boom(patterns):
            raise OSError("config unwritable")

        monkeypatch.setattr(approval_mod, "save_permanent_allowlist", _boom)
        with _gateway_context()[0], _gateway_context()[1], _gateway("always"):
            out = request_elicitation_consent("run tool?", "desc", persist_key="mcp:srv/tool")
        # The call was already approved; a persistence failure only means re-asking next time.
        assert out == "accept"


class TestCliChoicePersistence:
    def test_cli_session_choice_remembered_when_keyed(self, clean_approval_state):
        with patch("tools.approval_prompt.prompt_dangerous_approval", return_value="session"):
            out = request_elicitation_consent("run tool?", "desc", persist_key="mcp:srv/tool")
        assert out == "accept"
        assert approval_mod.is_approved("default", "mcp:srv/tool")
        assert not approval_mod.is_approved("other", "mcp:srv/tool")


class TestTrustGateShortCircuit:
    def test_permanent_entry_skips_the_prompt(self, clean_approval_state):
        from tools import mcp_tool, mcp_tool_handlers
        approval_mod.load_permanent({"mcp:srv/tool"})
        with patch.dict(mcp_tool._server_trust_levels, {"srv": "untrusted"}), \
             patch("tools.mcp_tool_scope._resolve_server_key", return_value="srv"), \
             patch("tools.approval_prompt.request_elicitation_consent",
                   side_effect=AssertionError("approved entry must not re-prompt")) as prompt:
            out = mcp_tool_handlers._trust_gate_check("srv", "tool")
        assert out is None
        prompt.assert_not_called()

    def test_prompt_passes_per_server_tool_key(self, clean_approval_state):
        from tools import mcp_tool, mcp_tool_handlers
        with patch.dict(mcp_tool._server_trust_levels, {"srv": "untrusted"}), \
             patch("tools.mcp_tool_scope._resolve_server_key", return_value="srv"), \
             patch("tools.approval_prompt.request_elicitation_consent", return_value="accept") as prompt:
            out = mcp_tool_handlers._trust_gate_check("srv", "tool")
        assert out is None
        assert prompt.call_args.kwargs["persist_key"] == "mcp:srv/tool"

    def test_sibling_tool_not_covered_by_another_tools_approval(self, clean_approval_state):
        """The key is per (server, tool): approving one write-capable tool never widens
        to the server's other tools."""
        from tools import mcp_tool, mcp_tool_handlers
        approval_mod.load_permanent({"mcp:srv/tool"})
        with patch.dict(mcp_tool._server_trust_levels, {"srv": "untrusted"}), \
             patch("tools.mcp_tool_scope._resolve_server_key", return_value="srv"), \
             patch("tools.approval_prompt.request_elicitation_consent", return_value="decline") as prompt:
            out = mcp_tool_handlers._trust_gate_check("srv", "other-tool")
        assert out is not None
        prompt.assert_called_once()

    def test_allowlist_lookup_failure_falls_through_to_prompting(self, clean_approval_state):
        from tools import mcp_tool, mcp_tool_handlers
        with patch.dict(mcp_tool._server_trust_levels, {"srv": "untrusted"}), \
             patch("tools.mcp_tool_scope._resolve_server_key", return_value="srv"), \
             patch("tools.approval.is_approved", side_effect=RuntimeError("state unavailable")), \
             patch("tools.approval_prompt.request_elicitation_consent", return_value="decline") as prompt:
            out = mcp_tool_handlers._trust_gate_check("srv", "tool")
        assert out is not None  # declined — but the user was asked, never silently waved through
        prompt.assert_called_once()
