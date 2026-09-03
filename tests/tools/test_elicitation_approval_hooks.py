"""Regression test: request_elicitation_consent()'s CLI/TUI branch must fire
pre_approval_request / post_approval_response, same as its own gateway branch
(via _await_gateway_decision) and every other prompt_dangerous_approval call
site (check_all_command_guards, check_execute_code_guard) — see
tests/tools/test_approval_plugin_hooks.py.

Before this fix, an MCP elicitation answered on a CLI/TUI session called
prompt_dangerous_approval() directly with zero hook instrumentation, making
it invisible to the tool-approval observability pipeline while the
structurally identical dangerous-command approval flow fires hooks.
"""
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools.approval import request_elicitation_consent


@pytest.fixture
def cli_session(monkeypatch):
    """Force the CLI/TUI branch: no gateway session context."""
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(
        approval_module, "get_current_session_key", lambda default="default": "cli:test-session"
    )


class TestElicitationCliHooksFire:
    def test_accept_fires_pre_and_post_with_choice(self, cli_session, monkeypatch):
        captured = []
        monkeypatch.setattr(
            approval_module,
            "_fire_approval_hook",
            lambda hook_name, **kw: captured.append((hook_name, kw)),
        )
        monkeypatch.setattr(
            approval_module, "prompt_dangerous_approval", lambda *a, **kw: "once"
        )

        result = request_elicitation_consent("Allow X?", "MCP server wants to do X")

        assert result == "accept"
        hook_names = [c[0] for c in captured]
        assert hook_names == ["pre_approval_request", "post_approval_response"]

        pre_kwargs = captured[0][1]
        assert pre_kwargs["command"] == "Allow X?"
        assert pre_kwargs["description"] == "MCP server wants to do X"
        assert pre_kwargs["pattern_key"] == "mcp_elicitation"
        assert pre_kwargs["pattern_keys"] == ["mcp_elicitation"]
        assert pre_kwargs["session_key"] == "cli:test-session"
        assert pre_kwargs["surface"] == "mcp-elicitation"

        post_kwargs = captured[1][1]
        assert post_kwargs["choice"] == "once"
        assert post_kwargs["session_key"] == "cli:test-session"

    def test_decline_still_fires_post_hook(self, cli_session, monkeypatch):
        captured = []
        monkeypatch.setattr(
            approval_module,
            "_fire_approval_hook",
            lambda hook_name, **kw: captured.append((hook_name, kw)),
        )
        monkeypatch.setattr(
            approval_module, "prompt_dangerous_approval", lambda *a, **kw: "declined"
        )

        result = request_elicitation_consent("Allow X?", "desc")

        assert result == "decline"
        assert [c[0] for c in captured] == ["pre_approval_request", "post_approval_response"]
        assert captured[1][1]["choice"] == "declined"

    def test_prompt_exception_still_fires_post_hook_and_fails_closed(
        self, cli_session, monkeypatch
    ):
        """Fail-closed contract must hold even with hooks added: an
        exception in the CLI prompt still declines, and still reports a
        completed (not silently dropped) post_approval_response."""
        captured = []
        monkeypatch.setattr(
            approval_module,
            "_fire_approval_hook",
            lambda hook_name, **kw: captured.append((hook_name, kw)),
        )

        def _raise(*a, **kw):
            raise RuntimeError("prompt backend unavailable")

        monkeypatch.setattr(approval_module, "prompt_dangerous_approval", _raise)

        result = request_elicitation_consent("Allow X?", "desc")

        assert result == "decline"
        assert [c[0] for c in captured] == ["pre_approval_request", "post_approval_response"]
        assert captured[1][1]["choice"] == "error"

    def test_surface_kwarg_propagates_to_hooks(self, cli_session, monkeypatch):
        captured = []
        monkeypatch.setattr(
            approval_module,
            "_fire_approval_hook",
            lambda hook_name, **kw: captured.append((hook_name, kw)),
        )
        monkeypatch.setattr(
            approval_module, "prompt_dangerous_approval", lambda *a, **kw: "once"
        )

        request_elicitation_consent("Allow X?", "desc", surface="custom-mcp-server")

        assert captured[0][1]["surface"] == "custom-mcp-server"
        assert captured[1][1]["surface"] == "custom-mcp-server"
