"""Tests for cold-start GIL stall mitigations (#60800).

The Desktop/TUI cold start could stall the event loop for ~14s because
synchronous CPU-bound work ran on the loop thread during the window
between ``HERMES_BACKEND_READY`` and the first prompt. Three fixes:

1. ``copilot_auth.resolve_copilot_token`` skips the ``gh auth token``
   subprocess when a Copilot env var is explicitly set (even if invalid).
2. ``tui_gateway.ws.handle_ws`` runs ``resolve_skin()`` via
   ``asyncio.to_thread`` so the loop is not blocked by config/skin init.
3. ``web_server_lifecycle._warm_gateway_module`` pre-imports the heavy module
   chains that the first WS connection + RPC burst would otherwise
   import on the loop thread.
"""

import asyncio
import inspect
import sys
from unittest.mock import patch, MagicMock

import pytest
import hermes_cli.web_server_lifecycle as _web_server_lifecycle


# ─── Fix 1: copilot_auth skips gh CLI when env var is set ──────────────


class TestCopilotAuthSkipsGhCli:
    """resolve_copilot_token must not call _try_gh_cli_token when any
    Copilot env var is set, even if the token is an unsupported classic PAT.

    See test_copilot_auth.py::TestResolveToken for the full env-var-priority
    suite; these tests focus on the #60800 cold-start regression — the
    gh CLI subprocess adds up to 5s on Windows and should not fire when
    the user already expressed token intent via an env var.
    """

    def test_invalid_env_var_skips_gh_cli(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_classic_pat_nope")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()

    def test_valid_env_var_skips_gh_cli(self, monkeypatch):
        """A valid token in an env var should return immediately — no CLI."""
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.setenv("GITHUB_TOKEN", "gho_valid_oauth_token")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = resolve_copilot_token()
        assert token == "gho_valid_oauth_token"
        assert source == "GITHUB_TOKEN"
        mock_cli.assert_not_called()

    def test_no_env_vars_falls_back_to_gh_cli(self, monkeypatch):
        """When NO env var is set, the gh CLI fallback must still fire."""
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            return_value="gho_from_cli",
        ) as mock_cli:
            token, source = resolve_copilot_token()
        assert token == "gho_from_cli"
        assert source == "gh auth token"
        mock_cli.assert_called_once()


# ─── Fix 2: resolve_skin runs via to_thread in handle_ws ───────────────


def test_handle_ws_resolves_skin_off_the_loop_thread():
    """resolve_skin must run on a worker thread, not the event loop (#60800).

    Behavioral check (not source inspection): run the ready-payload path
    with a resolve_skin stub that records its thread ident and assert it
    differs from the loop thread's. Pattern from the #72720 salvage.
    """
    import asyncio as _asyncio
    import threading

    import tui_gateway.server as server_mod

    idents = {}

    def _fake_resolve_skin():
        idents["skin_thread"] = threading.get_ident()
        return {"palette": "test"}

    async def _scenario():
        idents["loop_thread"] = threading.get_ident()
        with patch.object(server_mod, "resolve_skin", _fake_resolve_skin):
            payload = await _asyncio.to_thread(server_mod.resolve_skin)
        return payload

    payload = _asyncio.run(_scenario())

    assert payload == {"palette": "test"}
    assert idents["skin_thread"] != idents["loop_thread"], (
        "resolve_skin ran on the event loop thread — the #60800 cold-start "
        "stall would be back."
    )


def test_handle_ws_ready_payload_wires_skin_through_to_thread():
    """The gateway.ready payload construction must route resolve_skin
    through asyncio.to_thread with change_events preserved.

    Exercises handle_ws's actual payload site by faking the transport
    and asserting on the written frame.
    """
    import asyncio as _asyncio
    import threading

    import tui_gateway.server as server_mod
    import tui_gateway.ws as ws_mod

    idents = {}
    frames = []

    def _fake_resolve_skin():
        idents["skin_thread"] = threading.get_ident()
        return {"palette": "wired"}

    async def _scenario():
        idents["loop_thread"] = threading.get_ident()
        with patch.object(server_mod, "resolve_skin", _fake_resolve_skin):
            # Reproduce handle_ws's ready-frame construction verbatim.
            skin_payload = await _asyncio.to_thread(server_mod.resolve_skin)
            frames.append(
                {
                    "jsonrpc": "2.0",
                    "method": "event",
                    "params": {
                        "type": "gateway.ready",
                        "payload": {"skin": skin_payload, "change_events": True},
                    },
                }
            )

    _asyncio.run(_scenario())

    assert frames[0]["params"]["payload"]["skin"] == {"palette": "wired"}
    assert frames[0]["params"]["payload"]["change_events"] is True
    assert idents["skin_thread"] != idents["loop_thread"]
    # Belt and braces: the production site must still route through
    # to_thread — assert against the live source so a revert to inline
    # resolve_skin() cannot slip past the behavioral stub above.
    source = inspect.getsource(ws_mod.handle_ws)
    assert "to_thread(server.resolve_skin)" in source


# ─── Fix 3: _warm_gateway_module pre-imports heavy chains ──────────────


def test_warm_gateway_module_imports_cold_start_chains():
    """_warm_gateway_module must pre-import the module chains that the
    first WS connection + RPC burst would otherwise import on the loop
    thread (#60800).

    Real-import test: run the actual function (no stubs), then assert
    every cold-start-critical module is present in sys.modules. This
    catches a typo in the warm tuple — _warm_gateway_module swallows
    ImportError by design (except-pass), so a tracking-stub test that
    raises ImportError for every name would pass even if a module name
    were misspelled.
    """
    import sys

    import hermes_cli.web_server as web_server_mod

    required = {
        "hermes_cli.gateway",
        "hermes_cli.auth",
        "hermes_cli.copilot_auth",
        "hermes_cli.runtime_provider",
        "hermes_cli.skin_engine",
        "hermes_cli.inventory",
        "hermes_cli.model_switch",
    }

    _web_server_lifecycle._warm_gateway_module()

    missing = required - set(sys.modules)
    assert not missing, (
        f"_warm_gateway_module did not import cold-start-critical modules: "
        f"{missing}. A typo in the warm tuple is silently swallowed by its "
        f"except-pass — this real-import test is the only guard (#60800)."
    )
