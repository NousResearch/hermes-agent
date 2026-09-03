"""Tests for cold-start GIL stall mitigations (#60800).

The Desktop/TUI cold start could stall the event loop for ~14s because
synchronous CPU-bound work ran on the loop thread during the window
between ``HERMES_BACKEND_READY`` and the first prompt. Three fixes:

1. ``copilot_auth.resolve_copilot_token`` skips the ``gh auth token``
   subprocess when a Copilot env var is explicitly set (even if invalid).
2. ``gateway.ready`` reads a startup-primed skin snapshot and a rare cold
   cache refreshes once in the background, never in the liveness path.
3. ``web_server._warm_gateway_module`` pre-imports the heavy module
   chains that the first WS connection + RPC burst would otherwise
   import on the loop thread.
"""

import asyncio
import sys
from unittest.mock import patch, MagicMock

import pytest


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


# ─── Fix 2: gateway.ready never waits for skin resolution ─────────────


def test_cold_skin_refresh_runs_off_the_loop_thread():
    """The cache-miss fallback must resolve on a worker and remain awaitable."""
    import asyncio as _asyncio
    import threading

    import tui_gateway.server as server_mod
    import tui_gateway.ws as ws_mod

    idents = {}

    def _fake_resolve_skin_snapshot():
        idents["skin_thread"] = threading.get_ident()
        return {"palette": "test"}, 1

    async def _scenario():
        idents["loop_thread"] = threading.get_ident()
        ws_mod._skin_refresh_task = None
        ws_mod._skin_refresh_loop = None
        try:
            with (
                patch.object(
                    server_mod,
                    "resolve_skin_snapshot",
                    _fake_resolve_skin_snapshot,
                ),
                patch.object(server_mod, "_note_cached_skin_broadcast", return_value=True),
                patch.object(server_mod, "_broadcast_global_event"),
            ):
                return await ws_mod._ensure_skin_cache_refresh()
        finally:
            ws_mod._skin_refresh_task = None
            ws_mod._skin_refresh_loop = None

    payload = _asyncio.run(_scenario())

    assert payload == {"palette": "test"}
    assert idents["skin_thread"] != idents["loop_thread"]


def test_handle_ws_ready_payload_uses_no_io_skin_snapshot(monkeypatch):
    """A warm cache is serialized into ready without calling the resolver."""
    import json

    import tui_gateway.server as server_mod
    import tui_gateway.ws as ws_mod

    frames = []
    resolver = MagicMock(side_effect=AssertionError("ready path resolved skin"))
    monkeypatch.setattr(server_mod, "get_cached_skin_payload", lambda: {"name": "cached"})
    monkeypatch.setattr(server_mod, "resolve_skin_snapshot", resolver)
    monkeypatch.setattr(server_mod, "_WS_ORPHAN_REAP_GRACE_S", 0)

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            frames.append(json.loads(line))

        async def receive_text(self):
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws_mod.handle_ws(FakeWS()))

    assert frames[0]["params"]["type"] == "gateway.ready"
    assert frames[0]["params"]["payload"]["skin"] == {"name": "cached"}
    resolver.assert_not_called()


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

    web_server_mod._warm_gateway_module()

    from tui_gateway import server as gateway_server

    assert gateway_server.get_cached_skin_payload().get("name")

    missing = required - set(sys.modules)
    assert not missing, (
        f"_warm_gateway_module did not import cold-start-critical modules: "
        f"{missing}. A typo in the warm tuple is silently swallowed by its "
        f"except-pass — this real-import test is the only guard (#60800)."
    )
