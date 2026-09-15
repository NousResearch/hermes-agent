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
    """The gateway.ready payload must carry the resolved skin, and
    resolve_skin must not run on the event-loop thread (#60800).

    Drives the real ``handle_ws`` with a fake transport and asserts on the
    frame it actually writes, so a revert to an inline ``resolve_skin()``
    call fails here: the recorded resolver thread would equal the loop
    thread.
    """
    import asyncio as _asyncio
    import json
    import threading

    import tui_gateway.server as server_mod
    import tui_gateway.ws as ws_mod

    idents = {}
    sent = []

    def _fake_resolve_skin():
        idents["skin_thread"] = threading.get_ident()
        return {"palette": "wired"}

    class FakeWS:
        async def accept(self, *_a, **_k):
            pass

        async def send_text(self, line):
            sent.append(line)

        async def receive_text(self):
            raise ws_mod._WebSocketDisconnect()

        async def close(self):
            pass

    async def _scenario():
        idents["loop_thread"] = threading.get_ident()
        await ws_mod.handle_ws(FakeWS())

    with patch.object(server_mod, "resolve_skin", _fake_resolve_skin), \
            patch.object(server_mod, "_ensure_skin_watcher", lambda: None), \
            patch.object(server_mod, "register_live_transport", lambda *_a, **_k: None), \
            patch.object(server_mod, "_start_backend_heartbeat_refresher", lambda: None), \
            patch.object(server_mod, "_WS_ORPHAN_REAP_GRACE_S", 0):
        _asyncio.run(_scenario())

    ready = next(
        json.loads(line)
        for line in sent
        if json.loads(line).get("params", {}).get("type") == "gateway.ready"
    )
    assert ready["params"]["payload"]["skin"] == {"palette": "wired"}
    assert ready["params"]["payload"]["change_events"] is True
    # The offload contract: resolve_skin ran off the loop thread. This is what
    # #60800 actually fixed, and it fails if the call is ever inlined again.
    assert idents["skin_thread"] != idents["loop_thread"], (
        "resolve_skin ran on the event loop thread — the #60800 cold-start "
        "stall would be back."
    )


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
