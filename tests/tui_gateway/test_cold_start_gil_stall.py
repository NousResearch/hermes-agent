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

from unittest.mock import patch



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






# ─── Fix 3: _warm_gateway_module pre-imports heavy chains ──────────────


