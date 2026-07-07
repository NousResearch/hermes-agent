"""Regression test: save_trajectories config must reach the TUI gateway AIAgent.

The TUI runs agent turns inside tui_gateway.slash_worker subprocesses. Each
worker calls _make_agent() in tui_gateway.server to construct the AIAgent.
That function reads agent_cfg from _load_cfg(), but save_trajectories was
never forwarded to AIAgent(), so the config setting had no effect for --tui
sessions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def tui_home(tmp_path):
    """Isolated HERMES_HOME pointing to a tmp directory.

    Uses set_hermes_home_override so _load_cfg() inside tui_gateway.server
    reads the fixture's config.yaml without reimporting the module.
    """
    from tui_gateway import server as server_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    token = set_hermes_home_override(str(home))
    # Bust the mtime/path cache so _load_cfg() reads the new file.
    prev = (server_mod._cfg_cache, server_mod._cfg_mtime, server_mod._cfg_path)
    server_mod._cfg_cache = None
    server_mod._cfg_mtime = None
    server_mod._cfg_path = None

    yield home, server_mod

    server_mod._cfg_cache = prev[0]
    server_mod._cfg_mtime = prev[1]
    server_mod._cfg_path = prev[2]
    reset_hermes_home_override(token)


def _call_make_agent(server_mod, mock_agent_cls):
    """Call server._make_agent with the minimum mocking to reach AIAgent(...)."""
    with patch.object(
        server_mod,
        "_resolve_startup_runtime",
        return_value=("anthropic/claude-sonnet-4-6", "openrouter"),
    ), patch.object(
        server_mod,
        "_resolve_runtime_with_fallback",
        return_value={
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        },
    ), patch(
        "run_agent.AIAgent", mock_agent_cls
    ):
        server_mod._make_agent(
            sid="test-sid",
            key="test-key",
            session_db=MagicMock(),  # skip _get_db() DB init
        )


class TestMakeAgentSaveTrajectoriesWiring:
    def test_passes_save_trajectories_true_from_config(self, tui_home):
        """_make_agent must forward save_trajectories=True when config says so."""
        home, server_mod = tui_home
        (home / "config.yaml").write_text("agent:\n  save_trajectories: true\n")

        mock_agent_cls = MagicMock()
        _call_make_agent(server_mod, mock_agent_cls)

        mock_agent_cls.assert_called_once()
        _, kwargs = mock_agent_cls.call_args
        assert kwargs.get("save_trajectories") is True

    def test_passes_save_trajectories_false_by_default(self, tui_home):
        """When config omits save_trajectories, _make_agent must pass False."""
        home, server_mod = tui_home
        (home / "config.yaml").write_text("agent: {}\n")

        mock_agent_cls = MagicMock()
        _call_make_agent(server_mod, mock_agent_cls)

        mock_agent_cls.assert_called_once()
        _, kwargs = mock_agent_cls.call_args
        assert kwargs.get("save_trajectories") is False
