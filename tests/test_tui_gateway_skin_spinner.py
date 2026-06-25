"""Tests for TUI gateway skin integration.

These tests verify that the TUI gateway properly passes all skin configuration
to the frontend including spinner, status bar colors, and tool emojis.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestResolveSkinSpinner:
    """Tests for resolve_skin() including spinner config in payload."""

    @pytest.fixture
    def mock_skin(self):
        """Create a mock skin with full configuration."""
        skin = MagicMock()
        skin.name = "test_skin"
        skin.colors = {
            "banner_title": "#FFD700",
            "status_bar_bg": "#1a1a2e",
            "status_bar_text": "#C0C0C0",
            "status_bar_strong": "#FFD700",
            "status_bar_dim": "#8B8682",
            "status_bar_good": "#8FBC8F",
            "status_bar_warn": "#FFD700",
            "status_bar_bad": "#FF8C00",
            "status_bar_critical": "#FF6B6B",
            "selection_bg": "#333355",
        }
        skin.branding = {"agent_name": "Test Agent"}
        skin.banner_logo = "test logo"
        skin.banner_hero = "test hero"
        skin.tool_prefix = "|"
        skin.spinner = {
            "thinking_verbs": ["custom_verb1", "custom_verb2", "custom_verb3"],
            "thinking_faces": ["(¬‿¬)", "(⌐■_■)"],
            "waiting_faces": ["(¬_¬)"],
            "wings": [["<<", ">>"]],
        }
        skin.tool_emojis = {"terminal": "⚡"}
        return skin

    def test_resolve_skin_includes_spinner_config(self, mock_skin):
        """Verify resolve_skin() includes spinner in the returned payload."""
        from tui_gateway.server import resolve_skin

        with patch("tui_gateway.server._load_cfg") as mock_load_cfg, \
             patch("hermes_cli.skin_engine.init_skin_from_config") as mock_init, \
             patch("hermes_cli.skin_engine.get_active_skin", return_value=mock_skin):

            mock_load_cfg.return_value = {"display": {"skin": "test_skin"}}

            result = resolve_skin()

            assert "spinner" in result
            assert result["spinner"] == mock_skin.spinner
            assert result["spinner"]["thinking_verbs"] == ["custom_verb1", "custom_verb2", "custom_verb3"]
            assert result["spinner"]["thinking_faces"] == ["(¬‿¬)", "(⌐■_■)"]

    def test_resolve_skin_includes_tool_emojis(self, mock_skin):
        """Verify resolve_skin() includes tool_emojis in the returned payload."""
        from tui_gateway.server import resolve_skin

        with patch("tui_gateway.server._load_cfg") as mock_load_cfg, \
             patch("hermes_cli.skin_engine.init_skin_from_config") as mock_init, \
             patch("hermes_cli.skin_engine.get_active_skin", return_value=mock_skin):

            mock_load_cfg.return_value = {"display": {"skin": "test_skin"}}

            result = resolve_skin()

            assert "tool_emojis" in result
            assert result["tool_emojis"] == {"terminal": "⚡"}

    def test_resolve_skin_includes_status_bar_colors(self, mock_skin):
        """Verify resolve_skin() includes all status bar colors in payload."""
        from tui_gateway.server import resolve_skin

        with patch("tui_gateway.server._load_cfg") as mock_load_cfg, \
             patch("hermes_cli.skin_engine.init_skin_from_config") as mock_init, \
             patch("hermes_cli.skin_engine.get_active_skin", return_value=mock_skin):

            mock_load_cfg.return_value = {"display": {"skin": "test_skin"}}

            result = resolve_skin()

            # All 8 status bar colors + selection_bg should be present
            assert result["status_bar_bg"] == "#1a1a2e"
            assert result["status_bar_text"] == "#C0C0C0"
            assert result["status_bar_strong"] == "#FFD700"
            assert result["status_bar_dim"] == "#8B8682"
            assert result["status_bar_good"] == "#8FBC8F"
            assert result["status_bar_warn"] == "#FFD700"
            assert result["status_bar_bad"] == "#FF8C00"
            assert result["status_bar_critical"] == "#FF6B6B"
            assert result["selection_bg"] == "#333355"

    def test_resolve_skin_status_bar_defaults_to_none_when_missing(self):
        """Verify status bar colors default to None when not in skin."""
        from tui_gateway.server import resolve_skin

        mock_skin = MagicMock()
        mock_skin.name = "minimal_skin"
        mock_skin.colors = {"banner_title": "#FFD700"}  # No status bar colors
        mock_skin.branding = {}
        mock_skin.banner_logo = ""
        mock_skin.banner_hero = ""
        mock_skin.tool_prefix = ""
        mock_skin.spinner = {}
        mock_skin.tool_emojis = {}

        with patch("tui_gateway.server._load_cfg") as mock_load_cfg, \
             patch("hermes_cli.skin_engine.init_skin_from_config") as mock_init, \
             patch("hermes_cli.skin_engine.get_active_skin", return_value=mock_skin):

            mock_load_cfg.return_value = {"display": {"skin": "minimal_skin"}}

            result = resolve_skin()

            # Status bar colors should be None (not missing)
            assert result.get("status_bar_bg") is None
            assert result.get("status_bar_text") is None
            assert result.get("status_bar_strong") is None
            assert result.get("status_bar_dim") is None
            assert result.get("status_bar_good") is None
            assert result.get("status_bar_warn") is None
            assert result.get("status_bar_bad") is None
            assert result.get("status_bar_critical") is None
            assert result.get("selection_bg") is None

    def test_resolve_skin_handles_empty_spinner(self):
        """Verify resolve_skin() handles skins without spinner config."""
        from tui_gateway.server import resolve_skin

        mock_skin = MagicMock()
        mock_skin.name = "minimal_skin"
        mock_skin.colors = {}
        mock_skin.branding = {}
        mock_skin.banner_logo = ""
        mock_skin.banner_hero = ""
        mock_skin.tool_prefix = ""
        mock_skin.spinner = {}
        mock_skin.tool_emojis = {}

        with patch("tui_gateway.server._load_cfg") as mock_load_cfg, \
             patch("hermes_cli.skin_engine.init_skin_from_config") as mock_init, \
             patch("hermes_cli.skin_engine.get_active_skin", return_value=mock_skin):

            mock_load_cfg.return_value = {"display": {"skin": "minimal_skin"}}

            result = resolve_skin()

            assert "spinner" in result
            assert result["spinner"] == {}
            assert "tool_emojis" in result
            assert result["tool_emojis"] == {}

    def test_resolve_skin_handles_skin_engine_failure(self):
        """Verify resolve_skin() returns empty dict when skin engine fails."""
        from tui_gateway.server import resolve_skin

        with patch("tui_gateway.server._load_cfg") as mock_load_cfg, \
             patch("hermes_cli.skin_engine.init_skin_from_config") as mock_init, \
             patch("hermes_cli.skin_engine.get_active_skin", side_effect=Exception("skin error")):

            mock_load_cfg.return_value = {"display": {"skin": "broken"}}

            result = resolve_skin()

            assert result == {}

    def test_resolve_skin_preserves_other_skin_fields(self, mock_skin):
        """Verify resolve_skin() still includes all other skin fields."""
        from tui_gateway.server import resolve_skin

        with patch("tui_gateway.server._load_cfg") as mock_load_cfg, \
             patch("hermes_cli.skin_engine.init_skin_from_config") as mock_init, \
             patch("hermes_cli.skin_engine.get_active_skin", return_value=mock_skin):

            mock_load_cfg.return_value = {"display": {"skin": "test_skin"}}

            result = resolve_skin()

            # Verify all expected fields are present
            assert result["name"] == "test_skin"
            assert result["colors"]["banner_title"] == "#FFD700"
            assert result["branding"] == {"agent_name": "Test Agent"}
            assert result["banner_logo"] == "test logo"
            assert result["banner_hero"] == "test hero"
            assert result["tool_prefix"] == "|"
            assert "help_header" in result
