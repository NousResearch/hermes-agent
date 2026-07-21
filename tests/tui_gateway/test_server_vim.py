"""Tests for vi_mode / tui_vi_mode config.set in the TUI JSON-RPC gateway.

The gateway's ``config.set`` handler previously rejected ''vi_mode'' as an
unknown key.  These tests verify both keys are accepted and write values to the
correct YAML config keys.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Use pytest's temp dir as HERMES_HOME for this test."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def set_config(key: str, value: str):
    """Call config.set(key, value) and return the result dict."""
    from tui_gateway.server import handle_request

    return handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "config.set", "params": {"key": key, "value": value}}
    )


def get_full_config():
    """Call config.get(key=full) and return {'display': ...} or empty dict."""
    from tui_gateway.server import handle_request

    resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "config.get", "params": {"key": "full"}})
    cfg = resp.get("result", {}).get("config", {})
    return cfg.get("display") or {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigSetViMode:
    @pytest.mark.usefixtures("isolated_home")
    def test_set_vi_mode_on(self):
        """Setting vi_mode=true writes display.vi_mode=True."""
        result = set_config("vi_mode", "true")
        assert result.get("result", {}).get("value") == "on"

        display = get_full_config()
        assert display.get("vi_mode") is True

    @pytest.mark.usefixtures("isolated_home")
    def test_set_vi_mode_off(self):
        """Setting vi_mode=false writes display.vi_mode=False."""
        result = set_config("vi_mode", "false")
        assert result.get("result", {}).get("value") == "off"

        display = get_full_config()
        assert display.get("vi_mode") is False

    @pytest.mark.usefixtures("isolated_home")
    def test_set_vi_mode_toggle(self):
        """Toggle flips the vi_mode value."""
        # Start with true
        set_config("vi_mode", "true")

        # Now toggle
        result = set_config("vi_mode", "")
        assert result.get("result", {}).get("value") == "off"

        display = get_full_config()
        assert display.get("vi_mode") is False

    @pytest.mark.usefixtures("isolated_home")
    def test_set_tui_vi_mode_separate_key(self):
        """tui_vi_mode writes display.tui_vi_mode, not display.vi_mode."""
        # Pre-set vi_mode to true
        set_config("vi_mode", "true")

        # Then set tui_vi_mode to false
        result = set_config("tui_vi_mode", "false")
        assert result.get("result", {}).get("value") == "off"

        display = get_full_config()
        # vi_mode should still be true
        assert display.get("vi_mode") is True
        # tui_vi_mode should be false
        assert display.get("tui_vi_mode") is False

    @pytest.mark.usefixtures("isolated_home")
    def test_tui_vi_mode_does_not_affect_vi_mode(self):
        """Toggling tui_vi_mode leaves vi_mode unchanged."""
        set_config("vi_mode", "true")

        set_config("tui_vi_mode", "on")
        display1 = get_full_config()
        assert display1.get("tui_vi_mode") is True
        assert display1.get("vi_mode") is True

        set_config("tui_vi_mode", "off")
        display2 = get_full_config()
        assert display2.get("tui_vi_mode") is False
        # vi_mode not affected
        assert display2.get("vi_mode") is True

    @pytest.mark.usefixtures("isolated_home")
    def test_vi_mode_accepts_alias_values(self):
        """Various truthy/falsy aliases work."""
        for val in ("1", "true", "yes", "on"):
            set_config("vi_mode", val)
            display = get_full_config()
            assert display.get("vi_mode") is True, f"value {val!r} should be True"

        for val in ("0", "false", "no", "off"):
            set_config("vi_mode", val)
            display = get_full_config()
            assert display.get("vi_mode") is False, f"value {val!r} should be False"

    @pytest.mark.usefixtures("isolated_home")
    def test_unknown_value_returns_error(self):
        """Invalid value returns a 4002 error."""
        result = set_config("vi_mode", "maybe")
        assert "error" in result
        assert result["error"]["code"] == 4002
        assert "unknown vi_mode" in result["error"]["message"].lower()
