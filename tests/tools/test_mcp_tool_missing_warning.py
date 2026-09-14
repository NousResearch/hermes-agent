"""Tests for the top-level mcp-missing warning in the MCP tool module."""

from __future__ import annotations

import importlib
import logging
from unittest.mock import patch

import pytest


class TestTopLevelMcpMissingWarning:
    """The import-time availability fallback must warn (not debug) when ``mcp`` is absent."""

    def test_missing_mcp_warns_at_import_time(self, caplog):
        import tools.mcp_tool as mt

        try:
            with patch("importlib.util.find_spec", return_value=None):
                with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
                    importlib.reload(mt)
            assert any(
                "mcp package not installed" in r.message and "pip install mcp" in r.message
                for r in caplog.records
            ), "import-time fallback should emit a warning with the install hint"
        finally:
            importlib.reload(mt)  # restore real availability for later tests

    def test_mcp_present_no_warning_at_import_time(self, caplog):
        import tools.mcp_tool as mt

        if not mt._MCP_AVAILABLE:
            pytest.skip("mcp not installed in this environment; presence path not exercisable")
        with caplog.at_level(logging.WARNING, logger="tools.mcp_tool"):
            importlib.reload(mt)
        assert not any("mcp package not installed" in r.message for r in caplog.records)
