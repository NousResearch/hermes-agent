"""Contract tests for MCP tool-timeout resolution (#85125 Phase 2g).

Default-behavior invariance (tracker regression policy rule 1): with no
``timeouts:`` section and no per-server key, every resolved value must equal
today's — the resolver only unifies WHERE the value is read, never what it is.
"""

from __future__ import annotations

import pytest

from tools.mcp_tool import _DEFAULT_TOOL_TIMEOUT
from tools.mcp_tool_common import _resolve_tool_timeout


class TestMcpToolTimeoutResolution:
    def test_default_unchanged_with_nothing_configured(self, monkeypatch):
        monkeypatch.setattr("agent.deadline._timeouts_section", lambda: {})
        assert _resolve_tool_timeout({}) == _DEFAULT_TOOL_TIMEOUT == 300

    def test_per_server_timeout_always_wins(self, monkeypatch):
        # Per-server config beats the global timeouts section (documented
        # precedence: most specific wins).
        monkeypatch.setattr(
            "agent.deadline._timeouts_section",
            lambda: {"mcp": {"tool_call": 120}},
        )
        assert _resolve_tool_timeout({"timeout": 45}) == 45

    def test_timeouts_section_beats_default(self, monkeypatch):
        monkeypatch.setattr(
            "agent.deadline._timeouts_section",
            lambda: {"mcp": {"tool_call": 120}},
        )
        assert _resolve_tool_timeout({}) == 120.0

    def test_invalid_timeouts_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(
            "agent.deadline._timeouts_section",
            lambda: {"mcp": {"tool_call": "soon"}},
        )
        assert _resolve_tool_timeout({}) == _DEFAULT_TOOL_TIMEOUT

    def test_resolution_failure_fails_back_to_default(self, monkeypatch):
        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("agent.deadline._timeouts_section", _boom)
        assert _resolve_tool_timeout({}) == _DEFAULT_TOOL_TIMEOUT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
