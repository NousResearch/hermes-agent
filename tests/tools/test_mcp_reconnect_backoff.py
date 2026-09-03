"""Tests for the per-server ``reconnect_backoff`` config option in
``tools/mcp_tool.py``.

Hosted MCP endpoints can fail in multi-minute bursts (e.g. 503 storms from
a shared SaaS backend). With the default 1s ladder base every retry lands
inside the burst, the ladder exhausts in seconds, and the server parks
until the next self-probe. ``reconnect_backoff`` raises the ladder base per
server so the same number of attempts spans the burst.
"""

import logging

from tools.mcp_tool import (
    _DEFAULT_RECONNECT_BACKOFF,
    _MAX_BACKOFF_SECONDS,
    _resolve_reconnect_backoff,
)


class TestResolveReconnectBackoff:
    def test_default_when_absent(self):
        assert _resolve_reconnect_backoff("srv", {}) == _DEFAULT_RECONNECT_BACKOFF

    def test_configured_value(self):
        assert _resolve_reconnect_backoff("srv", {"reconnect_backoff": 45}) == 45.0

    def test_float_value(self):
        assert _resolve_reconnect_backoff("srv", {"reconnect_backoff": 2.5}) == 2.5

    def test_string_number_accepted(self):
        # YAML normally types this, but a quoted value must still work.
        assert _resolve_reconnect_backoff("srv", {"reconnect_backoff": "30"}) == 30.0

    def test_clamped_to_max_backoff(self):
        assert _resolve_reconnect_backoff(
            "srv", {"reconnect_backoff": 9999}
        ) == float(_MAX_BACKOFF_SECONDS)

    def test_clamped_to_floor(self):
        assert _resolve_reconnect_backoff(
            "srv", {"reconnect_backoff": 0}
        ) == _DEFAULT_RECONNECT_BACKOFF
        assert _resolve_reconnect_backoff(
            "srv", {"reconnect_backoff": -5}
        ) == _DEFAULT_RECONNECT_BACKOFF

    def test_invalid_value_falls_back_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _resolve_reconnect_backoff(
                "srv", {"reconnect_backoff": "fast"}
            )
        assert result == _DEFAULT_RECONNECT_BACKOFF
        assert "invalid reconnect_backoff" in caplog.text
        assert "srv" in caplog.text

    def test_none_falls_back_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _resolve_reconnect_backoff(
                "srv", {"reconnect_backoff": None}
            )
        assert result == _DEFAULT_RECONNECT_BACKOFF
        assert "invalid reconnect_backoff" in caplog.text
