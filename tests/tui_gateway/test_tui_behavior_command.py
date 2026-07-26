"""Tests for the /behavior TUI/Desktop command (tui_gateway/server.py).

Mirrors tests/cli/test_cli_behavior_command.py and
tests/gateway/test_gateway_behavior_command.py: stubs BehavioralAnalyzer,
verifies --days / --source / positional arg parsing, config gate, error
handling, and that user_id=None is passed (TUI is single-user).

Covers:
  - behavior.get RPC method returns correct structure
  - Config gate (disabled → message, enabled → runs analyzer)
  - Argument parsing (--days, --source, positional) via _live_slash_command_output
  - _live_slash_command_output routes behavior commands correctly
  - Error handling (analyzer exception → error message, no crash)
  - Cross-user data safety: user_id=None is always passed
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import tui_gateway.server as srv


# =========================================================================
# Stubs and helpers
# =========================================================================


class _BehavioralAnalyzerStub:
    calls = []

    def __init__(self, db, config=None):
        self.db = db
        self.config = config

    def generate(self, *, days=30, source=None, user_id=None):
        self.calls.append({"days": days, "source": source, "user_id": user_id})
        return {
            "days": days,
            "source": source,
            "empty": False,
            "scores": {},
            "cards": {},
            "session_count": 1,
            "llm_available": False,
        }

    def format_terminal(self, report):
        return f"days={report['days']} source={report['source']}"


def _call_behavior_get(params: dict | None = None, config=None, db=None):
    """Invoke the behavior.get RPC method with stubs.

    Returns the full JSON-RPC envelope dict.
    Patches:
      - read_raw_config → config (default: {"behavior": {"enabled": True}})
      - _get_db → db (default: MagicMock)
      - BehavioralAnalyzer → stub
    """
    params = params or {}
    cfg = config if config is not None else {"behavior": {"enabled": True}}
    _BehavioralAnalyzerStub.calls = []

    with patch("hermes_cli.config.read_raw_config", return_value=cfg), \
         patch.object(srv, "_get_db", return_value=db if db is not None else MagicMock()), \
         patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
        return srv._methods["behavior.get"](1, params)


# =========================================================================
# Config gate
# =========================================================================


class TestBehaviorConfigGate:
    def test_disabled_returns_message(self):
        """When behavior.enabled is False, return a disabled message."""
        resp = _call_behavior_get(config={"behavior": {"enabled": False}})
        assert "error" not in resp
        output = resp["result"]["output"]
        assert "disabled" in output.lower()
        assert "behavior.enabled: true" in output
        # Analyzer should NOT have been called
        assert _BehavioralAnalyzerStub.calls == []

    def test_no_behavior_section_disabled(self):
        """No behavior section in config → disabled."""
        resp = _call_behavior_get(config={})
        output = resp["result"]["output"]
        assert "disabled" in output.lower()

    def test_no_enabled_key_disabled(self):
        """behavior section exists but no enabled key → disabled."""
        resp = _call_behavior_get(config={"behavior": {}})
        output = resp["result"]["output"]
        assert "disabled" in output.lower()

    def test_enabled_runs_analyzer(self):
        """When enabled, the analyzer runs and output is returned."""
        resp = _call_behavior_get(config={"behavior": {"enabled": True}})
        assert "error" not in resp
        assert len(_BehavioralAnalyzerStub.calls) == 1
        assert "days=30" in resp["result"]["output"]


# =========================================================================
# behavior.get RPC method structure
# =========================================================================


class TestBehaviorGetRpc:
    def test_default_days_30(self):
        resp = _call_behavior_get()
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        assert "output" in resp["result"]
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 30, "source": None, "user_id": None}
        ]

    def test_days_param(self):
        resp = _call_behavior_get({"days": 14})
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 14, "source": None, "user_id": None}
        ]
        assert "days=14" in resp["result"]["output"]

    def test_source_param(self):
        resp = _call_behavior_get({"source": "discord"})
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 30, "source": "discord", "user_id": None}
        ]

    def test_days_and_source(self):
        resp = _call_behavior_get({"days": 7, "source": "cli"})
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 7, "source": "cli", "user_id": None}
        ]

    def test_db_unavailable_returns_error(self):
        """When _get_db() returns None, return a db-unavailable error."""
        resp = _call_behavior_get(db=None, config={"behavior": {"enabled": True}})
        # Override: patch _get_db to return None explicitly
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=None), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            resp = srv._methods["behavior.get"](1, {})
        assert "error" in resp
        assert resp["error"]["code"] == 5029
        assert "state.db" in resp["error"]["message"]


# =========================================================================
# Error handling
# =========================================================================


class TestBehaviorErrors:
    def test_analyzer_exception_returns_error(self):
        """If BehavioralAnalyzer raises, return error envelope (no crash)."""

        class _ExplodingStub:
            def __init__(self, db, config=None):
                pass

            def generate(self, *, days=30, source=None, user_id=None):
                raise RuntimeError("boom")

            def format_terminal(self, report):
                return "should not reach"

        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _ExplodingStub):
            resp = srv._methods["behavior.get"](1, {})

        assert "error" in resp
        assert resp["error"]["code"] == 5029
        assert "boom" in resp["error"]["message"]


# =========================================================================
# Config passed to analyzer
# =========================================================================


class TestBehaviorConfigPassed:
    def test_config_dict_passed_to_analyzer(self):
        """The behavior config dict is passed to BehavioralAnalyzer."""
        captured_config = []

        class _CapturingStub:
            def __init__(self, db, config=None):
                captured_config.append(config)

            def generate(self, *, days=30, source=None, user_id=None):
                return {
                    "days": days,
                    "source": source,
                    "empty": False,
                    "scores": {},
                    "cards": {},
                    "session_count": 0,
                    "llm_available": False,
                }

            def format_terminal(self, report):
                return "ok"

        cfg = {"behavior": {"enabled": True, "model": "gpt-4o-mini"}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _CapturingStub):
            srv._methods["behavior.get"](1, {})

        assert len(captured_config) == 1
        assert captured_config[0].get("model") == "gpt-4o-mini"


# =========================================================================
# Cross-user data safety: user_id=None
# =========================================================================


class TestBehaviorUserIdSafety:
    """TUI is single-user (like CLI). user_id must always be None."""

    def test_user_id_is_none_default(self):
        _call_behavior_get()
        assert _BehavioralAnalyzerStub.calls[0]["user_id"] is None

    def test_user_id_is_none_with_days(self):
        _call_behavior_get({"days": 14})
        assert _BehavioralAnalyzerStub.calls[0]["user_id"] is None

    def test_user_id_is_none_with_source(self):
        _call_behavior_get({"source": "discord"})
        assert _BehavioralAnalyzerStub.calls[0]["user_id"] is None

    def test_no_user_id_param_accepted(self):
        """The method must NOT accept a user_id param (single-user safety)."""
        # Even if a user_id is passed in params, it should be ignored —
        # the method always passes user_id=None to generate().
        _call_behavior_get({"user_id": "attacker_user_123"})
        assert _BehavioralAnalyzerStub.calls[0]["user_id"] is None


# =========================================================================
# _live_slash_command_output routing
# =========================================================================


class TestLiveSlashCommandOutput:
    """Test that _live_slash_command_output routes /behavior correctly."""

    def test_behavior_in_direct_commands(self):
        """behavior must be in _LIVE_SESSION_DIRECT_COMMANDS."""
        assert "behavior" in srv._LIVE_SESSION_DIRECT_COMMANDS

    def test_default_days_30(self):
        """/behavior with no args → days=30."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            _BehavioralAnalyzerStub.calls = []
            result = srv._live_slash_command_output("sid", {}, "behavior", "")
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 30, "source": None, "user_id": None}
        ]
        assert "days=30" in result

    def test_positional_days(self):
        """/behavior 7 → days=7."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            _BehavioralAnalyzerStub.calls = []
            result = srv._live_slash_command_output("sid", {}, "behavior", "7")
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 7, "source": None, "user_id": None}
        ]

    def test_days_flag(self):
        """/behavior --days 14 → days=14."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            _BehavioralAnalyzerStub.calls = []
            result = srv._live_slash_command_output("sid", {}, "behavior", "--days 14")
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 14, "source": None, "user_id": None}
        ]

    def test_source_flag(self):
        """/behavior --source discord → source=discord."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            _BehavioralAnalyzerStub.calls = []
            result = srv._live_slash_command_output("sid", {}, "behavior", "--source discord")
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 30, "source": "discord", "user_id": None}
        ]

    def test_days_and_source_flags(self):
        """/behavior --days 14 --source discord → both parsed."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            _BehavioralAnalyzerStub.calls = []
            result = srv._live_slash_command_output("sid", {}, "behavior", "--days 14 --source discord")
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 14, "source": "discord", "user_id": None}
        ]

    def test_invalid_days_value(self):
        """/behavior --days abc → returns error message, analyzer not called."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            _BehavioralAnalyzerStub.calls = []
            result = srv._live_slash_command_output("sid", {}, "behavior", "--days abc")
        assert _BehavioralAnalyzerStub.calls == []
        assert "Invalid" in result

    def test_disabled_config_returns_message(self):
        """/behavior when disabled → returns disabled message."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": False}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            _BehavioralAnalyzerStub.calls = []
            result = srv._live_slash_command_output("sid", {}, "behavior", "")
        assert _BehavioralAnalyzerStub.calls == []
        assert "disabled" in result.lower()

    def test_positional_and_source_combined(self):
        """/behavior 7 --source cli → days=7, source=cli."""
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": True}}), \
             patch.object(srv, "_get_db", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
            _BehavioralAnalyzerStub.calls = []
            result = srv._live_slash_command_output("sid", {}, "behavior", "7 --source cli")
        assert _BehavioralAnalyzerStub.calls == [
            {"days": 7, "source": "cli", "user_id": None}
        ]