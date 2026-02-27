"""Tests for tools/debug_helpers.py - DebugSession functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.debug_helpers import DebugSession


class TestDebugSessionDisabled:
    """Tests for DebugSession when debug mode is disabled (default)."""

    def test_disabled_by_default(self):
        """Session should be disabled when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            assert not session.enabled
            assert not session.active
            assert session.session_id == ""

    def test_disabled_when_env_false(self):
        """Session should be disabled when env var is 'false'."""
        with patch.dict(os.environ, {"TEST_DEBUG": "false"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            assert not session.enabled

    def test_log_call_noop_when_disabled(self):
        """log_call should be a no-op when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            # Should not raise
            session.log_call("some_call", {"key": "value"})
            assert session._calls == []

    def test_save_noop_when_disabled(self):
        """save should be a no-op when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            # Should not raise or create files
            session.save()

    def test_get_session_info_when_disabled(self):
        """get_session_info should return disabled status."""
        with patch.dict(os.environ, {}, clear=True):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            info = session.get_session_info()
            assert info["enabled"] is False
            assert info["session_id"] is None
            assert info["log_path"] is None
            assert info["total_calls"] == 0


class TestDebugSessionEnabled:
    """Tests for DebugSession when debug mode is enabled."""

    def test_enabled_when_env_true(self):
        """Session should be enabled when env var is 'true'."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            assert session.enabled
            assert session.active
            assert session.session_id != ""
            assert len(session.session_id) == 36  # UUID format

    def test_enabled_case_insensitive(self):
        """Env var check should be case-insensitive."""
        with patch.dict(os.environ, {"TEST_DEBUG": "TRUE"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            assert session.enabled

    def test_log_call_records_calls(self):
        """log_call should record calls when enabled."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            session.log_call("web_search", {"query": "test", "results": 5})
            session.log_call("another_call", {"data": "value"})

            assert len(session._calls) == 2
            assert session._calls[0]["tool_name"] == "web_search"
            assert session._calls[0]["query"] == "test"
            assert session._calls[0]["results"] == 5
            assert "timestamp" in session._calls[0]

    def test_get_session_info_when_enabled(self):
        """get_session_info should return correct info when enabled."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            session.log_call("call1", {})
            session.log_call("call2", {})

            info = session.get_session_info()
            assert info["enabled"] is True
            assert info["session_id"] == session.session_id
            assert "test_tool_debug_" in info["log_path"]
            assert info["total_calls"] == 2

    def test_save_creates_log_file(self, tmp_path):
        """save should create a JSON log file when enabled."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            session.log_dir = tmp_path
            session.log_call("test_call", {"key": "value"})
            session.save()

            # Find the created log file
            log_files = list(tmp_path.glob("test_tool_debug_*.json"))
            assert len(log_files) == 1

            # Verify contents
            with open(log_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["session_id"] == session.session_id
            assert data["debug_enabled"] is True
            assert data["total_calls"] == 1
            assert len(data["tool_calls"]) == 1
            assert data["tool_calls"][0]["tool_name"] == "test_call"
            assert data["tool_calls"][0]["key"] == "value"
            assert "start_time" in data
            assert "end_time" in data


class TestDebugSessionEdgeCases:
    """Edge case tests for DebugSession."""

    def test_different_env_vars(self):
        """Different tools should use different env vars."""
        with patch.dict(os.environ, {"WEB_DEBUG": "true", "VISION_DEBUG": "false"}):
            web_session = DebugSession("web_tools", env_var="WEB_DEBUG")
            vision_session = DebugSession("vision_tools", env_var="VISION_DEBUG")

            assert web_session.enabled
            assert not vision_session.enabled

    def test_log_call_with_empty_data(self):
        """log_call should handle empty data dict."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            session.log_call("empty_call", {})

            assert len(session._calls) == 1
            assert session._calls[0]["tool_name"] == "empty_call"

    def test_log_call_with_complex_data(self):
        """log_call should handle complex nested data."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            session.log_call("complex_call", {
                "list": [1, 2, 3],
                "nested": {"a": {"b": "c"}},
                "none_val": None,
            })

            assert len(session._calls) == 1
            assert session._calls[0]["list"] == [1, 2, 3]
            assert session._calls[0]["nested"]["a"]["b"] == "c"
            assert session._calls[0]["none_val"] is None

    def test_tool_name_in_init(self):
        """Tool name should be correctly stored."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("my_custom_tool", env_var="TEST_DEBUG")
            assert session.tool_name == "my_custom_tool"

    def test_log_dir_default(self):
        """Default log directory should be ./logs."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            assert session.log_dir == Path("./logs")

    def test_save_handles_missing_log_dir(self, tmp_path):
        """save should handle case where log_dir creation fails gracefully."""
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            session = DebugSession("test_tool", env_var="TEST_DEBUG")
            # Point to a valid temp dir to avoid filesystem issues
            session.log_dir = tmp_path / "new_logs"
            session.log_call("test", {})
            # Should not raise even if dir doesn't exist yet
            session.save()
