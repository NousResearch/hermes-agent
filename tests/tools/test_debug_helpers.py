"""Tests for tools/debug_helpers.py — DebugSession class."""

import json
import os
from unittest.mock import patch

from tools.debug_helpers import DebugSession


class TestDebugSessionDisabled:
    """When the env var is not set, DebugSession should be a cheap no-op."""

    def test_not_active_by_default(self):
        ds = DebugSession("test_tool", env_var="FAKE_DEBUG_VAR_XYZ")
        assert ds.active is False
        assert ds.enabled is False

    def test_session_id_empty_when_disabled(self):
        ds = DebugSession("test_tool", env_var="FAKE_DEBUG_VAR_XYZ")
        assert ds.session_id == ""

    def test_log_call_noop(self):
        ds = DebugSession("test_tool", env_var="FAKE_DEBUG_VAR_XYZ")
        ds.log_call("search", {"query": "hello"})
        assert ds._calls == []

    def test_save_noop(self, tmp_path):
        ds = DebugSession("test_tool", env_var="FAKE_DEBUG_VAR_XYZ")
        ds.log_dir = tmp_path
        ds.save()
        assert list(tmp_path.iterdir()) == []

    def test_get_session_info_disabled(self):
        ds = DebugSession("test_tool", env_var="FAKE_DEBUG_VAR_XYZ")
        info = ds.get_session_info()
        assert info["enabled"] is False
        assert info["session_id"] is None
        assert info["log_path"] is None
        assert info["total_calls"] == 0


class TestDebugSessionEnabled:
    """When the env var is set to 'true', DebugSession records and saves."""

    def _make_enabled(self, tmp_path):
        with patch.dict(os.environ, {"TEST_DEBUG": "true"}):
            ds = DebugSession("test_tool", env_var="TEST_DEBUG")
        ds.log_dir = tmp_path
        return ds

    def test_active_when_env_set(self, tmp_path):
        ds = self._make_enabled(tmp_path)
        assert ds.active is True
        assert ds.enabled is True

    def test_session_id_generated(self, tmp_path):
        ds = self._make_enabled(tmp_path)
        assert len(ds.session_id) > 0

    def test_log_call_appends(self, tmp_path):
        ds = self._make_enabled(tmp_path)
        ds.log_call("search", {"query": "hello"})
        ds.log_call("extract", {"url": "http://x.com"})
        assert len(ds._calls) == 2
        assert ds._calls[0]["tool_name"] == "search"
        assert ds._calls[0]["query"] == "hello"
        assert "timestamp" in ds._calls[0]

    def test_save_creates_json_file(self, tmp_path):
        ds = self._make_enabled(tmp_path)
        ds.log_call("search", {"query": "test"})
        ds.save()

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        assert "test_tool_debug_" in files[0].name

        data = json.loads(files[0].read_text())
        assert data["session_id"] == ds.session_id
        assert data["debug_enabled"] is True
        assert data["total_calls"] == 1
        assert data["tool_calls"][0]["tool_name"] == "search"

    def test_get_session_info_enabled(self, tmp_path):
        ds = self._make_enabled(tmp_path)
        ds.log_call("a", {})
        ds.log_call("b", {})
        info = ds.get_session_info()
        assert info["enabled"] is True
        assert info["session_id"] == ds.session_id
        assert info["total_calls"] == 2
        assert "test_tool_debug_" in info["log_path"]

    def test_env_var_case_insensitive(self, tmp_path):
        with patch.dict(os.environ, {"TEST_DEBUG": "True"}):
            ds = DebugSession("t", env_var="TEST_DEBUG")
        assert ds.enabled is True

        with patch.dict(os.environ, {"TEST_DEBUG": "TRUE"}):
            ds = DebugSession("t", env_var="TEST_DEBUG")
        assert ds.enabled is True

    def test_env_var_false_disables(self):
        with patch.dict(os.environ, {"TEST_DEBUG": "false"}):
            ds = DebugSession("t", env_var="TEST_DEBUG")
        assert ds.enabled is False

    def test_save_empty_log(self, tmp_path):
        ds = self._make_enabled(tmp_path)
        ds.save()
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["total_calls"] == 0
        assert data["tool_calls"] == []
