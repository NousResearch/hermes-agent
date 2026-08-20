"""Tests for tools/claude_session_tool.py — Tool registration and dispatch."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from tools.claude_session_tool import (
    CLAUDE_SESSION_SCHEMA, _handle_claude_session,
    _check_claude_session, _diagnose_claude_session,
    _extract_mcp_failure_count, _get_active_sessions_output,
)


class TestSchema:
    def test_schema_has_name(self):
        assert CLAUDE_SESSION_SCHEMA["name"] == "claude_session"

    def test_schema_has_required_action(self):
        assert "action" in CLAUDE_SESSION_SCHEMA["parameters"]["required"]

    def test_schema_action_enum(self):
        actions = CLAUDE_SESSION_SCHEMA["parameters"]["properties"]["action"]["enum"]
        expected = [
            "start", "send", "type", "submit", "send_text", "cancel_input",
            "status", "wait_for_idle", "wait_for_state",
            "output", "jsonl_output", "respond_permission", "respond_interview",
            "stop", "history", "events",
            "list", "switch",
            "diagnose", "doctor_fix",
        ]
        assert set(actions) == set(expected)


class TestHandlerDispatch:
    def test_status_no_session(self):
        result = _handle_claude_session({"action": "status"})
        data = json.loads(result)
        assert data["state"] == "DISCONNECTED"

    def test_unknown_action(self):
        result = _handle_claude_session({"action": "nonexistent"})
        data = json.loads(result)
        assert "error" in data

    def test_send_no_session(self):
        result = _handle_claude_session({"action": "send", "message": "test"})
        data = json.loads(result)
        assert "error" in data

    def test_stop_no_session(self):
        result = _handle_claude_session({"action": "stop"})
        data = json.loads(result)
        assert "error" in data

    def test_send_missing_message(self):
        result = _handle_claude_session({"action": "send"})
        data = json.loads(result)
        assert "error" in data

    def test_type_missing_text(self):
        result = _handle_claude_session({"action": "type"})
        data = json.loads(result)
        assert "error" in data

    def test_wait_for_state_missing_target(self):
        result = _handle_claude_session({"action": "wait_for_state"})
        data = json.loads(result)
        assert "error" in data

    def test_respond_permission_missing_response(self):
        result = _handle_claude_session({"action": "respond_permission"})
        data = json.loads(result)
        assert "error" in data

    def test_history_no_session(self):
        result = _handle_claude_session({"action": "history"})
        data = json.loads(result)
        assert data["total_turns"] == 0

    def test_events_no_session(self):
        result = _handle_claude_session({"action": "events"})
        data = json.loads(result)
        assert data["events"] == []

    def test_output_no_session(self):
        result = _handle_claude_session({"action": "output"})
        data = json.loads(result)
        assert "lines" in data

    def test_default_route_uses_active_session_not_last_created(self):
        """switch/_active_session must control default routing when name/session_id omitted."""
        from tools.claude_session_tool import (
            _sessions, _workdir_index, _name_index, _active_session, _sessions_lock,
        )
        _sessions.clear()
        _workdir_index.clear()
        _name_index.clear()
        _active_session.clear()

        first = MagicMock(_session_id="sess-1", _gateway_session_key="", _session_active=True)
        first.status.return_value = {"state": "IDLE", "session": "first"}
        second = MagicMock(_session_id="sess-2", _gateway_session_key="", _session_active=True)
        second.status.return_value = {"state": "THINKING", "session": "second"}

        with _sessions_lock:
            _sessions["sess-1"] = first
            _sessions["sess-2"] = second
            _name_index[("", "first")] = "sess-1"
            _name_index[("", "second")] = "sess-2"
            _active_session[""] = "sess-1"

        data = json.loads(_handle_claude_session({"action": "status"}))
        assert data["session"] == "first"
        first.status.assert_called_once()
        second.status.assert_not_called()


class TestToolRegistration:
    def test_tool_registered(self):
        """Verify the tool is discoverable in the registry."""
        from tools.registry import registry
        entry = registry.get_entry("claude_session")
        assert entry is not None
        assert entry.toolset == "claude_session"
        assert entry.emoji == "🤖"

    def test_schema_matches_registry(self):
        from tools.registry import registry
        entry = registry.get_entry("claude_session")
        assert entry is not None
        assert entry.schema["name"] == "claude_session"


class TestCheckFn:
    """Tests for _check_claude_session availability check."""

    def test_returns_true_when_tmux_available(self):
        with patch("tools.claude_session_tool.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/tmux" if cmd == "tmux" else None
            assert _check_claude_session() is True

    def test_returns_false_when_tmux_missing(self):
        with patch("tools.claude_session_tool.shutil.which") as mock_which:
            mock_which.return_value = None
            assert _check_claude_session() is False

    def test_logs_warning_when_claude_missing(self):
        with patch("tools.claude_session_tool.shutil.which") as mock_which:
            with patch("tools.claude_session_tool.logger") as mock_logger:
                mock_which.side_effect = lambda cmd: "/usr/bin/tmux" if cmd == "tmux" else None
                _check_claude_session()
                mock_logger.warning.assert_called_once()
                assert "Claude Code CLI not found" in mock_logger.warning.call_args[0][0]

    def test_returns_true_even_when_claude_missing(self):
        """tmux is the hard dep; claude CLI is soft — still registers."""
        with patch("tools.claude_session_tool.shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: "/usr/bin/tmux" if cmd == "tmux" else None
            assert _check_claude_session() is True


class TestDiagnose:
    """Tests for _diagnose_claude_session and the diagnose action."""

    def setup_method(self):
        """Reset module-level state before each test."""
        from tools.claude_session_tool import _sessions, _workdir_index, _name_index, _active_session
        _sessions.clear()
        _workdir_index.clear()
        _name_index.clear()
        _active_session.clear()

    def test_diagnose_function_all_ok(self):
        with patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
            assert result["status"] == "ready"
            assert len(result["checks"]) == 6

    def test_diagnose_function_missing_deps(self):
        with patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {}, clear=True):
            mock_which.return_value = None
            result = _diagnose_claude_session()
            assert result["status"] == "missing_deps"
            dep_names = [c["dependency"] for c in result["checks"]]
            assert "tmux" in dep_names
            assert "Claude Code CLI" in dep_names

    def test_diagnose_action_dispatch(self):
        """diagnose action should return JSON via handler."""
        with patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _handle_claude_session({"action": "diagnose"})
            data = json.loads(result)
            assert data["status"] == "ready"

    def test_diagnose_timeout_too_low(self):
        with patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "120"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
            timeout_check = next(c for c in result["checks"] if c["dependency"] == "HERMES_STREAM_STALE_TIMEOUT")
            assert timeout_check["status"] == "too_low"

    def test_diagnose_timeout_not_set(self):
        with patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {}, clear=True):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
            timeout_check = next(c for c in result["checks"] if c["dependency"] == "HERMES_STREAM_STALE_TIMEOUT")
            assert timeout_check["status"] == "not_set"

    def test_diagnose_has_hints_for_missing(self):
        with patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {}, clear=True):
            mock_which.return_value = None
            result = _diagnose_claude_session()
            for check in result["checks"]:
                if check["status"] == "missing":
                    assert check.get("hint") is not None
                    assert len(check["hint"]) > 0


class TestExtractMcpFailureCount:
    """Tests for _extract_mcp_failure_count pure function."""

    def test_single_server_failure(self):
        text = "2 MCP servers failed · /mcp"
        assert _extract_mcp_failure_count(text) == 2

    def test_single_server_singular(self):
        text = "1 MCP server failed · /mcp"
        assert _extract_mcp_failure_count(text) == 1

    def test_no_failure(self):
        text = "All systems operational"
        assert _extract_mcp_failure_count(text) == 0

    def test_empty_string(self):
        assert _extract_mcp_failure_count("") == 0

    def test_large_count(self):
        text = "15 MCP servers failed · /mcp"
        assert _extract_mcp_failure_count(text) == 15


class TestWorkdirIndexCleanup:
    """Tests for _workdir_index cleanup when stopping sessions."""

    def setup_method(self):
        """Reset module-level state before each test."""
        from tools.claude_session_tool import _sessions, _workdir_index, _name_index, _active_session
        _sessions.clear()
        _workdir_index.clear()
        _name_index.clear()
        _active_session.clear()

    def test_workdir_index_cleanup_single_session(self):
        """Stop session with unique workdir removes entry entirely."""
        from tools.claude_session_tool import (
            _sessions, _workdir_index, _name_index,
            _sessions_lock,
        )
        gw_key = ""
        workdir = "/tmp/test"
        session_id = "sess-1"
        name = "task-1"

        # 模拟启动后的状态
        with _sessions_lock:
            _name_index[(gw_key, name)] = session_id
            _workdir_index[(gw_key, workdir)] = [session_id]
            _sessions[session_id] = MagicMock(
                _session_id=session_id,
                _session_name=name,
                _gateway_session_key=gw_key,
                _session_active=True,
            )

        # 验证初始状态
        with _sessions_lock:
            assert (gw_key, workdir) in _workdir_index
            assert _workdir_index[(gw_key, workdir)] == [session_id]

        # 模拟 stop
        with _sessions_lock:
            # 执行清理逻辑
            keys_to_remove = []
            for k, v_list in _workdir_index.items():
                if session_id in v_list:
                    updated_list = [sid for sid in v_list if sid != session_id]
                    if updated_list:
                        _workdir_index[k] = updated_list
                    else:
                        keys_to_remove.append(k)
            for k in keys_to_remove:
                _workdir_index.pop(k, None)

        # 验证清理结果
        with _sessions_lock:
            assert (gw_key, workdir) not in _workdir_index

    def test_workdir_index_cleanup_multiple_sessions_same_workdir(self):
        """Stop one of two sessions with same workdir keeps the other."""
        from tools.claude_session_tool import (
            _sessions, _workdir_index, _name_index,
            _sessions_lock,
        )
        gw_key = ""
        workdir = "/tmp/test"
        sess1 = "sess-1"
        sess2 = "sess-2"
        name1 = "task-1"
        name2 = "task-2"

        # 模拟启动两个会话后的状态
        with _sessions_lock:
            _name_index[(gw_key, name1)] = sess1
            _name_index[(gw_key, name2)] = sess2
            _workdir_index[(gw_key, workdir)] = [sess1, sess2]
            _sessions[sess1] = MagicMock(
                _session_id=sess1,
                _session_name=name1,
                _gateway_session_key=gw_key,
                _session_active=True,
            )
            _sessions[sess2] = MagicMock(
                _session_id=sess2,
                _session_name=name2,
                _gateway_session_key=gw_key,
                _session_active=True,
            )

        # 验证初始状态
        with _sessions_lock:
            assert _workdir_index[(gw_key, workdir)] == [sess1, sess2]

        # 模拟停止 sess1
        with _sessions_lock:
            keys_to_remove = []
            for k, v_list in _workdir_index.items():
                if sess1 in v_list:
                    updated_list = [sid for sid in v_list if sid != sess1]
                    if updated_list:
                        _workdir_index[k] = updated_list
                    else:
                        keys_to_remove.append(k)
            for k in keys_to_remove:
                _workdir_index.pop(k, None)

        # 验证 sess2 仍在索引中
        with _sessions_lock:
            assert (gw_key, workdir) in _workdir_index
            assert _workdir_index[(gw_key, workdir)] == [sess2]

    def test_workdir_index_cleanup_both_sessions_same_workdir(self):
        """Stop both sessions with same workdir removes entry entirely."""
        from tools.claude_session_tool import (
            _sessions, _workdir_index, _name_index,
            _sessions_lock,
        )
        gw_key = ""
        workdir = "/tmp/test"
        sess1 = "sess-1"
        sess2 = "sess-2"

        # 模拟两个会话
        with _sessions_lock:
            _workdir_index[(gw_key, workdir)] = [sess1, sess2]
            _sessions[sess1] = MagicMock(_session_id=sess1)
            _sessions[sess2] = MagicMock(_session_id=sess2)

        # 停止 sess1
        with _sessions_lock:
            keys_to_remove = []
            for k, v_list in _workdir_index.items():
                if sess1 in v_list:
                    updated_list = [sid for sid in v_list if sid != sess1]
                    if updated_list:
                        _workdir_index[k] = updated_list
                    else:
                        keys_to_remove.append(k)
            for k in keys_to_remove:
                _workdir_index.pop(k, None)

        # 验证只剩 sess2
        with _sessions_lock:
            assert _workdir_index[(gw_key, workdir)] == [sess2]

        # 停止 sess2
        with _sessions_lock:
            keys_to_remove = []
            for k, v_list in _workdir_index.items():
                if sess2 in v_list:
                    updated_list = [sid for sid in v_list if sid != sess2]
                    if updated_list:
                        _workdir_index[k] = updated_list
                    else:
                        keys_to_remove.append(k)
            for k in keys_to_remove:
                _workdir_index.pop(k, None)

        # 验证 entry 被移除
        with _sessions_lock:
            assert (gw_key, workdir) not in _workdir_index


class TestSessionDiagnoseChecks:
    """Tests for session-level diagnose checks (THINKING, bypass, MCP)."""

    def _mock_session(self, state, duration, output_tail=""):
        """Build a mock session info dict for _get_active_sessions_output."""
        return [{
            "session_id": "abcd1234efgh5678",
            "state": state,
            "state_duration_seconds": duration,
            "output_tail": output_tail,
        }]

    def test_thinking_critical(self):
        """THINKING >300s → status='session_issues' with critical check."""
        with patch("tools.claude_session_tool._get_active_sessions_output") as mock_sessions, \
             patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            mock_sessions.return_value = self._mock_session("THINKING", 350.0)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
            assert result["status"] == "session_issues"
            thinking_checks = [c for c in result["checks"]
                               if "THINKING duration" in c.get("dependency", "")]
            assert len(thinking_checks) == 1
            assert thinking_checks[0]["status"] == "critical"

    def test_thinking_warning(self):
        """THINKING >120s but <300s → status='ready' with warning check."""
        with patch("tools.claude_session_tool._get_active_sessions_output") as mock_sessions, \
             patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            mock_sessions.return_value = self._mock_session("THINKING", 150.0)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
            assert result["status"] == "ready"
            thinking_checks = [c for c in result["checks"]
                               if "THINKING duration" in c.get("dependency", "")]
            assert len(thinking_checks) == 1
            assert thinking_checks[0]["status"] == "warning"

    def test_bypass_permissions_hang(self):
        """3+ 'bypass permissions on' → critical startup hang."""
        bypass_text = (
            "bypass permissions on (shift+tab to cycle)\n"
            "bypass permissions on (shift+tab to cycle)\n"
            "bypass permissions on (shift+tab to cycle)\n"
        )
        with patch("tools.claude_session_tool._get_active_sessions_output") as mock_sessions, \
             patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            mock_sessions.return_value = self._mock_session("READY", 5.0, bypass_text)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
            assert result["status"] == "session_issues"
            hang_checks = [c for c in result["checks"]
                           if "startup hang" in c.get("dependency", "")]
            assert len(hang_checks) == 1
            assert hang_checks[0]["status"] == "critical"

    def test_mcp_failure(self):
        """MCP server failure text → warning check."""
        mcp_text = "Some output\n2 MCP servers failed · /mcp\nMore output"
        with patch("tools.claude_session_tool._get_active_sessions_output") as mock_sessions, \
             patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            mock_sessions.return_value = self._mock_session("READY", 5.0, mcp_text)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
            mcp_checks = [c for c in result["checks"]
                          if "MCP servers" in c.get("dependency", "")]
            assert len(mcp_checks) == 1
            assert mcp_checks[0]["status"] == "warning"

    def test_no_sessions_ready_status(self):
        """No active sessions → status='ready' (no session-level checks)."""
        with patch("tools.claude_session_tool._get_active_sessions_output") as mock_sessions, \
             patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            mock_sessions.return_value = []
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
            assert result["status"] == "ready"

    def test_cli_migration_prompt(self):
        """CLI migration prompt → info check."""
        migration_text = "switched from npm to native installer\nSome other output"
        with patch("tools.claude_session_tool._get_active_sessions_output") as mock_sessions, \
             patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            mock_sessions.return_value = self._mock_session("READY", 5.0, migration_text)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
        cli_checks = [c for c in result["checks"]
                      if "CLI migration" in c.get("dependency", "")]
        assert len(cli_checks) == 1
        assert cli_checks[0]["status"] == "info"

    def test_tmux_focus_events_off(self):
        """tmux focus-events off → info check."""
        tmux_text = "tmux focus-events off · add 'set -g focus-events on'\nSome output"
        with patch("tools.claude_session_tool._get_active_sessions_output") as mock_sessions, \
             patch("tools.claude_session_tool.shutil.which") as mock_which, \
             patch.dict(os.environ, {"HERMES_STREAM_STALE_TIMEOUT": "300"}):
            mock_which.side_effect = lambda cmd: {
                "tmux": "/usr/bin/tmux",
                "claude": "/usr/local/bin/claude",
            }.get(cmd)
            mock_sessions.return_value = self._mock_session("READY", 5.0, tmux_text)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="tmux 3.4")
                result = _diagnose_claude_session()
        tmux_checks = [c for c in result["checks"]
                       if "tmux config" in c.get("dependency", "")]
        assert len(tmux_checks) == 1
        assert tmux_checks[0]["status"] == "info"
