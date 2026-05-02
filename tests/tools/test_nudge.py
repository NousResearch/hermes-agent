"""Tests for the nudge tool."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock, TestCase

sys.path.insert(0, '/workspace/hermes-agent')

from tools.nudge_tool import (
    _error,
    _get_origin_session,
    _nudge_file_path,
    _read_nudges,
    _success,
    _write_nudges,
    nudge_tool,
)


class TestHelpers(TestCase):
    """Test helper functions."""

    def test_error(self):
        """Test _error returns correct dict."""
        result = _error("test message")
        self.assertEqual(result, {"error": "test message"})

    def test_success(self):
        """Test _success returns correct dict."""
        result = _success("test message")
        self.assertEqual(result, {"success": True, "message": "test message"})

    def test_success_with_data(self):
        """Test _success with additional data."""
        result = _success("test message", {"key": "value"})
        self.assertEqual(result, {"success": True, "message": "test message", "key": "value"})


class TestNudgeFileOperations(TestCase):
    """Test file operations with temp directory."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.patcher = mock.patch("tools.nudge_tool._NUDGE_DIR", self.temp_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_nudge_file_path(self):
        """Test nudge file path generation."""
        path = _nudge_file_path("discord:123456:thread_789")
        expected = self.temp_path / "discord-123456-thread_789.json"
        self.assertEqual(path, expected)

    def test_read_nudges_empty(self):
        """Test reading nudges when file doesn't exist."""
        nudges = _read_nudges("discord:123456")
        self.assertEqual(nudges, [])

    def test_write_and_read_nudges(self):
        """Test writing and reading nudges."""
        test_nudges = [{"content": "test", "timestamp": 12345.0}]
        self.assertTrue(_write_nudges("discord:123456", test_nudges))

        read = _read_nudges("discord:123456")
        self.assertEqual(read, test_nudges)

    def test_write_nudges_failure(self):
        """Test write failure handling."""
        import builtins
        original_open = builtins.open
        def mock_open(*args, **kwargs):
            raise PermissionError("denied")
        
        try:
            builtins.open = mock_open
            result = _write_nudges("discord:123456", [])
            self.assertFalse(result)
        finally:
            builtins.open = original_open

    def test_read_nudges_corrupted(self):
        """Test reading corrupted nudge file."""
        path = _nudge_file_path("discord:123456")
        path.write_text("invalid json")
        nudges = _read_nudges("discord:123456")
        self.assertEqual(nudges, [])


class TestGetOriginSession(TestCase):
    """Test origin session resolution."""

    def test_get_origin_from_env_vars(self):
        """Test resolving origin from environment variables."""
        env_vars = {
            "HERMES_CRON_AUTO_DELIVER_PLATFORM": "discord",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "987654",
            "HERMES_CRON_AUTO_DELIVER_THREAD_ID": "321",
        }
        with mock.patch.dict("os.environ", env_vars, clear=True):
            result = _get_origin_session()
            self.assertEqual(result, "discord:987654:321")

    def test_get_origin_from_env_vars_no_thread(self):
        """Test resolving origin without thread_id."""
        env_vars = {
            "HERMES_CRON_AUTO_DELIVER_PLATFORM": "telegram",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "-100123456",
        }
        with mock.patch.dict("os.environ", env_vars, clear=True):
            result = _get_origin_session()
            self.assertEqual(result, "telegram:-100123456")

    def test_get_origin_no_context(self):
        """Test resolving origin when no context available."""
        with mock.patch.dict("os.environ", {}, clear=True):
            result = _get_origin_session()
            self.assertIsNone(result)


class TestNudgeSend(TestCase):
    """Test nudge send action."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.patcher = mock.patch("tools.nudge_tool._NUDGE_DIR", self.temp_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_send_requires_content(self):
        """Test send action requires content."""
        result = nudge_tool(action="send", target="discord:123456")
        self.assertIn("error", result)
        self.assertIn("content", result["error"].lower())

    def test_send_no_origin_session(self):
        """Test send with origin target but no origin context."""
        with mock.patch.dict("os.environ", {}, clear=True):
            result = nudge_tool(action="send", target="origin", content="test")
            self.assertIn("error", result)
            self.assertIn("no origin session", result["error"].lower())

    def test_send_explicit_target(self):
        """Test send with explicit target."""
        result = nudge_tool(action="send", target="discord:123456", content="Hello!")
        self.assertTrue(result["success"])
        self.assertEqual(result["target"], "discord:123456")
        self.assertEqual(result["nudge_count"], 1)

        # Verify file was written
        nudges = _read_nudges("discord:123456")
        self.assertEqual(len(nudges), 1)
        self.assertEqual(nudges[0]["content"], "Hello!")
        self.assertIn("timestamp", nudges[0])

    def test_send_multiple_nudges(self):
        """Test sending multiple nudges to same session."""
        nudge_tool(action="send", target="discord:123456", content="First")
        nudge_tool(action="send", target="discord:123456", content="Second")

        nudges = _read_nudges("discord:123456")
        self.assertEqual(len(nudges), 2)
        self.assertEqual(nudges[0]["content"], "First")
        self.assertEqual(nudges[1]["content"], "Second")

    def test_send_with_origin_resolution(self):
        """Test send with origin target resolves correctly."""
        env_vars = {
            "HERMES_CRON_AUTO_DELIVER_PLATFORM": "telegram",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "-100123",
        }
        with mock.patch.dict("os.environ", env_vars, clear=True):
            result = nudge_tool(action="send", target="origin", content="From cronjob")
            self.assertTrue(result["success"])
            self.assertEqual(result["target"], "telegram:-100123")


class TestNudgeList(TestCase):
    """Test nudge list action."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.patcher = mock.patch("tools.nudge_tool._NUDGE_DIR", self.temp_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_list_empty(self):
        """Test listing when no nudges exist."""
        result = nudge_tool(action="list")
        self.assertTrue(result["success"])
        self.assertEqual(result["nudges"], {})

    def test_list_all(self):
        """Test listing all nudges."""
        nudge_tool(action="send", target="discord:123", content="First")
        nudge_tool(action="send", target="telegram:456", content="Second")
        nudge_tool(action="send", target="discord:123", content="Third")

        result = nudge_tool(action="list")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["nudges"]), 2)  # Two unique sessions

    def test_list_specific_target(self):
        """Test listing nudges for specific target."""
        nudge_tool(action="send", target="discord:123", content="For 123")
        nudge_tool(action="send", target="discord:456", content="For 456")

        result = nudge_tool(action="list", target="discord:123")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["nudges"]), 1)
        self.assertEqual(result["nudges"][0]["content"], "For 123")

    def test_list_no_nudges_for_target(self):
        """Test listing nudges for target with none."""
        result = nudge_tool(action="list", target="discord:nonexistent")
        self.assertTrue(result["success"])
        self.assertEqual(result["nudges"], [])


class TestNudgeClear(TestCase):
    """Test nudge clear action."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.patcher = mock.patch("tools.nudge_tool._NUDGE_DIR", self.temp_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_clear_existing(self):
        """Test clearing existing nudges."""
        nudge_tool(action="send", target="discord:123", content="Test")

        result = nudge_tool(action="clear", target="discord:123")
        self.assertTrue(result["success"])
        self.assertIn("cleared", result["message"].lower())

        # Verify cleared
        self.assertEqual(_read_nudges("discord:123"), [])

    def test_clear_nonexistent(self):
        """Test clearing when no nudges exist."""
        result = nudge_tool(action="clear", target="discord:nonexistent")
        self.assertTrue(result["success"])
        self.assertIn("no nudges", result["message"].lower())

    def test_clear_origin_resolution(self):
        """Test clear with origin target."""
        env_vars = {
            "HERMES_CRON_AUTO_DELIVER_PLATFORM": "discord",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "123",
        }
        with mock.patch.dict("os.environ", env_vars, clear=True):
            nudge_tool(action="send", target="origin", content="Test")
            result = nudge_tool(action="clear", target="origin")
            self.assertTrue(result["success"])


class TestNudgeValidation(TestCase):
    """Test input validation."""

    def test_invalid_action(self):
        """Test invalid action handling."""
        result = nudge_tool(action="invalid_action")
        self.assertIn("error", result)
        self.assertIn("unknown action", result["error"].lower())

    def test_case_insensitive_action(self):
        """Test action is case insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("tools.nudge_tool._NUDGE_DIR", Path(tmpdir)):
                result = nudge_tool(action="SEND", target="discord:123", content="Test")
                self.assertTrue(result["success"])

    def test_whitespace_action(self):
        """Test action handles whitespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("tools.nudge_tool._NUDGE_DIR", Path(tmpdir)):
                result = nudge_tool(action="  send  ", target="discord:123", content="Test")
                self.assertTrue(result["success"])

    def test_non_string_action(self):
        """Test non-string action handling."""
        result = nudge_tool(action=123)
        self.assertIn("error", result)


class TestNudgeFileSanitization(TestCase):
    """Test filename sanitization for various session keys."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.patcher = mock.patch("tools.nudge_tool._NUDGE_DIR", self.temp_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_sanitization_colon(self):
        """Test colons are replaced."""
        path = _nudge_file_path("discord:123:456")
        self.assertNotIn(":", path.name)
        self.assertIn("-", path.name)

    def test_sanitization_slash(self):
        """Test slashes are replaced."""
        path = _nudge_file_path("discord/123")
        self.assertNotIn("/", path.name)
        self.assertIn("_", path.name)

    def test_sanitization_backslash(self):
        """Test backslashes are replaced."""
        path = _nudge_file_path("discord\\123")
        self.assertNotIn("\\", path.name)
        self.assertIn("_", path.name)

    def test_complex_session_key(self):
        """Test complex session key handling."""
        # Telegram group with thread
        key = "telegram:-1001234567890:17585"
        path = _nudge_file_path(key)
        self.assertEqual(path.parent, self.temp_path)
        self.assertEqual(path.suffix, ".json")
        # Should be writable
        self.assertTrue(_write_nudges(key, [{"content": "test"}]))
        self.assertEqual(_read_nudges(key), [{"content": "test"}])


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
