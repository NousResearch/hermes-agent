"""Tests for tools/claude_session/tmux_interface.py"""

import pytest
from unittest.mock import patch, MagicMock
from tools.claude_session.tmux_interface import TmuxInterface


@pytest.fixture
def tmux():
    return TmuxInterface(session_name="test-claude")


class TestTmuxInterface:
    def test_session_exists_check(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert tmux.session_exists() is True

    def test_session_not_exists(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert tmux.session_exists() is False

    def test_create_session(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            result = tmux.create_session(workdir="/tmp/test")
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "new-session" in args
            assert "-d" in args

    def test_capture_pane(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="line1\nline2\n❯ "
            )
            output = tmux.capture_pane()
            assert "line1" in output
            assert "❯" in output

    def test_send_keys_uses_literal_flag(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            tmux.send_keys("hello world")
            args = mock_run.call_args[0][0]
            assert "send-keys" in args
            assert "-l" in args
            assert "hello world" in args

    def test_send_keys_with_enter_sends_separately(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            tmux.send_keys("hello", enter=True)
            # Should have two calls: one for text (-l), one for Enter
            assert mock_run.call_count == 2
            first_args = mock_run.call_args_list[0][0][0]
            second_args = mock_run.call_args_list[1][0][0]
            assert "-l" in first_args
            assert "Enter" in second_args

    def test_send_special_key(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            tmux.send_special_key("C-c")
            args = mock_run.call_args[0][0]
            assert "C-c" in args

    def test_kill_session(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            tmux.kill_session()
            args = mock_run.call_args[0][0]
            assert "kill-session" in args

    def test_send_keys_no_shell_escaping_needed(self, tmux):
        """With -l flag and subprocess.run list args, special chars are safe."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            tmux.send_keys("it's $HOME `whoami`")
            args = mock_run.call_args[0][0]
            # Text should pass through unmodified
            assert "it's $HOME `whoami`" in args

    def test_create_session_with_env(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            tmux.create_session(workdir="/tmp", env={"FOO": "bar"})
            args = mock_run.call_args[0][0]
            assert "-e" in args
            assert "FOO=bar" in args

    def test_session_not_found_error(self, tmux):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("tmux not found")
            with pytest.raises(RuntimeError, match="not installed"):
                tmux.session_exists()
