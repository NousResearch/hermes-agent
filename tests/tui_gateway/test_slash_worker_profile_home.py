"""Tests for TUI gateway slash_worker profile_home propagation (#40677)."""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch, call

import pytest


def test_slash_worker_accepts_profile_home():
    """_SlashWorker.__init__ accepts profile_home parameter."""
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
    }):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = MagicMock()
            mock_popen.return_value.stderr = MagicMock()
            
            from tui_gateway.server import _SlashWorker
            
            # Test initialization with profile_home
            worker = _SlashWorker(
                session_key="test_key",
                model="test-model",
                profile_home="/home/luke/.hermes/profiles/work"
            )
            
            # Verify Popen was called
            assert mock_popen.called
            
            # Check that HERMES_HOME was set in the environment
            call_kwargs = mock_popen.call_args[1]
            assert "env" in call_kwargs
            assert call_kwargs["env"]["HERMES_HOME"] == "/home/luke/.hermes/profiles/work"


def _mock_slash_worker_proc(mock_popen: MagicMock) -> None:
    mock_popen.return_value.stdout = MagicMock()
    mock_popen.return_value.stderr = MagicMock()
    mock_popen.return_value.poll = MagicMock(return_value=None)
    mock_popen.return_value.stdin = MagicMock()


def test_slash_worker_reapplies_home_for_session_profile(tmp_path, monkeypatch):
    """Under home_mode=profile, HOME must follow session HERMES_HOME, not launch."""
    launch = tmp_path / "launch"
    session = tmp_path / "session"
    (launch / "home").mkdir(parents=True)
    (session / "home").mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")

    with patch("subprocess.Popen") as mock_popen:
        _mock_slash_worker_proc(mock_popen)
        from tui_gateway.server import _SlashWorker

        _SlashWorker("test_key", "test-model", profile_home=str(session))

        env = mock_popen.call_args.kwargs["env"]
        assert env["HERMES_HOME"] == str(session)
        assert env["HOME"] == str(session / "home")


def test_slash_worker_home_follows_profile_despite_stale_override(tmp_path, monkeypatch):
    """In-process HERMES_HOME override must not leave HOME on the launch profile."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch = tmp_path / "launch"
    session = tmp_path / "session"
    (launch / "home").mkdir(parents=True)
    (session / "home").mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")

    token = set_hermes_home_override(str(launch))
    try:
        with patch("subprocess.Popen") as mock_popen:
            _mock_slash_worker_proc(mock_popen)
            from tui_gateway.server import _SlashWorker

            _SlashWorker("test_key", "test-model", profile_home=str(session))

            env = mock_popen.call_args.kwargs["env"]
            assert env["HERMES_HOME"] == str(session)
            assert env["HOME"] == str(session / "home")
    finally:
        reset_hermes_home_override(token)
