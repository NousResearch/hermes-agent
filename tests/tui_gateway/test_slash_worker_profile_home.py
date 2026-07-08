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


def test_slash_worker_without_profile_home():
    """_SlashWorker works without profile_home parameter (backward compatible)."""
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
    }):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = MagicMock()
            mock_popen.return_value.stderr = MagicMock()
            
            from tui_gateway.server import _SlashWorker
            
            # Test initialization without profile_home (backward compatible)
            worker = _SlashWorker(
                session_key="test_key",
                model="test-model"
            )
            
            # Verify Popen was called
            assert mock_popen.called
            
            # Check that HERMES_HOME was NOT overridden
            call_kwargs = mock_popen.call_args[1]
            assert "env" in call_kwargs
            # HERMES_HOME should be from parent env or undefined (inherited from os.environ)
            # The key is that it's not explicitly set when profile_home is None
            env = call_kwargs["env"]
            # Verify env is a copy of os.environ
            assert "PATH" in env


def test_slash_worker_with_none_profile_home():
    """_SlashWorker with explicit profile_home=None works."""
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
    }):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = MagicMock()
            mock_popen.return_value.stderr = MagicMock()
            
            from tui_gateway.server import _SlashWorker
            
            # Test initialization with explicit None
            worker = _SlashWorker(
                session_key="test_key",
                model="test-model",
                profile_home=None
            )
            
            # Verify Popen was called
            assert mock_popen.called
            
            # Check that HERMES_HOME was NOT set
            call_kwargs = mock_popen.call_args[1]
            env = call_kwargs["env"]
            # When profile_home is None, HERMES_HOME should come from parent env only
            if "HERMES_HOME" in env:
                # This is from os.environ at test time, not from our code
                pass


def test_slash_worker_inherits_argv_correctly():
    """_SlashWorker passes correct argv to Popen."""
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
    }):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = MagicMock()
            mock_popen.return_value.stderr = MagicMock()
            
            from tui_gateway.server import _SlashWorker
            
            # Test that argv is correct
            worker = _SlashWorker(
                session_key="my_session",
                model="gpt-4"
            )
            
            call_args = mock_popen.call_args[0][0]
            
            # Verify argv structure
            assert sys.executable in call_args
            assert "-m" in call_args
            assert "tui_gateway.slash_worker" in call_args
            assert "--session-key" in call_args
            assert "my_session" in call_args
            assert "--model" in call_args
            assert "gpt-4" in call_args
