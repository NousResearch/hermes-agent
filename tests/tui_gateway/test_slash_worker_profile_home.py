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


