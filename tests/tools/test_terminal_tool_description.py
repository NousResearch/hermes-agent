"""Regression test for the terminal tool description wording (issue #7672).

The description used to claim "Execute shell commands on a Linux environment",
which is misleading for users running locally on Windows/macOS or in a non-Linux
configured backend. Pin the invariant: the description must describe the
environment as *configured* rather than hardcoding Linux.
"""

from __future__ import annotations

from tools.terminal_tool import TERMINAL_TOOL_DESCRIPTION


def test_terminal_description_does_not_hardcode_linux():
    # The core of issue #7672: don't assert every user is on Linux.
    assert "Linux environment" not in TERMINAL_TOOL_DESCRIPTION


def test_terminal_description_mentions_configured_environment():
    # It should frame the environment as setup-dependent, naming the real options.
    assert "configured execution environment" in TERMINAL_TOOL_DESCRIPTION
    assert "local machine" in TERMINAL_TOOL_DESCRIPTION
    assert "Docker container" in TERMINAL_TOOL_DESCRIPTION


def test_terminal_description_keeps_persistence_contract():
    # The persistence guidance the rest of the prompt relies on must survive.
    assert "persist between calls" in TERMINAL_TOOL_DESCRIPTION
