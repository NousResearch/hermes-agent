"""Tests for tools.browser_tool.warm_agent_browser_npx_cache (#43564).

agent-browser was moved out of root package.json dependencies and now
resolves lazily via `npx agent-browser`. warm_agent_browser_npx_cache() is
the fire-and-forget helper `hermes update` / `hermes doctor --fix` call to
pre-fetch it so the first real browser-tool invocation in a session doesn't
pay npx's registry-lookup cost. It must never raise and must accurately
report success/failure via its return value — callers (main.py, doctor.py)
make decisions based on that return value.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from tools.browser_tool import warm_agent_browser_npx_cache


def test_returns_false_without_calling_subprocess_when_npx_missing():
    with patch("shutil.which", return_value=None), patch(
        "subprocess.run"
    ) as mock_run:
        assert warm_agent_browser_npx_cache() is False
    mock_run.assert_not_called()


def test_invokes_npx_with_prefer_offline_version_check():
    with patch("shutil.which", return_value="/usr/bin/npx"), patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout="1.2.3\n", stderr=""),
    ) as mock_run:
        assert warm_agent_browser_npx_cache() is True

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == ["/usr/bin/npx", "--prefer-offline", "-y", "agent-browser", "--version"]
    assert kwargs.get("check") is False


def test_custom_timeout_is_forwarded_to_subprocess_run():
    with patch("shutil.which", return_value="/usr/bin/npx"), patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    ) as mock_run:
        warm_agent_browser_npx_cache(timeout=5.0)

    assert mock_run.call_args.kwargs.get("timeout") == 5.0


def test_returns_false_on_nonzero_exit():
    with patch("shutil.which", return_value="/usr/bin/npx"), patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="registry unreachable"),
    ):
        assert warm_agent_browser_npx_cache() is False


def test_returns_false_instead_of_raising_on_timeout():
    with patch("shutil.which", return_value="/usr/bin/npx"), patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["npx"], timeout=60.0),
    ):
        assert warm_agent_browser_npx_cache() is False


def test_returns_false_instead_of_raising_on_unexpected_exception():
    """Fire-and-forget contract: callers never wrap this in try/except
    themselves (main.py and doctor.py both call it bare), so any exception
    inside the subprocess call must be swallowed here, not propagated."""
    with patch("shutil.which", return_value="/usr/bin/npx"), patch(
        "subprocess.run", side_effect=OSError("fork failed")
    ):
        assert warm_agent_browser_npx_cache() is False
