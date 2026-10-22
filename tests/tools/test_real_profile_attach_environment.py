"""Real-profile attachment must not request a different browser executable."""
from unittest.mock import patch
from tools import browser_tool_real_profile as rp


def test_real_profile_env_drops_launch_executable_but_keeps_socket():
    source = {"AGENT_BROWSER_EXECUTABLE_PATH": "/bundled/chromium", "DISPLAY": ":0"}
    with patch.object(rp._session, "_prepare_session_socket_dir", return_value="/socket"), patch.object(rp._session, "_agent_browser_command_env", return_value=source.copy()):
        env, socket_dir = rp._real_profile_daemon_env()
    assert "AGENT_BROWSER_EXECUTABLE_PATH" not in env
    assert env["DISPLAY"] == ":0"
    assert socket_dir == "/socket"
    assert source["AGENT_BROWSER_EXECUTABLE_PATH"] == "/bundled/chromium"
