"""``terminal.max_foreground_timeout`` in a profile's config.yaml sets the foreground cap (#21864).

The cap used to be read once at import from TERMINAL_MAX_FOREGROUND_TIMEOUT, so config.yaml could
not set it and a multiplexed process pinned the launch profile's cap onto every routed profile.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope


@pytest.fixture
def routed_profile(tmp_path, monkeypatch):
    """A launch env capping at 300s, and a routed profile whose config.yaml raises it to 900s."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    monkeypatch.setenv("TERMINAL_MAX_FOREGROUND_TIMEOUT", "300")
    monkeypatch.setattr("agent.secret_scope.build_profile_secret_scope", lambda _h: {})
    home = tmp_path / "profiles" / "slow-builds"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("terminal:\n  max_foreground_timeout: 900\n", encoding="utf-8")
    token = install_profile_terminal_scope(home)
    yield
    reset_terminal_scope(token)


def _run_foreground(timeout):
    from tools.terminal_tool import terminal_tool

    env_config = {"env_type": "local", "timeout": 180, "cwd": "/tmp", "host_cwd": None, "modal_mode": "auto",
                  "docker_image": "", "singularity_image": "", "modal_image": "", "daytona_image": ""}
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "done", "returncode": 0}
    with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
         patch("tools.terminal_tool._last_activity", {"default": 0}), \
         patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
        return json.loads(terminal_tool(command="echo hello", timeout=timeout)), mock_env


def test_profile_cap_keeps_a_long_call_in_the_foreground(routed_profile):
    result, mock_env = _run_foreground(800)

    assert "promoted_from_foreground" not in result
    assert mock_env.execute.call_args[1]["timeout"] == 800


def test_schema_names_the_profile_cap(routed_profile):
    from tools.registry import registry

    (definition,) = registry.get_definitions({"terminal"}, quiet=True)
    description = definition["function"]["parameters"]["properties"]["timeout"]["description"]
    assert "foreground max 900" in description and "300" not in description
