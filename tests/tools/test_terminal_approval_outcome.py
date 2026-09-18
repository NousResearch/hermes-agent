"""Approval waits are not command execution or network observations."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.terminal_tool import terminal_tool


@pytest.mark.parametrize("outcome", ["timeout", "denied", "pending_approval"])
def test_unapproved_command_never_runs_and_retains_outcome(tmp_path, outcome):
    env = MagicMock(cwd=str(tmp_path))
    approval = {
        "approved": False, "outcome": outcome, "user_consent": False,
        "message": "Approval timed out" if outcome == "timeout" else "Not approved",
        "status": outcome,
    }
    config = {"env_type": "local", "timeout": 10, "cwd": str(tmp_path)}
    with (
        patch("tools.terminal_tool._get_env_config", return_value=config),
        patch("tools.terminal_tool._start_cleanup_thread"),
        patch("tools.terminal_tool._active_environments", {"default": env}),
        patch("tools.terminal_tool._last_activity", {"default": 0}),
        patch("tools.terminal_tool._session_cwd", {}),
        patch("tools.terminal_tool._check_all_guards", return_value=approval),
    ):
        result = json.loads(terminal_tool("curl http://192.0.2.95:8188/v1/models"))
    env.execute.assert_not_called()
    assert result["executed"] is False
    assert result["outcome"] == outcome
    assert result["user_consent"] is False
    assert "not run" in result["execution_note"].lower()
    assert result["output"] == ""
