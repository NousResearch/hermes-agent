"""A background launch that never started must be reported as an error, not a session."""

import json

import pytest

from tools.process_registry import process_registry
from tools.terminal_tool_background import spawn_background_process


class _SandboxEnv:
    """Environment-backed (non-local) backend whose launch wrapper cannot run."""

    cwd = "/root"

    def __init__(self, outcome):
        self._outcome = outcome

    def get_temp_dir(self):
        return "/tmp"

    def execute(self, command, **kwargs):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


@pytest.mark.parametrize("outcome, reason", [
    ({"output": 'exec: "bash": executable file not found in $PATH', "returncode": 127},
     'exec: "bash": executable file not found'),
    (RuntimeError("container 808ea027557c is not running"), "container 808ea027557c is not running"),
])
def test_failed_sandbox_launch_returns_error_not_started_session(outcome, reason):
    before = set(process_registry._running) | set(process_registry._finished)

    result = json.loads(spawn_background_process(
        command="sleep 30", env=_SandboxEnv(outcome), env_type="docker",
        effective_task_id="bg-fail", task_id="bg-fail", session_key="", workdir=None, cwd="/root",
        effective_pty=False, notify_on_complete=True, watch_patterns=None, approval_note=None,
        pty_disabled_reason=None,
    ))

    assert result["exit_code"] == -1
    assert "session_id" not in result
    assert result["output"] != "Background process started"
    assert reason in result["error"]
    assert set(process_registry._running) | set(process_registry._finished) == before
