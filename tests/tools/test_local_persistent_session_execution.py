"""Local persistence and session routing compose through terminal and file bring-up."""
import json

import pytest

from hermes_cli.session_execution import (
    SessionExecutionContext, register_session_execution_context, remove_session_execution_context,
)
from tools.file_tools import read_file_tool
from tools.terminal_tool import terminal_tool
from tools.terminal_tool_lifecycle import cleanup_vm


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("persistent, expected", [(None, "kept"), ("true", "kept"), ("false", "none")])
@pytest.mark.parametrize("file_first", [False, True])
def test_session_routing_wins_with_both_local_persistence_modes(
        tmp_path, monkeypatch, persistent, expected, file_first):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    if persistent is None:
        monkeypatch.delenv("TERMINAL_LOCAL_PERSISTENT", raising=False)
    else:
        monkeypatch.setenv("TERMINAL_LOCAL_PERSISTENT", persistent)
    monkeypatch.setenv("ROUTING_HOST_VALUE", "host")
    task = f"local-persistent-session-{file_first}-{persistent}"
    owner = f"local-persistent-owner-{file_first}-{persistent}"
    register_session_execution_context(owner, SessionExecutionContext(
        env_set={"ROUTING_OWNER": "owner"}, env_unset={"ROUTING_HOST_VALUE"}), task_ids=(task,))
    (tmp_path / "sub").mkdir()
    sample = tmp_path / "sample.txt"
    sample.write_text("sample", encoding="utf-8")

    def run(command):
        result = json.loads(terminal_tool(command, task_id=task, session_id=owner))
        assert result.get("exit_code") == 0, result
        assert not result.get("error"), result
        return result["output"].strip()

    try:
        if file_first:
            result = json.loads(read_file_tool(str(sample), task_id=task))
            assert not result.get("error"), result
            assert "sample" in result["content"]
        run("cd sub && export LOCAL_USER_VALUE=kept ROUTING_OWNER=wrong ROUTING_HOST_VALUE=wrong")
        assert run("echo ${LOCAL_USER_VALUE:-none} $ROUTING_OWNER ${ROUTING_HOST_VALUE:-none}") == f"{expected} owner none"
        assert run("pwd -P") == str((tmp_path / "sub").resolve())
    finally:
        cleanup_vm(task)
        remove_session_execution_context(owner)
