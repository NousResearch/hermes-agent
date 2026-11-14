"""Tool calls made from inside an execute_code cell fire pre_tool_call hooks under the calling
conversation's identity (#145870): without it a hook sees ``session_id=""`` and, in a
multiplexed gateway, can only fall back to the process-wide HERMES_SESSION_ID of whichever
conversation bound a turn last."""

import json
from unittest.mock import patch

import pytest

from model_tools import handle_function_call
from tools.code_execution_rpc import _default_dispatch
from tools.code_kernel import shutdown_all_kernels


@pytest.fixture(autouse=True)
def _local_kernel(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    shutdown_all_kernels()
    with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict", "timeout": 30}):
        yield
    shutdown_all_kernels()


@pytest.fixture
def hook_calls():
    calls = []

    def _record(tool_name, args, **hook_kwargs):
        calls.append((tool_name, hook_kwargs))
        return None, None

    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=_record):
        yield calls


def _nested(calls, tool):
    return [kw for name, kw in calls if name == tool]


def test_nested_call_in_a_local_cell_carries_the_outer_session_and_turn(tmp_path, hook_calls):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    code = f"from hermes_tools import read_file\nprint(read_file({str(target)!r})['content'])"

    result = json.loads(handle_function_call(
        "execute_code", {"code": code}, task_id="task-A", session_id="sess-A", turn_id="turn-A",
        enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    nested = _nested(hook_calls, "read_file")
    assert len(nested) == 1
    assert nested[0]["session_id"] == "sess-A"
    assert nested[0]["turn_id"] == "turn-A"


def test_nested_call_over_the_remote_rpc_dispatch_carries_the_outer_session_and_turn(tmp_path, hook_calls):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    dispatch = _default_dispatch("task-A", "sess-A", "turn-A")
    assert "hello" in json.loads(dispatch("read_file", {"path": str(target)}))["content"]

    nested = _nested(hook_calls, "read_file")
    assert len(nested) == 1
    assert nested[0]["session_id"] == "sess-A"
    assert nested[0]["turn_id"] == "turn-A"
