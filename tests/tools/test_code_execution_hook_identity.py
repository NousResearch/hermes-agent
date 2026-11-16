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
        conversation_id="conv-A", enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    nested = _nested(hook_calls, "read_file")
    assert len(nested) == 1
    assert nested[0]["session_id"] == "sess-A"
    assert nested[0]["turn_id"] == "turn-A"
    assert nested[0]["conversation_id"] == "conv-A"


def test_nested_call_over_the_remote_rpc_dispatch_carries_the_outer_session_and_turn(tmp_path, hook_calls):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    dispatch = _default_dispatch("task-A", "sess-A", "turn-A", "conv-A")
    assert "hello" in json.loads(dispatch("read_file", {"path": str(target)}))["content"]

    nested = _nested(hook_calls, "read_file")
    assert len(nested) == 1
    assert nested[0]["session_id"] == "sess-A"
    assert nested[0]["turn_id"] == "turn-A"
    assert nested[0]["conversation_id"] == "conv-A"


@pytest.mark.parametrize("route", ["kernel", "per_call"])
def test_both_remote_routes_hand_the_outer_identity_to_the_rpc_dispatch(route, monkeypatch):
    import inspect
    from contextlib import ExitStack

    from tests.tools.test_code_kernel_remote import ScriptedEnv, _cell, _spawn_ok_handlers
    from tools import code_execution_rpc
    from tools.code_kernel_remote import shutdown_all_remote_kernels

    real, seen = code_execution_rpc._default_dispatch, []

    def _recording(*args, **kwargs):
        seen.append(inspect.signature(real).bind(*args, **kwargs).arguments)
        return real(*args, **kwargs)

    handlers = [("command -v python3", lambda c: {"output": "OK\n", "returncode": 0})]
    if route == "kernel":
        handlers += _spawn_ok_handlers([_cell(stdout="ran\n")])
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    with ExitStack() as stack:
        stack.callback(shutdown_all_remote_kernels)
        stack.enter_context(patch("tools.code_execution_tool._get_or_create_env",
                                  return_value=(ScriptedEnv(handlers), "ssh")))
        stack.enter_context(patch("tools.code_execution_tool._ship_file_to_remote"))
        stack.enter_context(patch.object(code_execution_rpc, "_default_dispatch", side_effect=_recording))
        if route == "per_call":
            stack.enter_context(patch("tools.code_kernel_remote.execute_in_remote_kernel", return_value=None))
        result = json.loads(handle_function_call(
            "execute_code", {"code": "print(1)"}, task_id="task-A", session_id="sess-A", turn_id="turn-A",
            conversation_id="conv-A", enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    assert [(a["session_id"], a["turn_id"], a["conversation_id"]) for a in seen] == [("sess-A", "turn-A", "conv-A")]
