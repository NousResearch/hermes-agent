"""Focused tests for execute_code sandbox RPC dispatch policy."""

from __future__ import annotations

import json

from tools.code_execution_tool import _dispatch_sandbox_tool_request


def _dispatch_request(
    request: dict,
    *,
    allowed_tools=frozenset({"terminal", "read_file"}),
    counter=None,
    log=None,
    max_tool_calls=3,
    dispatch_fn=None,
):
    if counter is None:
        counter = [0]
    if log is None:
        log = []
    if dispatch_fn is None:
        dispatch_fn = lambda name, args, task_id=None: json.dumps({"ok": name})

    result = _dispatch_sandbox_tool_request(
        request,
        task_id="rpc-test-task",
        tool_call_log=log,
        tool_call_counter=counter,
        max_tool_calls=max_tool_calls,
        allowed_tools=allowed_tools,
        call_start=1.0,
        dispatch_fn=dispatch_fn,
    )
    return json.loads(result), counter, log


def test_dispatch_rejects_tools_outside_execute_code_allowlist():
    payload, counter, log = _dispatch_request({"tool": "send_message", "args": {}})

    assert payload == {
        "error": "Tool 'send_message' is not available in execute_code. Available: read_file, terminal"
    }
    assert counter == [0]
    assert log == []


def test_dispatch_rejects_requests_after_tool_call_limit():
    payload, counter, log = _dispatch_request(
        {"tool": "read_file", "args": {"path": "README.md"}},
        counter=[3],
        max_tool_calls=3,
    )

    assert payload == {
        "error": "Tool call limit reached (3). No more tool calls allowed in this execution."
    }
    assert counter == [3]
    assert log == []


def test_dispatch_strips_blocked_terminal_parameters_before_calling_tool():
    seen = {}

    def dispatch_fn(name, args, task_id=None):
        seen.update({"name": name, "args": dict(args), "task_id": task_id})
        return json.dumps({"success": True})

    payload, counter, log = _dispatch_request(
        {
            "tool": "terminal",
            "args": {
                "command": "printf safe",
                "background": True,
                "pty": True,
                "notify_on_complete": True,
                "watch_patterns": ["DONE"],
            },
        },
        dispatch_fn=dispatch_fn,
    )

    assert payload == {"success": True}
    assert seen == {
        "name": "terminal",
        "args": {"command": "printf safe"},
        "task_id": "rpc-test-task",
    }
    assert counter == [1]
    assert log == [{"tool": "terminal", "args_preview": "{'command': 'printf safe'}", "duration": log[0]["duration"]}]


def test_dispatch_logs_successful_tool_calls():
    payload, counter, log = _dispatch_request(
        {"tool": "read_file", "args": {"path": "README.md"}},
        dispatch_fn=lambda name, args, task_id=None: json.dumps({"content": "hello"}),
    )

    assert payload == {"content": "hello"}
    assert counter == [1]
    assert len(log) == 1
    assert log[0]["tool"] == "read_file"
    assert log[0]["args_preview"] == "{'path': 'README.md'}"
    assert isinstance(log[0]["duration"], float)


def test_dispatch_converts_tool_exceptions_to_tool_error():
    def dispatch_fn(name, args, task_id=None):
        raise RuntimeError("boom")

    payload, counter, log = _dispatch_request(
        {"tool": "read_file", "args": {"path": "README.md"}},
        dispatch_fn=dispatch_fn,
    )

    assert payload == {"error": "boom"}
    assert counter == [1]
    assert log[0]["tool"] == "read_file"
