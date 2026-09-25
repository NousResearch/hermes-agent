"""A timed-out script must report tool calls already dispatched on its behalf."""
import json
import subprocess
import threading
import time

import pytest

from tools.code_execution_tool import _run_remote_per_call
from tools.code_execution_rpc import _handle_rpc_request


@pytest.mark.platforms("posix")
def test_remote_timeout_reports_the_tool_that_is_still_running(tmp_path, monkeypatch):
    from pm.shell import bash
    entered, release = threading.Event(), threading.Event()
    calls = []

    def dispatch(name, args, **kw):
        calls.append(name)
        entered.set()
        assert release.wait(20)
        return json.dumps({"ok": True})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)

    class Remote:
        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, cwd=None, timeout=120):
            from agent.deadline import kill_process_tree
            process = subprocess.Popen([bash(), "-c", command], cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                output, _ = process.communicate(timeout=timeout)
                return {"returncode": process.returncode, "output": output}
            except subprocess.TimeoutExpired:
                kill_process_tree(process.pid)
                process.communicate(timeout=5)
                return {"returncode": 124, "output": ""}

    try:
        result = json.loads(_run_remote_per_call(
            Remote(), "ssh", "from hermes_tools import read_file; read_file('fixture')",
            "owner", frozenset({"read_file"}), timeout=2, max_tool_calls=1,
            exec_start=time.monotonic()))
        assert entered.is_set() and calls == ["read_file"]
        assert result["status"] == "timeout"
        assert result["tool_calls_made"] == len(calls)
    finally:
        release.set()


def test_dispatch_failure_consumes_one_call_but_refusals_are_free():
    counter, log, calls = [0], [], []

    def dispatch(name, args):
        calls.append(name)
        raise RuntimeError("handler failed")

    def call(name):
        return json.loads(_handle_rpc_request(
            {"tool": name, "args": {}}, allowed_tools=frozenset({"read_file"}),
            tool_call_counter=counter, max_tool_calls=1, dispatch=dispatch,
            tool_call_log=log, call_start=time.monotonic(), where="test"))

    assert "not available" in call("forbidden")["error"]
    assert counter == [0]
    assert "handler failed" in call("read_file")["error"]
    assert "limit reached" in call("read_file")["error"]
    assert counter == [1] and calls == ["read_file"] and len(log) == 1
