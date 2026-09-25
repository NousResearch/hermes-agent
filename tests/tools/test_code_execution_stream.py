"""Real subprocess/socket round trips, remote cells, and cancellation isolation."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import pytest

from agent import secret_scope
from tools.code_execution_stream import open_remote_rpc
from tools.code_execution_tool import _run_remote_per_call, generate_hermes_tools_module
from tools.code_kernel_remote import execute_in_remote_kernel, shutdown_all_remote_kernels


class ShellRemote:
    """Exercise remote staging/protocols with actual processes, without a daemon."""

    def __init__(self, root, streaming="stream"):
        from pm.shell import bash
        self.shell = bash()
        self.root = root
        self.streaming = streaming
        self.commands = []
        self.processes = []
        if streaming == "unsupported":
            self.open_code_rpc = None

    def get_temp_dir(self):
        return self.root

    def execute(self, command, cwd=None, timeout=120):
        self.commands.append(command)
        from tools.terminal_tool_sudo import _rewrite_compound_background
        command = _rewrite_compound_background(command)
        result = subprocess.run([self.shell, "-c", command], capture_output=True,
                                text=True, cwd=cwd, timeout=timeout)
        return {"output": result.stdout, "returncode": result.returncode}

    def open_code_rpc(self, command):
        if self.streaming == "unavailable":
            command = "exit 1"
        process = subprocess.Popen([self.shell, "-c", command], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.processes.append(process)
        return process


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("streaming", ["stream", "unsupported", "unavailable"])
@pytest.mark.parametrize("persistent", [True, False])
def test_remote_calls_preserve_order_budget_and_profile_scope(tmp_path, monkeypatch, streaming, persistent):
    """A→B→A on real remote processes; stubs and the host agree on results/counts."""
    seen = []

    def dispatch(name, args, **kw):
        home = secret_scope.current_secret_scope_home()
        seen.append((home, name, args["path"]))
        return json.dumps({"home": home, "path": args["path"]})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    code = '''
from hermes_tools import read_file, _call
import json
results = [read_file(str(i)) for i in range(4)]
assert "not available" in _call("forbidden", {})["error"]
assert "limit reached" in read_file("over-budget")["error"]
print(json.dumps(results))
'''
    secret_scope.set_multiplex_active(True)
    try:
        # Keep Unix socket paths below sun_path on macOS.
        with tempfile.TemporaryDirectory(prefix="hrpc-", dir="/tmp") as root:
            env = ShellRemote(root, streaming)
            for profile in ("A", "B", "A"):
                home = tmp_path / profile
                home.mkdir(exist_ok=True)
                token = secret_scope.set_secret_scope({}, profile_home=str(home))
                try:
                    if persistent:
                        result = execute_in_remote_kernel(
                            code, env=env, env_type="ssh", task_env_id=profile,
                            sandbox_tools=frozenset({"read_file"}), timeout=15,
                            max_tool_calls=4, reset=False)
                        output = result["stdout"]
                    else:
                        result = json.loads(_run_remote_per_call(
                            env, "ssh", code, profile, frozenset({"read_file"}),
                            timeout=15, max_tool_calls=4, exec_start=time.monotonic()))
                        output = result["output"]
                    assert result["status"] == "success", result
                    assert result["tool_calls_made"] == 4
                    assert json.loads(output) == [{"home": str(home), "path": str(i)} for i in range(4)]
                finally:
                    secret_scope.reset_secret_scope(token)
            assert [entry[2] for entry in seen] == [str(i) for i in range(4)] * 3
            assert bool(env.processes) is (streaming != "unsupported")
            assert all(p.poll() is not None for p in env.processes)
            assert any("ls -1" in cmd for cmd in env.commands) is (streaming != "stream")
            shutdown_all_remote_kernels()
    finally:
        secret_scope.set_multiplex_active(False)
        shutdown_all_remote_kernels()


@pytest.mark.platforms("posix")
def test_stream_disconnect_never_replays_and_other_owner_keeps_working(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def dispatch(name, args, **kw):
        calls.append(args["path"])
        if args["path"] == "slow":
            entered.set()
            assert release.wait(15)
        return json.dumps({"path": args["path"]})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    with tempfile.TemporaryDirectory(prefix="hrpc-", dir="/tmp") as root:
        env = ShellRemote(root)
        streams = []
        try:
            for owner in ("A", "B"):
                stream = open_remote_rpc(env, root, owner, [0], 5, frozenset({"read_file"}), owner)
                assert stream is not None
                streams.append(stream)
            # Generated clients in separate processes, so environment and globals cannot cross.
            stub = Path(root) / "hermes_tools.py"
            stub.write_text(generate_hermes_tools_module(["read_file"], transport="remote"))

            def call(index, path):
                import sys
                child_env = {"PATH": os.environ["PATH"], "HERMES_RPC_SOCKET": streams[index].endpoint,
                             "HERMES_RPC_TOKEN": "AB"[index]}
                return subprocess.run([sys.executable, "-c",
                                       f"from hermes_tools import read_file; print(read_file({path!r}))"],
                                      env=child_env, cwd=root, capture_output=True, text=True, timeout=15)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                slow = pool.submit(call, 0, "slow")
                assert entered.wait(10)
                # Terminate just this transport after dispatch, with no response delivered.
                streams[0].process.terminate()
                streams[0].process.wait(timeout=5)
                assert call(1, "neighbor").returncode == 0
                failed = slow.result(timeout=10)
                assert failed.returncode != 0
                assert "not retried" in failed.stderr
                assert calls == ["slow", "neighbor"]
                assert not list(Path(root).glob("req_*"))
        finally:
            release.set()
            for stream in streams:
                stream.close()


@pytest.mark.platforms("posix")
def test_stream_refusals_and_large_payloads_do_not_corrupt_the_next_call(monkeypatch):
    large_request, large_response = "r" * (2 * 1024 * 1024), "s" * (17 * 1024 * 1024)
    calls, counter = [], [0]

    def dispatch(name, args, **kw):
        calls.append(args["path"])
        return json.dumps({"path": large_response if args["path"] == "large" else args["path"]})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    with tempfile.TemporaryDirectory(prefix="hrpc-", dir="/tmp") as root:
        env = ShellRemote(root)
        stream = open_remote_rpc(env, root, "owner", counter, 10, frozenset({"read_file"}), "right")
        assert stream is not None
        try:
            monkeypatch.setenv("HERMES_RPC_SOCKET", stream.endpoint)
            monkeypatch.setenv("HERMES_RPC_TOKEN", "wrong")
            namespace = {}
            exec(generate_hermes_tools_module(["read_file"], transport="remote"), namespace)
            assert "Unauthorized" in namespace["read_file"]("denied")["error"]
            monkeypatch.setenv("HERMES_RPC_TOKEN", "right")
            assert "not available" in namespace["_call"]("write_file", {})["error"]
            assert calls == [] and counter == [0]
            assert namespace["read_file"](large_request) == {"path": large_request}
            assert namespace["read_file"]("large") == {"path": large_response}
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(namespace["read_file"], map(str, range(8))))
            assert results == [{"path": str(i)} for i in range(8)]
            assert counter == [10] and len(calls) == len(set(calls)) == 10
        finally:
            stream.close()
