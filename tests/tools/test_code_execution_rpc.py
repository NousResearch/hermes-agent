"""Sandbox RPC authorization: drive the real generated ``hermes_tools._call()``
against the UDS server loop and the remote file-RPC poller, and check that only
the authenticated grant intersection is ever dispatched (#84271)."""

import fnmatch
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from typing import Any, Callable, cast
from unittest.mock import MagicMock, patch

from tools.code_execution_tool import _sandbox_tools_for, generate_hermes_tools_module


def _mock_handle_function_call(function_name, function_args, task_id=None, user_task=None):
    if function_name == "terminal":
        cmd = function_args.get("command", "")
        return json.dumps({"output": f"mock output for: {cmd}", "exit_code": 0})
    return json.dumps({"error": f"Unknown tool in mock: {function_name}"})


class _FakeFileRpcEnvironment:
    """Map the remote poller's shell operations onto a local temp directory."""

    def __init__(self, rpc_dir):
        self.rpc_dir = rpc_dir

    def execute(self, command, cwd=None, timeout=None, *, stdin_data=None):
        if stdin_data is not None:
            result = subprocess.run(
                ["bash", "-c", command],
                input=stdin_data,
                text=True,
                capture_output=True,
                cwd=cwd,
                timeout=timeout,
                check=False,
            )
            return {"output": result.stdout, "returncode": result.returncode}

        parts = shlex.split(command)
        if parts[:2] == ["ls", "-1"]:
            pattern = os.path.normpath(parts[2])
            directory, filename = os.path.split(pattern)
            paths = sorted(
                os.path.join(directory, name)
                for name in os.listdir(directory)
                if fnmatch.fnmatch(name, filename)
            )
            return {"output": "\n".join(paths)}

        if parts[0] == "cat":
            with open(parts[1], encoding="utf-8") as request_file:
                return {"output": request_file.read()}

        if parts[:2] == ["rm", "-f"]:
            try:
                os.unlink(parts[2])
            except FileNotFoundError:
                pass
            return {"output": ""}

        raise AssertionError(f"Unexpected fake remote command: {command}")


class TestSandboxRpcAuthorization(unittest.TestCase):
    """Exercise the real generated _call() authorization boundaries."""

    _RPC_TOKEN = "test-rpc-token"
    _REQUESTS = (
        ("terminal", {"command": "echo denied"}),
        ("write_file", {"path": "blocked.txt", "content": "blocked"}),
    )

    @staticmethod
    def _dispatch_recorder(dispatched):
        def dispatch(function_name, function_args, task_id=None, user_task=None):
            dispatched.append((function_name, function_args))
            return _mock_handle_function_call(
                function_name,
                function_args,
                task_id=task_id,
                user_task=user_task,
            )

        return dispatch

    def _run_uds_calls(self, enabled_tools, requests):
        from tools.code_execution_rpc import _rpc_server_loop

        allowed_tools = _sandbox_tools_for(enabled_tools)
        dispatched = []
        tool_call_log = []
        tool_call_counter = [0]
        stop_event = threading.Event()

        # A real Unix socket pair avoids sockaddr_un path limits in deeply
        # nested CI/base checkouts while preserving the RPC authorization loop.
        server_connection, client_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with server_connection, client_socket:
            server_sock = MagicMock(spec=socket.socket)
            server_sock.accept.return_value = (server_connection, ("peer", 0))

            def run_server():
                with patch(
                    "model_tools.handle_function_call",
                    side_effect=self._dispatch_recorder(dispatched),
                ):
                    _rpc_server_loop(
                        server_sock,
                        "test-task",
                        tool_call_log,
                        tool_call_counter,
                        max_tool_calls=10,
                        allowed_tools=allowed_tools,
                        stop_event=stop_event,
                        rpc_token=self._RPC_TOKEN,
                    )

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            namespace: dict[str, object] = {"__name__": "hermes_tools"}
            try:
                with patch.dict(
                    os.environ,
                    {
                        "HERMES_RPC_TOKEN": self._RPC_TOKEN,
                    },
                ):
                    exec(
                        generate_hermes_tools_module(
                            list(allowed_tools),
                            transport="uds",
                        ),
                        namespace,
                    )
                    namespace["_sock"] = client_socket
                    call = cast(
                        Callable[[str, dict[str, str]], dict[str, Any]],
                        namespace["_call"],
                    )
                    responses = [
                        call(tool_name, args)
                        for tool_name, args in requests
                    ]
            finally:
                client_sock = cast(Any, namespace.get("_sock"))
                if client_sock is not None:
                    client_sock.close()
                stop_event.set()
                server_sock.close()
                server_thread.join(timeout=5)

        self.assertFalse(server_thread.is_alive())
        return responses, dispatched

    def _run_file_calls(self, enabled_tools, requests):
        from tools.code_execution_tool import _rpc_poll_loop

        allowed_tools = _sandbox_tools_for(enabled_tools)
        dispatched = []
        tool_call_log = []
        tool_call_counter = [0]
        stop_event = threading.Event()

        with tempfile.TemporaryDirectory(prefix="hermes-rpc-") as temp_dir:
            rpc_dir = os.path.join(temp_dir, "rpc")
            os.mkdir(rpc_dir)
            env = _FakeFileRpcEnvironment(rpc_dir)

            def run_poller():
                with patch(
                    "model_tools.handle_function_call",
                    side_effect=self._dispatch_recorder(dispatched),
                ):
                    _rpc_poll_loop(
                        env,
                        rpc_dir,
                        "test-task",
                        tool_call_log,
                        tool_call_counter,
                        max_tool_calls=10,
                        allowed_tools=allowed_tools,
                        stop_event=stop_event,
                        rpc_token=self._RPC_TOKEN,
                    )

            poller_thread = threading.Thread(target=run_poller, daemon=True)
            poller_thread.start()
            namespace = {"__name__": "hermes_tools"}
            try:
                with patch.dict(
                    os.environ,
                    {
                        "HERMES_RPC_DIR": rpc_dir,
                        "HERMES_RPC_TOKEN": self._RPC_TOKEN,
                    },
                ):
                    exec(
                        generate_hermes_tools_module(
                            list(allowed_tools),
                            transport="file",
                        ),
                        namespace,
                    )
                    call = cast(
                        Callable[[str, dict[str, str]], dict[str, Any]],
                        namespace["_call"],
                    )
                    responses = [
                        call(tool_name, args)
                        for tool_name, args in requests
                    ]
            finally:
                stop_event.set()
                poller_thread.join(timeout=5)

        self.assertFalse(poller_thread.is_alive())
        return responses, dispatched

    @unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
    def test_uds_empty_and_nonoverlapping_grants_reject_both_tools(self):
        for enabled_tools in ([], ["vision_analyze"]):
            with self.subTest(enabled_tools=enabled_tools):
                responses, dispatched = self._run_uds_calls(
                    enabled_tools,
                    self._REQUESTS,
                )

                for response, (tool_name, _) in zip(responses, self._REQUESTS):
                    self.assertIn(
                        f"Tool '{tool_name}' is not available",
                        response.get("error", ""),
                    )
                self.assertEqual(dispatched, [])

    @unittest.skipIf(sys.platform == "win32", "UDS not available on Windows")
    def test_uds_partial_grant_dispatches_only_the_intersection(self):
        responses, dispatched = self._run_uds_calls(
            ["terminal", "vision_analyze"],
            (
                ("terminal", {"command": "echo allowed"}),
                ("write_file", {"path": "blocked.txt", "content": "blocked"}),
            ),
        )

        self.assertIn("mock output for: echo allowed", responses[0].get("output", ""))
        self.assertIn("Available: terminal", responses[1].get("error", ""))
        self.assertEqual([name for name, _ in dispatched], ["terminal"])

    def test_file_empty_and_nonoverlapping_grants_reject_both_tools(self):
        for enabled_tools in ([], ["vision_analyze"]):
            with self.subTest(enabled_tools=enabled_tools):
                responses, dispatched = self._run_file_calls(
                    enabled_tools,
                    self._REQUESTS,
                )

                for response, (tool_name, _) in zip(responses, self._REQUESTS):
                    self.assertIn(
                        f"Tool '{tool_name}' is not available",
                        response.get("error", ""),
                    )
                self.assertEqual(dispatched, [])

    def test_file_partial_grant_dispatches_only_the_intersection(self):
        responses, dispatched = self._run_file_calls(
            ["terminal", "vision_analyze"],
            (
                ("terminal", {"command": "echo allowed"}),
                ("write_file", {"path": "blocked.txt", "content": "blocked"}),
            ),
        )

        self.assertIn("mock output for: echo allowed", responses[0].get("output", ""))
        self.assertIn("Available: terminal", responses[1].get("error", ""))
        self.assertEqual([name for name, _ in dispatched], ["terminal"])
