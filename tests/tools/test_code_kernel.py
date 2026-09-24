#!/usr/bin/env python3
"""Tests for execute_code's session kernel mode.

``code_execution.kernel_mode: session`` keeps one Python child alive per
(task, mode, interpreter, cwd, tool-set) so state survives across calls.
These tests pin the contract:

  - default stays per-call (no state carries over unless opted in)
  - state persists across cells and reset=true discards it
  - a raised exception keeps the kernel (and its state) alive
  - a timeout kills the kernel; the next call gets a fresh one
  - fd-level output from user-spawned subprocesses reaches the result
  - sys.exit() inside a cell ends the kernel deliberately

Mode is sourced from ``code_execution.kernel_mode`` in config.yaml only;
tests patch ``_load_config`` directly, mirroring test_code_execution_modes.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Mirror test_code_execution.py — guarantee local backend under xdist."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


from tools.code_execution_tool import execute_code
from tools.code_kernel import _KERNELS, shutdown_all_kernels


@contextmanager
def _kernel_config(**overrides):
    """Pin code_execution config; strict mode keeps the test hermetic."""
    config = {"mode": "strict", "kernel_mode": "session", "timeout": 30}
    config.update(overrides)
    with patch("tools.code_execution_tool._load_config", return_value=config):
        yield


@pytest.fixture(autouse=True)
def _fresh_kernel_registry():
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def _run(code, **kwargs):
    return json.loads(execute_code(code, task_id="kernel-test", **kwargs))


class TestSessionStatePersistence(unittest.TestCase):
    def test_state_persists_across_cells(self):
        with _kernel_config():
            first = _run("x = 41")
            self.assertEqual(first["status"], "success", first)
            self.assertEqual(first["kernel"]["reused"], False)
            second = _run("print(x + 1)")
        self.assertEqual(second["status"], "success", second)
        self.assertIn("42", second["output"])
        self.assertEqual(second["kernel"]["reused"], True)
        self.assertEqual(second["kernel"]["execution_count"], 2)

    def test_reset_discards_state(self):
        with _kernel_config():
            _run("x = 41")
            second = _run("print(x + 1)", reset=True)
        self.assertEqual(second["status"], "error", second)
        self.assertIn("NameError", second.get("error", ""))
        self.assertEqual(second["kernel"]["state_reset"], True)

    def test_exception_keeps_the_kernel_alive(self):
        with _kernel_config():
            _run("a = 7")
            boom = _run("1 / 0")
            self.assertEqual(boom["status"], "error")
            self.assertIn("ZeroDivisionError", boom["error"])
            after = _run("print(a)")
        self.assertEqual(after["status"], "success", after)
        self.assertIn("7", after["output"])
        self.assertEqual(after["kernel"]["reused"], True)

    def test_imports_persist(self):
        with _kernel_config():
            _run("import json as _j")
            second = _run("print(_j.dumps({'k': 1}))")
        self.assertIn('{"k": 1}', second["output"])


class TestKernelLifecycle(unittest.TestCase):
    @pytest.mark.linux_only
    def test_cross_uid_rpc_listener_rejects_unexpected_peer_before_serving(self):
        from tools.code_kernel import _UidFilteringSocket

        unexpected = Mock()
        unexpected.getsockopt.return_value = (
            (123).to_bytes(4, sys.byteorder)
            + (os.geteuid() + 1).to_bytes(4, sys.byteorder)
            + (456).to_bytes(4, sys.byteorder)
        )
        expected = Mock()
        expected.getsockopt.return_value = (
            (789).to_bytes(4, sys.byteorder)
            + os.geteuid().to_bytes(4, sys.byteorder)
            + (456).to_bytes(4, sys.byteorder)
        )
        listener = Mock()
        listener.accept.side_effect = [(unexpected, None), (expected, None)]

        accepted, address = _UidFilteringSocket(listener, os.geteuid()).accept()

        unexpected.close.assert_called_once_with()
        self.assertIs(accepted, expected)
        self.assertIsNone(address)

    @pytest.mark.linux_only
    def test_cross_uid_rpc_listener_closes_peer_when_credential_read_fails(self):
        from tools.code_kernel import _UidFilteringSocket

        unreadable = Mock()
        unreadable.getsockopt.side_effect = OSError("peer disappeared")
        expected = Mock()
        expected.getsockopt.return_value = (
            (789).to_bytes(4, sys.byteorder)
            + os.geteuid().to_bytes(4, sys.byteorder)
            + os.getegid().to_bytes(4, sys.byteorder)
        )
        listener = Mock()
        listener.accept.side_effect = [(unreadable, None), (expected, None)]

        accepted, address = _UidFilteringSocket(listener, os.geteuid()).accept()

        unreadable.close.assert_called_once_with()
        self.assertIs(accepted, expected)
        self.assertIsNone(address)

    @pytest.mark.linux_only
    def test_runner_imports_generated_tools_without_listing_staging_dir(self):
        from tools.code_kernel import KERNEL_RUNNER_SOURCE

        with tempfile.TemporaryDirectory() as outer:
            staging = Path(outer, "staging")
            staging.mkdir(mode=0o700)
            Path(staging, "hermes_tools.py").write_text(
                "VALUE = 42\n", encoding="utf-8"
            )
            runner = Path(staging, "hermes_kernel_runner.py")
            runner.write_text(KERNEL_RUNNER_SOURCE, encoding="utf-8")
            for path in (runner, Path(staging, "hermes_tools.py")):
                path.chmod(0o644)
            staging.chmod(0o311)

            work = Path(outer, "work")
            work.mkdir()
            env = os.environ.copy()
            env.update(
                HERMES_KERNEL_SENTINEL="@@TEST@@",
                HERMES_KERNEL_SPILL_DIR=str(work),
                HERMES_KERNEL_TOOLS_PATH=str(Path(staging, "hermes_tools.py")),
            )
            runner_fd = os.open(runner, os.O_RDONLY)
            try:
                proc = subprocess.run(
                    [sys.executable, f"/proc/self/fd/{runner_fd}"],
                    input=json.dumps({
                        "id": "import",
                        "code": "import hermes_tools; print(hermes_tools.VALUE)",
                    })
                    + "\n",
                    cwd=work,
                    env=env,
                    pass_fds=(runner_fd,),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            finally:
                os.close(runner_fd)

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('"status": "ok"', proc.stdout)
        self.assertIn('"stdout": "42\\n"', proc.stdout)

    def test_plain_kernel_teardown_does_not_import_linux_broker(self):
        from tools.code_kernel import SessionKernel

        proc = Mock()
        proc.poll.return_value = 1
        kernel = SessionKernel(("plain-teardown",))
        kernel.proc = proc
        real_import = __import__

        def import_without_linux_broker(name, *args, **kwargs):
            if name == "tools.local_exec_broker":
                raise ModuleNotFoundError("fcntl")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_linux_broker):
            kernel.teardown()

    @pytest.mark.linux_only
    def test_configured_local_broker_launches_kernel_and_holds_lease(self):
        from tools.code_kernel import SessionKernel, _spawn

        class Connection:
            def __init__(self):
                self.closed = False

            def setblocking(self, _blocking):
                pass

            def recv(self, _size):
                raise BlockingIOError

            def close(self):
                self.closed = True

        launched = {}
        connection = Connection()

        def request_launch(socket_path, **kwargs):
            launched["socket_path"] = socket_path
            launched.update(kwargs)
            return connection, {"ok": True, "pid": 4321, "start_time": 100}, b""

        kernel = SessionKernel(("broker-launch",))
        broker_uid = os.geteuid() + 1
        with (
            patch(
                "hermes_cli.config.load_config_readonly",
                return_value={
                    "terminal": {
                        "local_exec_broker": {
                            "socket": "/run/hermes-broker/broker.sock",
                            "uid": broker_uid,
                        }
                    }
                },
            ),
            patch("tools.local_exec_broker.request_launch", side_effect=request_launch),
            patch("tools.code_kernel.threading.Thread.start"),
            patch("tools.code_kernel._ensure_background_reaper"),
        ):
            _spawn(
                kernel,
                task_id="broker-launch",
                child_python=sys.executable,
                child_cwd="",
                sandbox_tools=frozenset(),
                max_tool_calls=1,
            )

        try:
            self.assertEqual(launched["socket_path"], "/run/hermes-broker/broker.sock")
            self.assertEqual(launched["expected_peer_uid"], broker_uid)
            self.assertIsInstance(launched["runner_fd"], int)
            self.assertIsInstance(launched["stdin_fd"], int)
            self.assertIsInstance(launched["stdout_fd"], int)
            self.assertIsInstance(launched["stderr_fd"], int)
            self.assertNotEqual(launched["stdout_fd"], launched["stderr_fd"])
            self.assertEqual(Path(kernel.tmpdir).parent, Path("/tmp"))
            self.assertEqual(os.stat(kernel.tmpdir).st_mode & 0o777, 0o711)
            self.assertEqual(
                os.stat(Path(kernel.tmpdir, "hermes_tools.py")).st_mode & 0o777,
                0o644,
            )
            rpc_path = Path(launched["env"]["HERMES_RPC_SOCKET"])
            self.assertEqual(rpc_path.parent, Path(kernel.tmpdir))
            self.assertEqual(os.stat(rpc_path).st_mode & 0o777, 0o666)
            self.assertEqual(os.stat(rpc_path.parent).st_mode & 0o777, 0o711)
            self.assertIsNone(launched["cwd"])
            self.assertTrue(launched["scratch"])
            self.assertEqual(launched["env"]["HERMES_KERNEL_INLINE_SPILL"], "1")
            self.assertNotEqual(launched["env"].get("TMPDIR"), str(Path(kernel.tmpdir, "work")))
            self.assertNotIn("HERMES_KERNEL_SPILL_DIR", launched["env"])
            self.assertFalse(Path(kernel.tmpdir, "work").exists())
            self.assertFalse(Path(kernel.tmpdir, "spill").exists())
            self.assertEqual(kernel.proc.pid, 4321)
            self.assertIs(kernel.broker_lease, connection)
            self.assertIsNone(kernel.death_pipe_w)
            self.assertIn("b", kernel.proc.stdin.mode)
            self.assertIn("b", kernel.proc.stdout.mode)
            self.assertIn("b", kernel.proc.stderr.mode)
            self.assertIsNot(kernel.proc.stdout, kernel.proc.stderr)
        finally:
            kernel.teardown()

    @pytest.mark.linux_only
    def test_broker_kernel_launch_does_not_reclose_transferred_pipe_fd(self):
        from tools.code_kernel import _spawn_through_broker

        class Connection:
            def close(self):
                pass

        reused = []

        def request_launch(*_args, **_kwargs):
            return Connection(), {"pid": 123, "start_time": 100}, b""

        def failing_process(
            _conn,
            _pid,
            _start_time,
            stdin_fd,
            stdout_fd,
            stderr_fd,
            _remainder,
        ):
            os.close(stdin_fd)
            replacement = os.open("/dev/null", os.O_RDONLY)
            os.dup2(replacement, stdin_fd)
            if replacement != stdin_fd:
                os.close(replacement)
            reused.append(stdin_fd)
            os.close(stdout_fd)
            os.close(stderr_fd)
            raise RuntimeError("constructor failed after taking descriptor ownership")

        with tempfile.NamedTemporaryFile() as runner:
            try:
                with (
                    patch(
                        "tools.local_exec_broker.request_launch",
                        side_effect=request_launch,
                    ),
                    patch("tools.code_kernel._BrokerKernelProcess", failing_process),
                    self.assertRaisesRegex(RuntimeError, "constructor failed"),
                ):
                    _spawn_through_broker(
                        runner.name,
                        sys.executable,
                        "",
                        {},
                        ("/run/test-broker.sock", os.geteuid()),
                    )
                os.fstat(reused[0])
            finally:
                if reused:
                    try:
                        os.close(reused[0])
                    except OSError:
                        pass

    def test_inline_broker_spill_is_materialized_in_host_staging(self):
        from tools.code_kernel import SessionKernel, _materialize_inline_spill

        kernel = SessionKernel(("broker-spill",))
        kernel.tmpdir = tempfile.mkdtemp(prefix="hermes_kernel_test_")
        payload = {
            "execution_count": 7,
            "stdout_clipped": True,
            "stdout_spill_content": "full output\nfrom worker",
            "stdout_spill_path": "",
        }

        try:
            _materialize_inline_spill(kernel, payload)
            spill = Path(payload["stdout_spill_path"])
            self.assertEqual(spill.parent, Path(kernel.tmpdir))
            self.assertEqual(spill.read_text(encoding="utf-8"), "full output\nfrom worker")
            self.assertEqual(os.stat(spill).st_mode & 0o777, 0o600)
            self.assertNotIn("stdout_spill_content", payload)
        finally:
            kernel.teardown()

    def test_inline_broker_spill_is_capped_by_host(self):
        from tools.code_kernel import SessionKernel, _materialize_inline_spill

        kernel = SessionKernel(("broker-spill-cap",))
        kernel.tmpdir = tempfile.mkdtemp(prefix="hermes_kernel_test_")
        payload = {
            "execution_count": 8,
            "stdout_clipped": True,
            "stdout_spill_content": "x" * 5_000_001,
            "stdout_spill_path": "",
        }

        try:
            _materialize_inline_spill(kernel, payload)
            spill = Path(payload["stdout_spill_path"])
            content = spill.read_text(encoding="utf-8")
            self.assertEqual(content[:5_000_000], "x" * 5_000_000)
            self.assertEqual(content[5_000_000:], "\n\n[... spill capped ...]")
        finally:
            kernel.teardown()

    def test_stdout_reader_reports_non_object_frame_as_protocol_error(self):
        from tools.code_kernel import SessionKernel, _stdout_reader

        body = b"[1, 2]"
        kernel = SessionKernel(("non-object-frame",))
        kernel.sentinel = "@@FRAME@@"
        kernel.proc = Mock()
        kernel.proc.stdout.read1.side_effect = [
            b"\n@@FRAME@@ " + str(len(body)).encode() + b"\n" + body,
            b"",
        ]

        _stdout_reader(kernel)

        self.assertEqual(kernel.response_q.get_nowait(), {"status": "protocol-error"})

    def test_stdout_reader_rejects_oversized_broker_frame_before_body(self):
        from tools.code_kernel import SessionKernel, _stdout_reader

        kernel = SessionKernel(("oversized-broker-frame",))
        kernel.sentinel = "@@FRAME@@"
        kernel.broker_lease = Mock()
        kernel.proc = Mock()
        kernel.proc.stdout.read1.side_effect = [
            b"\n@@FRAME@@ 67108865\n",
            b"",
        ]

        _stdout_reader(kernel)

        self.assertEqual(kernel.response_q.get_nowait(), {"status": "protocol-error"})
        self.assertEqual(kernel.proc.stdout.read1.call_count, 1)

    @pytest.mark.linux_only
    def test_configured_local_broker_failure_does_not_fall_back_to_popen(self):
        from tools.code_kernel import SessionKernel, _spawn
        from tools.local_exec_broker import BrokerError

        kernel = SessionKernel(("broker-refusal",))
        with (
            patch(
                "hermes_cli.config.load_config_readonly",
                return_value={
                    "terminal": {
                        "local_exec_broker": {
                            "socket": "/run/hermes-broker/broker.sock",
                            "uid": os.geteuid(),
                        }
                    }
                },
            ),
            patch(
                "tools.local_exec_broker.request_launch",
                side_effect=BrokerError("unauthorized", "refused"),
            ),
            patch("tools.code_kernel.subprocess.Popen") as popen,
        ):
            with self.assertRaisesRegex(BrokerError, "unauthorized: refused"):
                _spawn(
                    kernel,
                    task_id="broker-refusal",
                    child_python=sys.executable,
                    child_cwd="",
                    sandbox_tools=frozenset(),
                    max_tool_calls=1,
                )
        popen.assert_not_called()
        kernel.teardown()

    @pytest.mark.linux_only
    def test_broker_teardown_cleans_remaining_handles_after_bad_exit_reply(self):
        from tools.code_kernel import SessionKernel
        from tools.local_exec_broker import BrokerError

        proc = Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = BrokerError("bad_reply", "malformed exit status")
        proc.pid = 4321
        lease = Mock()
        server_sock = Mock()
        kernel = SessionKernel(("broker-bad-reply",))
        kernel.proc = proc
        kernel.broker_lease = lease
        kernel.server_sock = server_sock

        kernel.teardown()

        proc.kill.assert_called_once_with()
        lease.close.assert_called_once_with()
        server_sock.close.assert_called_once_with()
        assert kernel.broker_lease is None
        assert kernel.server_sock is None

    @pytest.mark.linux_only
    def test_broker_teardown_retires_output_handles_after_wait_timeout(self):
        from tools.code_kernel import SessionKernel

        proc = Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("local execution broker", 5)
        proc.pid = 4321
        kernel = SessionKernel(("broker-timeout",))
        kernel.proc = proc
        kernel.broker_lease = Mock()

        kernel.teardown()

        proc._close_owned_handles.assert_called_once_with()

    @pytest.mark.linux_only
    def test_broker_teardown_cleans_up_when_alive_check_gets_bad_reply(self):
        from tools.code_kernel import SessionKernel, _BrokerKernelProcess

        lease, peer = socket.socketpair()
        stdin_r, stdin_w = os.pipe()
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        proc = _BrokerKernelProcess(
            lease, 4321, 100, stdin_w, stdout_r, stderr_r, b"{}\n"
        )
        os.close(stdin_r)
        stdin_r = -1
        os.close(stdout_w)
        stdout_w = -1
        os.close(stderr_w)
        stderr_w = -1
        server_sock = Mock()
        kernel = SessionKernel(("broker-bad-alive-reply",))
        kernel.proc = proc
        kernel.broker_lease = lease
        kernel.server_sock = server_sock
        try:
            kernel.teardown()
            server_sock.close.assert_called_once_with()
            assert kernel.broker_lease is None
            assert kernel.server_sock is None
        finally:
            peer.close()
            lease.close()
            for fd in (stdin_r, stdout_w, stderr_w):
                if fd >= 0:
                    os.close(fd)

    @pytest.mark.linux_only
    def test_broker_process_releases_owned_handles_on_every_terminal_path(self):
        from tools.code_kernel import _BrokerKernelProcess
        from tools.local_exec_broker import BrokerError

        for action, remainder in (
            ("kill", b""),
            ("exit", b'{"exit": 0}\n'),
            ("malformed", b"{}\n"),
        ):
            with self.subTest(action=action):
                lease, peer = socket.socketpair()
                stdin_r, stdin_w = os.pipe()
                stdout_r, stdout_w = os.pipe()
                stderr_r, stderr_w = os.pipe()
                proc = _BrokerKernelProcess(
                    lease,
                    os.getpid(),
                    100,
                    stdin_w,
                    stdout_r,
                    stderr_r,
                    remainder,
                )
                os.close(stdin_r)
                os.close(stdout_w)
                os.close(stderr_w)
                try:
                    if action == "kill":
                        with patch(
                            "tools.local_exec_broker._process_start_time",
                            return_value=101,
                        ):
                            proc.kill()
                            assert proc.poll() == -9
                    elif action == "malformed":
                        with self.assertRaises(BrokerError):
                            proc.poll()
                    else:
                        self.assertEqual(proc.poll(), 0)
                    self.assertTrue(proc.stdin.closed)
                    self.assertTrue(proc.stdout.closed)
                    self.assertTrue(proc.stderr.closed)
                    self.assertEqual(lease.fileno(), -1)
                finally:
                    peer.close()
                    if not proc.stdin.closed:
                        proc.stdin.close()
                    if not proc.stdout.closed:
                        proc.stdout.close()
                    if not proc.stderr.closed:
                        proc.stderr.close()
                    lease.close()

    @pytest.mark.linux_only
    def test_broker_kill_closes_lease_before_retiring_stdout(self):
        from tools.code_kernel import _BrokerKernelProcess

        class GuardedStdout:
            closed = False

            def close(self):
                assert lease.fileno() == -1
                self.closed = True

        lease, peer = socket.socketpair()
        stdin_r, stdin_w = os.pipe()
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        proc = _BrokerKernelProcess(
            lease, 4321, 100, stdin_w, stdout_r, stderr_r, b""
        )
        guard = GuardedStdout()
        try:
            os.close(stdin_r)
            stdin_r = -1
            os.close(stdout_w)
            stdout_w = -1
            os.close(stderr_w)
            stderr_w = -1
            proc.stdout.close()
            proc.stdout = guard
            with patch(
                "tools.local_exec_broker._process_start_time",
                return_value=101,
            ):
                proc.kill()
                assert lease.fileno() == -1
                assert guard.closed is False
                assert proc.poll() == -9
                assert guard.closed is True
        finally:
            peer.close()
            lease.close()
            for fd in (
                stdin_r,
                stdin_w,
                stdout_r,
                stdout_w,
                stderr_r,
                stderr_w,
            ):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    @pytest.mark.linux_only
    def test_broker_poll_does_not_close_output_held_by_reader_thread(self):
        import threading

        from tools.code_kernel import _BrokerKernelProcess

        lease, peer = socket.socketpair()
        stdin_r, stdin_w = os.pipe()
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        proc = _BrokerKernelProcess(
            lease, 4321, 100, stdin_w, stdout_r, stderr_r, b""
        )
        reader_started = threading.Event()
        reader_done = threading.Event()
        poll_done = threading.Event()
        result = []

        def read_stdout():
            reader_started.set()
            proc.stdout.read()
            reader_done.set()

        def poll_process():
            result.append(proc.poll())
            poll_done.set()

        reader = threading.Thread(target=read_stdout, daemon=True)
        poller = threading.Thread(target=poll_process, daemon=True)
        try:
            os.close(stdin_r)
            stdin_r = -1
            os.close(stderr_w)
            stderr_w = -1
            reader.start()
            assert reader_started.wait(1)
            time.sleep(0.05)
            with patch(
                "tools.local_exec_broker._process_start_time",
                return_value=101,
            ):
                proc.kill()
                poller.start()
                assert poll_done.wait(1), "poll blocked closing stdout under its reader"
            self.assertEqual(result, [-9])
            self.assertFalse(proc.stdout.closed)
        finally:
            peer.close()
            lease.close()
            if stdout_w >= 0:
                os.close(stdout_w)
                stdout_w = -1
            reader.join(timeout=1)
            poller.join(timeout=1)
            for stream in (proc.stdout, proc.stderr):
                if not stream.closed:
                    stream.close()
            for fd in (stdin_r, stdout_w, stderr_w):
                if fd >= 0:
                    os.close(fd)

    @pytest.mark.linux_only
    def test_broker_process_does_not_wait_on_a_reused_pid(self):
        from tools.code_kernel import _BrokerKernelProcess

        lease, peer = socket.socketpair()
        stdin_r, stdin_w = os.pipe()
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        try:
            with patch(
                "tools.local_exec_broker._process_start_time",
                return_value=101,
            ):
                proc = _BrokerKernelProcess(
                    lease, 4321, 100, stdin_w, stdout_r, stderr_r, b""
                )
                os.close(stdin_r)
                stdin_r = -1
                os.close(stdout_w)
                stdout_w = -1
                os.close(stderr_w)
                stderr_w = -1
                proc.kill()
                self.assertEqual(proc.poll(), -9)
        finally:
            peer.close()
            lease.close()
            for fd in (stdin_r, stdout_w, stderr_w):
                if fd != -1:
                    os.close(fd)

    def test_kernel_exits_when_its_backend_parent_dies(self):
        """A kernel must not outlive the host that spawned it, even when the
        host dies without cleanup (SIGKILL/OOM/crash). Windows: inherited
        SYNCHRONIZE handle; POSIX: inherited death pipe. Both are proven the
        same way — kill the host mid-cell, the kernel is gone within seconds."""
        import psutil

        repo_root = str(Path(__file__).resolve().parents[2])
        host_src = textwrap.dedent(f"""
            import json, os, sys, time
            os.environ["HERMES_HOME"] = sys.argv[1]
            sys.path.insert(0, {repo_root!r})
            from tools.code_kernel import SessionKernel, _spawn
            k = SessionKernel(("parent-death",))
            _spawn(k, task_id="parent-death", child_python=sys.executable,
                   child_cwd="", sandbox_tools=frozenset(), max_tool_calls=1)
            cell = json.dumps({{"id": "x", "code": "import os, time\\n"
                "assert 'HERMES_KERNEL_PARENT_PROCESS_HANDLE' not in os.environ\\n"
                "assert 'HERMES_KERNEL_PARENT_DEATH_FD' not in os.environ\\n"
                "time.sleep(300)"}}) + "\\n"
            k.proc.stdin.write(cell.encode()); k.proc.stdin.flush()
            print(k.proc.pid, flush=True)
            time.sleep(600)
        """)
        with tempfile.TemporaryDirectory() as home:
            host = subprocess.Popen(
                [sys.executable, "-c", host_src, home],
                stdout=subprocess.PIPE, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                kernel = psutil.Process(int(host.stdout.readline()))
                time.sleep(0.5)
                self.assertTrue(kernel.is_running(), "kernel never came up")
                host.kill()
                host.wait(timeout=10)
                try:
                    kernel.wait(timeout=10)
                except psutil.TimeoutExpired:
                    kernel.kill()
                    self.fail("session kernel survived its backend parent")
            finally:
                if host.poll() is None:
                    host.kill()

    def test_timeout_kills_the_kernel_and_reports_state_loss(self):
        with _kernel_config(timeout=1):
            slow = _run("import time\ntime.sleep(30)")
            self.assertEqual(slow["status"], "timeout", slow)
            self.assertIn("state was lost", slow["error"])
        self.assertEqual(len(_KERNELS), 0)
        with _kernel_config():
            fresh = _run("print('alive')")
        self.assertEqual(fresh["status"], "success", fresh)
        self.assertEqual(fresh["kernel"]["reused"], False)
        self.assertIn("alive", fresh["output"])

    def test_sys_exit_ends_the_kernel(self):
        with _kernel_config():
            done = _run("import sys\nsys.exit(0)")
            self.assertEqual(done["kernel"].get("ended"), True, done)
            self.assertEqual(len(_KERNELS), 0)
            fresh = _run("print('respawned')")
        self.assertEqual(fresh["kernel"]["reused"], False)
        self.assertIn("respawned", fresh["output"])

    def test_subprocess_fd_output_reaches_the_result(self):
        code = (
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, '-c', \"print('raw-passthrough')\"])\n"
        )
        with _kernel_config():
            result = _run(code)
        self.assertEqual(result["status"], "success", result)
        self.assertIn("raw-passthrough", result["output"])


class TestModelFacingReset(unittest.TestCase):
    def test_reset_is_reachable_from_a_model_call_despite_stale_kernel_mode(self):
        """Session kernels are always on (#96787), so ``reset`` is the model's only
        way out of poisoned state. A stale ``kernel_mode: per-call`` key must not
        drop it from the schema, and a model-shaped call routed through the
        registered handler must actually discard the kernel's state."""
        from tools.code_execution_tool import _execute_code_handler, build_execute_code_schema

        with _kernel_config(kernel_mode="per-call"):
            schema = build_execute_code_schema(mode="strict")
            self.assertEqual(schema["parameters"]["properties"]["reset"]["type"], "boolean")
            _execute_code_handler({"code": "x = 41"}, task_id="kernel-test")
            kept = json.loads(_execute_code_handler({"code": "print(x + 1)"}, task_id="kernel-test"))
            self.assertIn("42", kept["output"], kept)
            reset = json.loads(_execute_code_handler(
                {"code": "print(x + 1)", "reset": True}, task_id="kernel-test"))
        self.assertEqual(reset["status"], "error", reset)
        self.assertIn("NameError", reset.get("error", ""))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestKernelOwnershipAndLifecycle(unittest.TestCase):
    """The kernel belongs to the conversation, and its lifetime is bounded.

    run_agent mints a fresh task id per top-level turn, so a task-keyed
    kernel would neither survive the next user turn nor ever be disposed
    with anything. The owner is the approval session key; disposal rides
    the same session boundary that clears approval/yolo state, idle
    kernels are reaped, and the process-wide live count is capped (the
    lifecycle shape carried forward from hermes-agent#88637).
    """

    def _run_as(self, session_key, code, task_id, **kwargs):
        from tools.approval_context import reset_current_session_key, set_current_session_key

        token = set_current_session_key(session_key)
        try:
            return json.loads(execute_code(code, task_id=task_id, **kwargs))
        finally:
            reset_current_session_key(token)

    def test_state_survives_across_turns_of_one_conversation(self):
        # Two top-level turns: same session, different per-turn task ids.
        with _kernel_config():
            first = self._run_as("conv-a", "x = 41", task_id="turn-1")
            self.assertEqual(first["status"], "success", first)
            second = self._run_as("conv-a", "print(x + 1)", task_id="turn-2")
        self.assertEqual(second["status"], "success", second)
        self.assertIn("42", second["output"])
        self.assertEqual(second["kernel"]["reused"], True)

    def test_sessions_are_isolated_from_each_other(self):
        # Same task id, different sessions: no state may cross.
        with _kernel_config():
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            other = self._run_as("conv-b", "print(x + 1)", task_id="turn-1")
        self.assertEqual(other["status"], "error", other)
        self.assertIn("NameError", other.get("error", ""))

    def test_delegated_children_get_their_own_kernels(self):
        """A delegated child runs in a COPY of the parent's context and
        inherits the parent's approval session key — the naive owner
        resolution attached the child to the parent's kernel and leaked
        in-memory state across the delegation boundary (both directions,
        verified live). The owner must be qualified for child contexts."""
        from agent.delegation_context import delegated_child_context

        with _kernel_config():
            self._run_as("conv-a", "parent_secret = 'p'", task_id="turn-1")
            with delegated_child_context("child-1"):
                leak = self._run_as(
                    "conv-a",
                    "print(globals().get('parent_secret', 'ISOLATED'))",
                    task_id="child-task",
                )
                self._run_as("conv-a", "child_secret = 'c'", task_id="child-task")
            back = self._run_as(
                "conv-a",
                "print(globals().get('child_secret', 'ISOLATED'))",
                task_id="turn-2",
            )
        self.assertIn("ISOLATED", leak.get("output", ""), leak)
        self.assertIn("ISOLATED", back.get("output", ""), back)

    def test_two_delegated_children_are_isolated_from_each_other(self):
        """Sibling children in one batch must not share a kernel either —
        each child context carries its own delegation session id."""
        from agent.delegation_context import delegated_child_context

        with _kernel_config():
            with delegated_child_context("child-A"):
                self._run_as("conv-a", "sibling_secret = 'A'", task_id="t")
            with delegated_child_context("child-B"):
                peek = self._run_as(
                    "conv-a",
                    "print(globals().get('sibling_secret', 'ISOLATED'))",
                    task_id="t",
                )
        self.assertIn("ISOLATED", peek.get("output", ""), peek)

    def test_live_children_keep_their_kernels_past_the_lru_cap(self):
        """A fan-out wider than max_session_kernels used to evict LIVE children's kernels (each
        child's execute_code spawned a kernel, the cap reaped the oldest sibling's), so a child's
        second call hit NameError on state its first call had set — 48 NameErrors across 28 lanes,
        while the schema promised persistence. A live child's kernel is pinned for the child's life."""
        import contextvars

        from agent.delegation_context import delegated_child_context

        with _kernel_config(max_session_kernels=2):
            contexts = []
            for index in range(5):
                def _set(index=index):
                    with delegated_child_context(f"child-{index}"):
                        self._run_as("conv", f"v = {index}", task_id=f"child-{index}")
                ctx = contextvars.copy_context()
                ctx.run(_set)
                contexts.append(ctx)
            outcomes = {}
            for index, ctx in enumerate(contexts):
                def _read(index=index):
                    with delegated_child_context(f"child-{index}"):
                        outcomes[index] = self._run_as("conv", "print(v)", task_id=f"child-{index}")
                ctx.run(_read)
        for index, outcome in outcomes.items():
            self.assertEqual(outcome["status"], "success", outcome)
            self.assertTrue(outcome["kernel"]["reused"], outcome)
            self.assertIn(str(index), outcome["output"])

    def test_finished_children_release_their_kernels(self):
        """The pin is not a leak: when the child is torn down (the delegate_task cleanup path calls
        ``shutdown_kernels_for_delegated_child``) its kernels die and stop counting."""
        from agent.delegation_context import delegated_child_context
        from tools.code_kernel import shutdown_kernels_for_delegated_child

        with _kernel_config():
            with delegated_child_context("child-done"):
                self._run_as("conv", "v = 1", task_id="child-done")
            with delegated_child_context("child-live"):
                self._run_as("conv", "v = 2", task_id="child-live")
            doomed = [k for k in _KERNELS.values() if k.owner.endswith("::child::child-done")]
            self.assertEqual(len(doomed), 1)
            shutdown_kernels_for_delegated_child("child-done")
            self.assertEqual([k for k in _KERNELS.values() if k.owner.endswith("::child::child-done")], [])
            doomed[0].proc.wait(timeout=10)
            self.assertFalse(doomed[0].alive())
            # The sibling's kernel is untouched.
            with delegated_child_context("child-live"):
                still = self._run_as("conv", "print(v)", task_id="child-live")
        self.assertIn("2", still["output"])

    def test_session_clear_disposes_the_owners_kernels(self):
        from tools.approval import clear_session

        with _kernel_config():
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            self.assertEqual(len(_KERNELS), 1)
            kernel = next(iter(_KERNELS.values()))
            self.assertTrue(kernel.alive())
            clear_session("conv-a")
            self.assertEqual(len(_KERNELS), 0)
            kernel.proc.wait(timeout=10)
            self.assertFalse(kernel.alive())
            # The next turn in a cleared session starts fresh.
            after = self._run_as("conv-a", "print('x' in dir())", task_id="turn-2")
        self.assertEqual(after["status"], "success", after)
        self.assertIn("False", after["output"])

    def test_live_kernels_are_capped_lru_across_owners(self):
        with _kernel_config(max_session_kernels=2):
            kernels = []
            for index in range(4):
                self._run_as(f"conv-{index}", "x = 1", task_id=f"turn-{index}")
                kernels.append(list(_KERNELS.values()))
            self.assertLessEqual(len(_KERNELS), 2)
            live_owners = {key[0] for key in _KERNELS}
            # The two most recently used owners survive.
            self.assertEqual(live_owners, {"conv-2", "conv-3"})
        # Evicted kernels are actually dead, not orphaned.
        evicted = [
            kernel
            for snapshot in kernels
            for kernel in snapshot
            if kernel.key not in _KERNELS
        ]
        for kernel in evicted:
            kernel.proc.wait(timeout=10)
            self.assertFalse(kernel.alive())

    def test_idle_kernels_are_reaped(self):
        import time as time_module

        with _kernel_config(kernel_idle_timeout=1):
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            stale = next(iter(_KERNELS.values()))
            time_module.sleep(1.2)
            # Any owner's next call sweeps expired kernels process-wide.
            self._run_as("conv-b", "y = 1", task_id="turn-2")
            self.assertNotIn(stale.key, _KERNELS)
            stale.proc.wait(timeout=10)
            self.assertFalse(stale.alive())

    def test_parallel_cells_share_one_kernel_process(self):
        """Parallel cells for one owner race the first spawn. Each racer
        used to see proc=None as 'dead', replace the registry entry, and
        orphan the winner's process — 110 live kernels under a 4-capped
        process (Sep 2026). Every kernel process must stay registry-owned."""
        import subprocess
        import threading

        results = []
        with _kernel_config():
            def _cell():
                results.append(self._run_as("conv-a", "import time; time.sleep(0.3)", task_id="t"))
            threads = [threading.Thread(target=_cell) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual([r["status"] for r in results], ["success"] * 6)
        self.assertEqual(len(_KERNELS), 1)
        live = subprocess.run(
            ["pgrep", "-fc", "-P", str(os.getpid()), "hermes_kernel_runner"],
            capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(live, "1")


class TestPerCellRpcAuthority(unittest.TestCase):
    """Interpreter state persists across cells; RPC authority must not."""

    def _recorder(self, seen):
        def _handle(tool_name, tool_args, task_id=None):
            from tools.thread_context import _callback_api

            (get_approval, _set_a), *_rest = _callback_api()
            seen.append(
                {
                    "tool": tool_name,
                    "task_id": task_id,
                    "approval_cb": get_approval(),
                }
            )
            return json.dumps({"ok": True})

        return _handle

    def test_a_later_cells_rpc_runs_under_that_cells_authority(self):
        from tools.terminal_tool import set_approval_callback

        seen = []
        cell = "import hermes_tools\nhermes_tools.web_search(query='q')\n"
        with _kernel_config(), patch(
            "model_tools.handle_function_call", new=self._recorder(seen)
        ):
            def cb_one():
                return "one"

            def cb_two():
                return "two"

            set_approval_callback(cb_one)
            try:
                first = _run(cell)
                set_approval_callback(cb_two)
                second = _run(cell)
            finally:
                set_approval_callback(None)
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(second["status"], "success", second)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0]["approval_cb"], cb_one)
        self.assertIs(seen[1]["approval_cb"], cb_two)
        self.assertEqual(seen[0]["task_id"], "kernel-test")

    def test_cross_cell_alias_dispatches_under_the_current_cell(self):
        # Adversarial cross-cell dataflow: a callable captured in cell 1 and
        # invoked by an opaque global name in cell 2 still crosses the RPC
        # boundary — under cell 2's authority, allow-list, and budget — the
        # operative enforcement a per-script static scan cannot provide once
        # state persists (composition contract with the execute-code guard).
        from tools.terminal_tool import set_approval_callback

        seen = []
        with _kernel_config(), patch(
            "model_tools.handle_function_call", new=self._recorder(seen)
        ):
            def cb_one():
                return "one"

            def cb_two():
                return "two"

            set_approval_callback(cb_one)
            try:
                first = _run("import hermes_tools\nalias = hermes_tools.web_search\n")
                set_approval_callback(cb_two)
                second = _run("alias(query='q')\n")
            finally:
                set_approval_callback(None)
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(second["status"], "success", second)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0]["approval_cb"], cb_two)

    def test_a_settled_cells_authority_refuses_dispatch(self):
        from tools.code_kernel import CellAuthority

        authority = CellAuthority("turn-1")
        authority.retire()
        result = authority.dispatch("web_search", {"query": "q"})
        self.assertIn("No active execute_code cell", result)

    def test_each_cell_installs_a_fresh_authority(self):
        with _kernel_config():
            _run("x = 1")
            kernel = next(iter(_KERNELS.values()))
            first_authority = kernel.cell_authority
            self.assertFalse(first_authority.active)
            _run("y = 2")
            self.assertIsNot(kernel.cell_authority, first_authority)
            self.assertFalse(kernel.cell_authority.active)


class TestBackgroundIdleReaper(unittest.TestCase):
    """#117169: the idle sweep must not depend on the next kernel acquire — a host
    that stays alive but wedged (e.g. pids exhaustion fail-closing every tool call)
    never acquires again, so a background reaper reapplies the acquire-path criteria
    on its own schedule, and staging dirs that outlived a dead host are swept by age."""

    def _run_as(self, session_key, code, task_id, **kwargs):
        from tools.approval_context import reset_current_session_key, set_current_session_key

        token = set_current_session_key(session_key)
        try:
            return json.loads(execute_code(code, task_id=task_id, **kwargs))
        finally:
            reset_current_session_key(token)

    def test_reap_once_sweeps_idle_kernels_without_a_new_acquire(self):
        import time as time_module

        from tools.code_kernel import _reap_once

        with _kernel_config(kernel_idle_timeout=1):
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            stale = next(iter(_KERNELS.values()))
            time_module.sleep(1.2)
            # No conv-b acquire here: the reaper pass alone must retire the kernel.
            _reap_once()
            self.assertNotIn(stale.key, _KERNELS)
            stale.proc.wait(timeout=10)
            self.assertFalse(stale.alive())

    def test_reap_once_spares_attached_and_fresh_kernels(self):
        from tools.code_kernel import _reap_once

        with _kernel_config(kernel_idle_timeout=1):
            fresh = self._run_as("conv-fresh", "x = 1", task_id="turn-1")
            self.assertEqual(fresh["status"], "success", fresh)
            kernel = next(iter(_KERNELS.values()))
            kernel.attached += 1  # a cell is mid-flight: reaping must skip it
            try:
                _reap_once()
                self.assertIn(kernel.key, _KERNELS)
                self.assertTrue(kernel.alive())
            finally:
                kernel.attached -= 1

class TestStaleStagingDirSweep(unittest.TestCase):
    def test_week_old_kernel_dirs_go_and_fresh_ones_stay(self):
        import time as time_module

        from tools.code_kernel import _sweep_stale_staging_dirs

        with tempfile.TemporaryDirectory() as tmp:
            with patch("tools.code_kernel.tempfile.gettempdir", return_value=tmp):
                old = Path(tmp, "hermes_kernel_old")
                young = Path(tmp, "hermes_kernel_young")
                bystander = Path(tmp, "unrelated_dir")
                for path in (old, young, bystander):
                    path.mkdir()
                week_and_a_bit = time_module.time() - 8 * 86400
                os.utime(old, (week_and_a_bit, week_and_a_bit))
                removed = _sweep_stale_staging_dirs()
                # Asserted inside the TemporaryDirectory: cleanup would flatten everything.
                self.assertEqual(removed, 1)
                self.assertFalse(old.exists())
                self.assertTrue(young.exists())
                self.assertTrue(bystander.exists())
