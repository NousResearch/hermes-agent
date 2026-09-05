"""Owned embedded-driver pipes/children: real subprocess lifecycle regressions."""
import json
import os
import subprocess
import sys
import threading

from tools.computer_use import cua_backend_daemon as daemon_module
from tools.computer_use import cua_backend_driver

import pytest

from tools.computer_use.cua_backend_daemon import _EmbeddedCuaDaemon


@pytest.mark.parametrize("output", [b"startup diagnostic\n", b"\xff\n"])
def test_stderr_reader_closes_its_pipe_on_eof_or_decode_error(output):
    daemon = _EmbeddedCuaDaemon(sys.executable, "unrestricted")
    process = subprocess.Popen(
        [sys.executable, "-c", f"import os; os.write(2, {output!r})"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
    )
    try:
        daemon._drain_stderr(process)
        process.wait(timeout=5)
        assert process.stderr.closed, "the daemon owns this pipe, including on reader errors"
        if output.isascii():
            assert list(daemon._stderr_tail) == [output.decode().strip()]
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stderr.close()


@pytest.mark.linux_only
@pytest.mark.parametrize("boundary", ["stop", "early_exit", "readiness_error", "thread_start_error",
                                     "stop_error", "timeout", "inherited_writer", "early_exit_stop_error"])
def test_lifecycle_reaps_owned_child_and_drains_diagnostics(tmp_path, monkeypatch, boundary):
    # An inert executable, never the host driver: control is via our private socket name.
    binary = tmp_path / "driver"
    log = tmp_path / "calls.jsonl"
    binary.write_text("#!" + sys.executable + "\n" + f'''
import json, pathlib, sys, time
args = sys.argv[1:]
with open({str(log)!r}, 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
if args[0] == 'serve':
    print('owned child diagnostic', file=sys.stderr, flush=True)
    if {boundary!r}.startswith('early_exit'):
        sys.exit(23)
    stopped = pathlib.Path(args[args.index('--socket') + 1] + '.stopped')
    while not stopped.exists():
        time.sleep(0.01)
elif args[0] == 'stop':
    pathlib.Path(args[args.index('--socket') + 1] + '.stopped').touch()
''')
    binary.chmod(0o700)
    daemon = _EmbeddedCuaDaemon(str(binary), "unrestricted")
    daemon.socket_path = str(tmp_path / "private.sock")
    monkeypatch.setattr(cua_backend_driver, "_resolve_mcp_invocation", lambda *a, **k: (str(binary), ["mcp"]))
    monkeypatch.setattr(cua_backend_driver, "_mcp_args_with_overlay_flag", lambda args, **k: args)
    original_popen = subprocess.Popen
    children = []
    held_writers = []

    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if kwargs.get("stderr") == subprocess.PIPE:
            children.append(child)
            if boundary == "inherited_writer":
                # Same EOF semantics as a descendant inheriting the child's stderr.
                held_writers.append(os.open(f"/proc/{child.pid}/fd/2", os.O_WRONLY))
        return child

    monkeypatch.setattr(subprocess, "Popen", spawn)
    release = threading.Event()
    readers = []
    original_thread = threading.Thread

    class ReaderThread(original_thread):
        def start(self):
            readers.append(self)
            if boundary == "thread_start_error":
                raise RuntimeError("reader launch refused")
            super().start()

        def join(self, timeout=None):
            assert timeout is not None and timeout > 0, "reader joins must be bounded"
            # Delay diagnostics until cleanup explicitly synchronizes with the reader.
            release.set()
            return super().join(timeout)

    monkeypatch.setattr(daemon_module.threading, "Thread", ReaderThread)
    original_drain = daemon._drain_stderr

    def delayed_drain(process):
        release.wait(10)
        original_drain(process)

    monkeypatch.setattr(daemon, "_drain_stderr", delayed_drain)

    def ready(env):
        if boundary.startswith("early_exit"):
            children[0].wait(timeout=5)
            return False
        if boundary == "readiness_error":
            raise RuntimeError("lease no longer valid")
        return True

    monkeypatch.setattr(daemon, "_socket_ready", ready)
    if boundary == "timeout":
        monkeypatch.setattr(daemon, "_START_TIMEOUT_SECONDS", 0)
    if boundary.endswith("stop_error"):
        original_quiet = daemon_module._cb()._run_quiet

        def fail_stop(argv, **kwargs):
            if argv[1] == "stop":
                raise RuntimeError("stop command failed")
            return original_quiet(argv, **kwargs)

        monkeypatch.setattr(daemon_module._cb(), "_run_quiet", fail_stop)
    expected = {"early_exit": "exited during startup: owned child diagnostic",
                "readiness_error": "lease no longer valid", "thread_start_error": "reader launch refused",
                "stop_error": "stop command failed", "timeout": "startup timed out: owned child diagnostic",
                "early_exit_stop_error": "exited during startup: owned child diagnostic"}
    stopper = None
    try:
        if boundary == "inherited_writer":
            daemon.start()
            stopped = threading.Event()
            errors = []

            def stop():
                try:
                    daemon.stop()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    stopped.set()

            stopper = original_thread(target=stop, daemon=True)
            stopper.start()
            assert stopped.wait(5), "stop hung waiting for inherited stderr EOF"
            assert not errors
            assert children[0].poll() is not None
            assert readers[0].is_alive(), "the held writer must actually delay EOF"
            os.close(held_writers.pop())
            readers[0].join(timeout=5)
        elif boundary in {"stop", "stop_error"}:
            daemon.start()
            if boundary == "stop_error":
                with pytest.raises(RuntimeError, match=expected[boundary]):
                    daemon.stop()
            else:
                daemon.stop()
        else:
            with pytest.raises(RuntimeError, match=expected[boundary]):
                daemon.start()
        assert children and children[0].poll() is not None, "failed startup must reap its own child"
        assert children[0].stderr.closed, "cleanup must synchronize with the owned pipe reader"
        assert not any(reader.is_alive() for reader in readers)
        assert daemon._process is None
        assert not daemon._running and not daemon._owns_runtime
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert all(call[call.index('--socket') + 1] == daemon.socket_path for call in calls)
        if boundary != "thread_start_error":
            assert list(daemon._stderr_tail) == ["owned child diagnostic"]
        daemon.stop()  # ownership is consumed; repeated cleanup never launches another child
        assert [json.loads(line) for line in log.read_text().splitlines()] == calls
    finally:
        release.set()
        for fd in held_writers:
            os.close(fd)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=5)
        if stopper is not None:
            stopper.join(timeout=5)
        for child in children:
            child.stderr.close()
