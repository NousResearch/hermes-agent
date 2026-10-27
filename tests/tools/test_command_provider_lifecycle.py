"""Finite command providers clean their owned group without changing native results."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import errno
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time

import psutil
import pytest

from tools import tts_command_provider as command_provider
from tools import tts_tool_lifecycle
from tools.tts_command_error_facts import (
    CommandOperation,
    CommandStream,
    command_failure_facts,
    set_command_failure_facts,
)
from tools.transcription_command import _transcribe_command_stt


_PROVIDER = '''
import json, os, pathlib, signal, subprocess, sys, time
import psutil
record, mode, output, input_path = sys.argv[1:]
record = pathlib.Path(record)
child_code = """
import pathlib, signal, sys
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text('ready', encoding='utf-8')
signal.pause()
"""
children = []
identities = {
    'parent': [os.getpid(), psutil.Process().create_time()],
    'group': os.getpgrp(),
}
def publish():
    staging = record.with_suffix('.tmp')
    staging.write_text(json.dumps(identities), encoding='utf-8')
    staging.replace(record)
publish()
for index, escaped in enumerate((False, True)):
    ready = str(record) + str(index)
    child = subprocess.Popen(
        [sys.executable, '-c', child_code, ready],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=escaped,
    )
    children.append(child)
    identities['escaped' if escaped else 'owned'] = [
        child.pid, psutil.Process(child.pid).create_time()
    ]
    publish()
    deadline = time.monotonic() + 5
    while not pathlib.Path(ready).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('fixture child did not become ready')
        time.sleep(0.01)
print('provider-out', flush=True)
print('provider-err', file=sys.stderr, flush=True)
record.with_suffix('.ready').write_text('ready', encoding='utf-8')
if output:
    data = b'ID3fixture' if mode == 'tts' else b'approved words'
    pathlib.Path(output).write_bytes(data)
if mode in ('timeout', 'interrupt'):
    signal.pause()
raise SystemExit(7 if mode in ('nonzero', 'warm-failure') else 0)
'''


@dataclass(frozen=True)
class _CommandFixture:
    script: Path
    record: Path

    def command(self, mode: str, output: str = "", input_path: str = "") -> str:
        return "exec " + " ".join(
            shlex.quote(str(value))
            for value in (sys.executable, self.script, self.record, mode, output, input_path)
        )

    def identities(self) -> dict:
        return json.loads(self.record.read_text(encoding="utf-8"))

    def running(self, key: str) -> bool:
        pid, created = self.identities()[key]
        try:
            proc = psutil.Process(pid)
            return proc.create_time() == created and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False


def _wait_until(predicate, timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


@contextmanager
def _delayed_readers(monkeypatch):
    ready = threading.Event()
    real_read = command_provider._CommandOutput._read_stream
    real_finish = command_provider._CommandOutput.finish
    timers = []

    def read(self, *args):
        if not ready.wait(10):
            raise RuntimeError("fixture reader was not released")
        real_read(self, *args)

    def finish(self, *args):
        if not timers:
            timer = threading.Timer(2, ready.set)
            timers.append(timer)
            timer.start()
        return real_finish(self, *args)

    monkeypatch.setattr(command_provider._CommandOutput, "_read_stream", read)
    monkeypatch.setattr(command_provider._CommandOutput, "finish", finish)
    try:
        yield
    finally:
        ready.set()
        for timer in timers:
            timer.cancel()
            timer.join(timeout=5)


class _DictionaryShadowOSError(OSError):
    @property
    def __dict__(self) -> dict[str, object]:
        raise AssertionError("native exception dictionary getter executed")


class _FalseyOSError(OSError):
    def __bool__(self) -> bool:
        return False


@contextmanager
def _faulty_reader(
    monkeypatch,
    *,
    name="stdout",
    close=False,
    read=True,
    late=False,
    shared=False,
    failure=None,
):
    if failure is None:
        failure = OSError(
            errno.EIO if read else errno.EBADF,
            "fixture pipe read failure" if read else "fixture pipe close failure",
        )
    closed = (
        failure if shared else OSError(errno.EBADF, "fixture secondary close failure")
    )
    released = threading.Event()
    real_read = command_provider._CommandOutput._read_stream
    real_receive = command_provider._CommandOutput.receive

    class FaultyPipe:
        encoding = "utf-8"

        def __init__(self, stream):
            self.stream = stream
            self.calls = 0
            if name == "stderr":
                self.buffer = self

        def read(self, size):
            self.calls += 1
            if read and self.calls > 1:
                raise failure
            return self.stream.read(1 if read else size)

        def read1(self, size):
            self.calls += 1
            if read and self.calls > 1:
                raise failure
            return self.stream.buffer.read1(1 if read else size)

        def close(self):
            self.stream.close()
            if close:
                raise closed if read else failure

    def read_stream(self, stream_name, stream):
        if stream_name == name:
            if late and not released.wait(10):
                raise RuntimeError("fixture faulty reader was not released")
            stream = FaultyPipe(stream)
        return real_read(self, stream_name, stream)

    def receive(self, *args):
        try:
            return real_receive(self, *args)
        finally:
            released.set()

    monkeypatch.setattr(command_provider._CommandOutput, "_read_stream", read_stream)
    monkeypatch.setattr(command_provider._CommandOutput, "receive", receive)
    try:
        yield failure, closed
    finally:
        released.set()


def _failure_facts(error):
    facts = command_failure_facts(error)
    if facts is None:
        return None
    return (
        facts.root_operation,
        facts.root_stream,
        tuple(
            (id(event.error), event.operation, event.stream)
            for event in facts.cleanup_events
        ),
    )


@contextmanager
def _fixture(tmp_path):
    script = tmp_path / "provider.py"
    script.write_text(_PROVIDER, encoding="utf-8")
    fixture = _CommandFixture(script, tmp_path / "processes.json")
    try:
        yield fixture
    finally:
        if fixture.record.exists():
            for key in ("parent", "owned", "escaped"):
                identity = fixture.identities().get(key)
                if identity is None:
                    continue
                pid, created = identity
                try:
                    proc = psutil.Process(pid)
                    if proc.create_time() == created:
                        proc.kill()
                        proc.wait(timeout=5)
                except psutil.NoSuchProcess:
                    pass


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "success-delayed-readers",
        "nonzero",
        "timeout",
        "interrupt",
        "interrupt-cleanup-error",
        "nonzero-cleanup-error",
        "timeout-cleanup-error",
        "reader-stdout-text",
        "reader-stderr-buffer",
        "reader-close",
        "reader-read-close",
        "reader-shadowed-storage",
        "reader-falsey-cleanup-error",
        "reader-after-exit",
        "reader-running",
        "reader-read-close-shared",
        "nonzero-reader-error",
        "timeout-reader-error",
        "interrupt-reader-error",
    ],
)
def test_runner_cleans_its_group_and_preserves_results(tmp_path, monkeypatch, mode):
    from contextlib import ExitStack

    with ExitStack() as stack:
        fixture = stack.enter_context(_fixture(tmp_path))
        if mode == "success-delayed-readers":
            stack.enter_context(_delayed_readers(monkeypatch))
        reader_failure = close_failure = None
        if mode.startswith("reader") or mode.endswith("-reader-error"):
            reader_failure, close_failure = stack.enter_context(
                _faulty_reader(
                    monkeypatch,
                    name="stderr" if mode == "reader-stderr-buffer" else "stdout",
                    failure={
                        "reader-shadowed-storage": _DictionaryShadowOSError(
                            errno.EIO, "fixture pipe read failure"
                        ),
                        "reader-falsey-cleanup-error": _FalseyOSError(
                            errno.EIO, "fixture pipe read failure"
                        ),
                    }.get(mode),
                    shared=mode == "reader-read-close-shared",
                    close=mode
                    in ("reader-close", "reader-read-close", "reader-read-close-shared")
                    or mode.endswith("-reader-error"),
                    read=mode != "reader-close",
                    late=mode == "reader-after-exit" or mode.endswith("-reader-error"),
                )
            )
        interrupting = mode.startswith("interrupt")
        native_mode = mode.removesuffix("-cleanup-error").removesuffix("-reader-error")
        if mode == "reader-running":
            native_mode = "timeout"
        command = fixture.command(native_mode)
        interrupted = KeyboardInterrupt("fixture cancellation")
        cleanup_failure = RuntimeError("fixture cleanup failure")
        real_get = queue.Queue.get

        def interrupt_after_ready(self, *args, **kwargs):
            if (
                kwargs.get("block", True)
                and fixture.record.with_suffix(".ready").exists()
            ):
                raise interrupted
            return real_get(self, *args, **kwargs)

        if interrupting:
            monkeypatch.setattr(queue.Queue, "get", interrupt_after_ready)
        if mode.endswith("-cleanup-error"):
            from tools.tts_command_process import CommandProcessScope

            real_close = CommandProcessScope.close

            def fail_after_cleanup(self, **kwargs):
                real_close(self, **kwargs)
                raise cleanup_failure

            monkeypatch.setattr(CommandProcessScope, "close", fail_after_cleanup)

        if mode.startswith("reader"):
            with pytest.raises(OSError) as caught:
                command_provider.run_command_provider(command, 5)
            assert caught.value is reader_failure
            expected_events = (
                ((id(close_failure), CommandOperation.CLOSE, CommandStream.STDOUT),)
                if mode in ("reader-read-close", "reader-read-close-shared")
                else ()
            )
            if mode == "reader-falsey-cleanup-error":
                expected_events = (
                    (id(cleanup_failure), CommandOperation.PROCESS_CLEANUP, None),
                )
            stream = (
                CommandStream.STDERR
                if mode == "reader-stderr-buffer"
                else CommandStream.STDOUT
            )
            operation = (
                CommandOperation.CLOSE
                if mode == "reader-close"
                else CommandOperation.READ
            )
            assert _failure_facts(caught.value) == (operation, stream, expected_events)
            assert (type(caught.value), caught.value.args, caught.value.errno) == (
                type(reader_failure),
                reader_failure.args,
                reader_failure.errno,
            )
            expected_notes = {
                "reader-read-close": ["Command-provider stdout close failed: OSError"],
                "reader-falsey-cleanup-error": [
                    "Command-provider process cleanup failed: RuntimeError"
                ],
            }.get(mode, [])
            assert getattr(caught.value, "__notes__", []) == expected_notes
        elif mode.startswith("success"):
            result = command_provider.run_command_provider(command, 5)
            expected = subprocess.CompletedProcess(
                command, 0, "provider-out\n", "provider-err\n"
            )
            assert (type(result), vars(result)) == (type(expected), vars(expected))
        elif interrupting:
            with pytest.raises(KeyboardInterrupt) as caught:
                command_provider.run_command_provider(command, 5, keep_children=True)
            assert caught.value is interrupted
            expected_events = {
                "interrupt": (),
                "interrupt-cleanup-error": (
                    (id(cleanup_failure), CommandOperation.PROCESS_CLEANUP, None),
                ),
                "interrupt-reader-error": (
                    (id(reader_failure), CommandOperation.READ, CommandStream.STDOUT),
                ),
            }[mode]
            assert _failure_facts(caught.value) == (
                (None, None, expected_events) if expected_events else None
            )
            if mode == "interrupt-cleanup-error":
                assert interrupted.__notes__ == [
                    "Command-provider process cleanup failed: RuntimeError"
                ]
            if mode == "interrupt-reader-error":
                assert interrupted.__notes__ == [
                    "Command-provider output cleanup failed: OSError"
                ]
        else:
            timing_out = mode.startswith("timeout")
            error = (
                subprocess.TimeoutExpired
                if timing_out
                else subprocess.CalledProcessError
            )
            with pytest.raises(error) as caught:
                command_provider.run_command_provider(
                    command, 2 if timing_out else 5, keep_children=timing_out
                )
            stdout = "p" if mode.endswith("-reader-error") else "provider-out\n"
            expected = (
                subprocess.CalledProcessError(
                    7, command, output=stdout, stderr="provider-err\n"
                )
                if mode.startswith("nonzero")
                else subprocess.TimeoutExpired(
                    command, 2, output=stdout, stderr="provider-err\n"
                )
            )
            if mode.endswith("-cleanup-error"):
                expected.add_note(
                    "Command-provider process cleanup failed: RuntimeError"
                )
            if mode.endswith("-reader-error"):
                expected.add_note("Command-provider output cleanup failed: OSError")
            expected_events = ()
            if mode.endswith("-cleanup-error"):
                expected_events = (
                    (id(cleanup_failure), CommandOperation.PROCESS_CLEANUP, None),
                )
            if mode.endswith("-reader-error"):
                expected_events = (
                    (id(reader_failure), CommandOperation.READ, CommandStream.STDOUT),
                )
                assert _failure_facts(reader_failure) == (
                    CommandOperation.READ,
                    CommandStream.STDOUT,
                    (
                        (
                            id(close_failure),
                            CommandOperation.CLOSE,
                            CommandStream.STDOUT,
                        ),
                    ),
                )
            operation = (
                CommandOperation.NATIVE_TIMEOUT
                if timing_out
                else CommandOperation.NATIVE_EXIT
            )
            assert _failure_facts(caught.value) == (operation, None, expected_events)
            facts = command_failure_facts(caught.value)
            assert facts is not None
            set_command_failure_facts(expected, facts)
            assert (type(caught.value), vars(caught.value)) == (
                type(expected),
                vars(expected),
            )

        assert _wait_until(lambda: not fixture.running("owned")), (
            "owned group child leaked"
        )
        assert (fixture.running("parent"), fixture.running("escaped")) == (False, True)


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "path", ["tts", "stt", "warm", "release", "warm-failure", "warm-reader-error"]
)
def test_native_paths_cleanup_and_successful_warm_keeps_its_service(
    tmp_path, monkeypatch, path
):
    from contextlib import ExitStack

    with ExitStack() as stack:
        fixture = stack.enter_context(_fixture(tmp_path))
        reader_failure = (
            stack.enter_context(_faulty_reader(monkeypatch, late=True))[0]
            if path == "warm-reader-error"
            else None
        )
        config = {
            "command": fixture.command(path, "{output_path}", "{input_path}"),
            "timeout": 5,
            "normalize": False,
        }
        if path == "tts":
            output = tmp_path / "speech.mp3"
            result = command_provider._generate_command_tts(
                "approved words", str(output), "fixture", config, {}
            )
            assert (result, output.read_bytes()) == (str(output), b"ID3fixture")
        elif path == "stt":
            audio = tmp_path / "audio.wav"
            audio.write_bytes(b"RIFFfixture")
            assert _transcribe_command_stt(str(audio), "fixture", config, {}) == {
                "success": True,
                "transcript": "approved words",
                "provider": "fixture",
            }
        else:
            hook = "release" if path == "release" else "warm"
            config[f"{hook}_command"] = fixture.command(path)
            done = threading.Event()
            errors = []
            real_run = command_provider.run_command_provider

            def track_run(*args, **kwargs):
                try:
                    return real_run(*args, **kwargs)
                except OSError as exc:
                    errors.append(exc)
                    raise
                finally:
                    done.set()

            monkeypatch.setattr(command_provider, "run_command_provider", track_run)
            assert (
                tts_tool_lifecycle._signal_user_tts_provider(
                    "fixture", {"providers": {"fixture": config}}, hook
                )
                == hook
            )
            assert done.wait(10), "native hook did not finish"
            assert errors == ([reader_failure] if reader_failure else [])
            if reader_failure is not None:
                assert _failure_facts(reader_failure) == (
                    CommandOperation.READ,
                    CommandStream.STDOUT,
                    (),
                )

        expected = path in ("warm", "warm-failure")
        assert _wait_until(lambda: fixture.running("owned") == expected), (
            "native group child leaked"
        )
        assert (fixture.running("parent"), fixture.running("escaped")) == (False, True)
