"""Real foreground and background pipe regressions for #94928."""

import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import pytest

from tools.environments.base_output import _BoundedOutputCollector, _drain_stdout
from tools.process_registry import ProcessRegistry, ProcessSession


@contextmanager
def _output_pipe(min_fd, *, hold_writer):
    import fcntl
    import resource

    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] <= min_fd:
        pytest.skip(f"descriptor limit cannot accommodate fd {min_fd}")
    read_fd, write_fd = os.pipe()
    try:
        if min_fd:
            elevated = fcntl.fcntl(read_fd, fcntl.F_DUPFD_CLOEXEC, min_fd)
            os.close(read_fd)
            read_fd = elevated
        assert read_fd >= min_fd
        with os.fdopen(read_fd, "r", encoding="utf-8") as stream:
            read_fd = None
            with subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdout.write('pipe-α\\n' * 500)"],
                stdin=subprocess.DEVNULL,
                stdout=write_fd,
                stderr=subprocess.DEVNULL,
            ) as proc:
                proc.stdout = stream
                if not hold_writer:
                    os.close(write_fd)
                    write_fd = None
                try:
                    yield proc
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=5)
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


def _read_output(proc, reader):
    if reader == "foreground":
        output = _BoundedOutputCollector(100_000)
        def target():
            _drain_stdout(proc, output)
        result = output.render
    else:
        registry = ProcessRegistry()
        session = ProcessSession(
            id="pipe-output", command="pipe output", task_id="test", started_at=time.time(),
            process=proc, pid=proc.pid,
        )
        registry._running[session.id] = session
        def target():
            registry._reader_loop(session)

        def result():
            return session.output_buffer
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "reader did not finish after the direct child exited"
    assert proc.wait(timeout=5) == 0
    return result()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("reader", ["foreground", "background"])
@pytest.mark.parametrize("min_fd", [0, 1050])
def test_pipe_output_survives_high_descriptors(reader, min_fd):
    with _output_pipe(min_fd, hold_writer=False) as proc:
        assert _read_output(proc, reader) == "pipe-α\n" * 500


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("reader", ["foreground", "background"])
def test_high_descriptor_drain_stops_with_an_inherited_writer(reader):
    # retain a writer after child exit to exercise the orphan-pipe idle bound.
    with _output_pipe(1050, hold_writer=True) as proc:
        assert _read_output(proc, reader) == "pipe-α\n" * 500
