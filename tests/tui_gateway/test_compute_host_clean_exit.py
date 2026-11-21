"""The ``python -m tui_gateway.compute_host`` child must exit cleanly while its control reader
is still parked on an open stdin.

The daemon ``compute-host-control-reader`` blocks in a stdin read that holds the
``BufferedReader`` lock. If the process then exits through normal interpreter finalization
(SIGTERM/SIGINT while the parent still holds the pipe open), closing ``sys.stdin`` dies with
``Fatal Python error: _enter_buffered_busy`` and the process aborts with -SIGABRT instead of
exiting 0, and the OS files a crash report.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FATAL = "Fatal Python error"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["sigterm", "sigint"])
def test_signal_with_reader_parked_exits_zero_without_fatal_error(tmp_path, signum):
    """The supervisor's terminate fallback: a signal while the parent still holds stdin open."""
    stderr_path = tmp_path / "stderr.txt"
    with stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.compute_host"], cwd=str(REPO), env=_env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
    try:
        assert proc.stdout is not None
        assert proc.stdin is not None
        hello = json.loads(proc.stdout.readline())
        assert hello["type"] == "hello" and hello["host_pid"] == proc.pid
        # A round trip proves the reader is up and back in its blocking stdin read.
        proc.stdin.write(b'{"type": "bogus", "request_id": "probe"}\n')
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline())["type"] == "error"
        proc.send_signal(signum)
        returncode = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()
    err = stderr_path.read_text(errors="replace")
    assert FATAL not in err, err
    assert returncode == 0, (returncode, err)


# A non-daemon thread standing in for a turn that outlives the drain. It touches the marker once
# the main thread has left ``run_host`` and the interpreter is joining it at exit.
_LINGERING_TURN = """
import runpy, threading, time
def turn():
    while threading.main_thread().is_alive():
        time.sleep(0.01)
    open({marker!r}, "w").close()
    time.sleep(120)
threading.Thread(target=turn, name="lingering-turn").start()
runpy.run_module("tui_gateway.compute_host", run_name="__main__")
"""


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["sigterm", "sigint"])
def test_signal_after_stdin_eof_is_not_ignored(tmp_path, signum):
    """Parent gone mid-turn (stdin EOF), then a stop signal: exit 0 instead of waiting out the turn."""
    from tests.waiting import wait_until

    marker = tmp_path / "joining"
    stderr_path = tmp_path / "stderr.txt"
    with stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            [sys.executable, "-c", _LINGERING_TURN.format(marker=str(marker))], cwd=str(REPO),
            env=_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
    try:
        assert proc.stdout is not None
        assert proc.stdin is not None
        assert json.loads(proc.stdout.readline())["type"] == "hello"
        proc.stdin.close()
        wait_until(marker.exists, timeout=30, what="the interpreter joining the lingering turn")
        assert proc.poll() is None
        proc.send_signal(signum)
        returncode = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()
    err = stderr_path.read_text(errors="replace")
    assert FATAL not in err, err
    assert returncode == 0, (returncode, err)
