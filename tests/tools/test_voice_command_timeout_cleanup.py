"""A command-provider timeout must retire the workers that still own its pipes."""

import shlex
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from tools.tts_command_provider import run_command_provider


def _assert_timeout_retires_worker(tmp_path: Path, launcher_exits: bool) -> None:
    stop = tmp_path / "stop"
    worker_pid = tmp_path / "worker.pid"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(worker_pid)!r}).write_text(str(os.getpid()))\n"
        "print('worker ready', file=sys.stderr, flush=True)\n"
        "deadline = time.monotonic() + 30\n"
        f"while not Path({str(stop)!r}).exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import subprocess, sys\n"
        f"child = subprocess.Popen([sys.executable, {str(worker)!r}])\n"
        + ("" if launcher_exits else "child.wait()\n"),
        encoding="utf-8",
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired) as failure:
            run_command_provider(shlex.join([sys.executable, str(launcher)]), timeout=2)
        assert "worker ready" in str(failure.value.stderr or "")
        pid = int(worker_pid.read_text(encoding="utf-8"))
        try:
            child = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        # Cooperatively retire our worker even when testing the unfixed implementation.
        stop.touch()
        if worker_pid.exists():
            try:
                psutil.Process(int(worker_pid.read_text(encoding="utf-8"))).wait(
                    timeout=5
                )
            except psutil.NoSuchProcess:
                pass


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "launcher_exits", [True, False], ids=["exited-launcher", "live-launcher"]
)
def test_linux_timeout_retires_worker(tmp_path, launcher_exits):
    # Real signals target only the test's owned process group; its exited leader
    # cannot be found by the live-system guard's parent-chain walk.
    _assert_timeout_retires_worker(tmp_path, launcher_exits)


@pytest.mark.macos_only
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "launcher_exits", [True, False], ids=["exited-launcher", "live-launcher"]
)
def test_macos_timeout_retires_worker(tmp_path, launcher_exits):
    _assert_timeout_retires_worker(tmp_path, launcher_exits)
