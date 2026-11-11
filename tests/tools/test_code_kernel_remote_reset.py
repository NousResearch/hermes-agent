"""A ``reset=True`` cell must not kill a remote session kernel another cell is still running on.

Two ``execute_code`` calls for one owner on a remote backend: the first is mid-cell when the
second asks for a fresh kernel. The second gets a new kernel; the old one is torn down only once
the first cell has settled, as the local session kernel does.
"""

from __future__ import annotations

import os
import shutil
import threading

import pytest

from tests.waiting import wait_until
from tools.code_kernel_remote import execute_in_remote_kernel, shutdown_all_remote_kernels
from tools.environments.local import LocalEnvironment


def _open_writer(fifo: str):
    """Writer fd once the cell holds the read end open, else None."""
    try:
        return os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return None


@pytest.mark.platforms("linux", "macos")
@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("python3")), reason="needs bash and python3")
def test_reset_leaves_the_kernel_of_a_running_cell_alive_until_that_cell_settles(tmp_path):
    fifo = tmp_path / "cell_input"
    os.mkfifo(fifo)
    env = LocalEnvironment(cwd=str(tmp_path), timeout=60)
    first: dict = {}

    def run(code: str, *, reset: bool, timeout: int) -> dict:
        return execute_in_remote_kernel(
            code, env=env, env_type="ssh", task_env_id="owner", sandbox_tools=frozenset(),
            timeout=timeout, max_tool_calls=5, reset=reset)

    try:
        cell = threading.Thread(target=lambda: first.update(
            run(f"print(open({str(fifo)!r}).read())", reset=False, timeout=20)))
        cell.start()
        writer = wait_until(lambda: _open_writer(str(fifo)), timeout=30, what="first cell reading its input")
        os.write(writer, b"first cell finished")

        second = run("print('second cell')", reset=True, timeout=30)
        assert second["status"] == "success", second
        assert second["kernel"]["state_reset"] is True

        os.close(writer)
        cell.join(timeout=60)
        assert not cell.is_alive()
        assert (first["status"], first["stdout"].strip()) == ("success", "first cell finished")
    finally:
        shutdown_all_remote_kernels()
        env.cleanup()
