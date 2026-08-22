"""Linux integration test for the start-time guard on
``ProcessRegistry._terminate_host_pid``.

Unlike ``tests/tools/test_browser_orphan_reaper.py`` (which mocks the process
lifecycle), this exercises the REAL start-time revalidation and signaling path
against a live, disposable child process:

  * a *mismatched* kernel-start fingerprint must never terminate the PID
    (protection against a recycled PID number landing on an unrelated process), and
  * a *matching* fingerprint must follow the intended cleanup path.
"""

import subprocess
import sys
import time

import pytest

from tools.process_registry import ProcessRegistry

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason=(
        "Real-process start-time reaping is /proc-specific "
        "(get_process_start_time reads /proc/<pid>/stat)"
    ),
)


@pytest.fixture
def disposable_child():
    """Factory yielding real, harmless, long-sleeping child processes.

    Every spawned child is force-killed and reaped on teardown, so no process
    leaks and no zombie survives even when a test asserts and fails partway.
    """
    spawned: list[subprocess.Popen] = []

    def _spawn() -> subprocess.Popen:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"]
        )
        spawned.append(child)
        # Wait until the kernel has the process registered so /proc start-time
        # is readable before the test captures the fingerprint.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if ProcessRegistry._is_host_pid_alive(child.pid):
                break
            time.sleep(0.02)
        return child

    yield _spawn

    for child in spawned:
        try:
            if child.poll() is None:
                child.kill()
        except OSError:
            pass
        try:
            child.wait(timeout=5)
        except Exception:
            pass


def _wait_terminated(child: subprocess.Popen, timeout: float = 5.0) -> bool:
    """Return True once the child has exited, else False at timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return True
        time.sleep(0.05)
    return child.poll() is not None


def test_matching_start_time_terminates_child(disposable_child):
    """A matching kernel-start fingerprint follows the cleanup path and kills it."""
    child = disposable_child()
    real = ProcessRegistry._safe_host_start_time(child.pid)
    assert real is not None  # sanity: procfs handed us a real fingerprint

    ProcessRegistry._terminate_host_pid(child.pid, expected_start=real)

    assert _wait_terminated(child), (
        "matching fingerprint must terminate the child via the cleanup path"
    )


def test_mismatched_start_time_refuses_to_terminate(disposable_child):
    """A mismatched fingerprint must never signal the PID (recycled-PID guard)."""
    child = disposable_child()
    real = ProcessRegistry._safe_host_start_time(child.pid)
    assert real is not None

    # ``real + 1`` is guaranteed different: a process's /proc start-time is fixed
    # for its whole lifetime, so this can never collide with the true value.
    ProcessRegistry._terminate_host_pid(child.pid, expected_start=real + 1)

    assert child.poll() is None, "a mismatched fingerprint must NOT signal the child"
    time.sleep(0.3)
    assert child.poll() is None, (
        "child must still be alive after a beat — no signal was sent"
    )


def test_identity_helper_matches_real_and_rejects_wrong(disposable_child):
    """``_host_pid_is_ours`` accepts the true fingerprint and rejects a wrong one."""
    child = disposable_child()
    real = ProcessRegistry._safe_host_start_time(child.pid)
    assert real is not None

    assert ProcessRegistry._host_pid_is_ours(child.pid, real) is True
    assert ProcessRegistry._host_pid_is_ours(child.pid, real + 1) is False
