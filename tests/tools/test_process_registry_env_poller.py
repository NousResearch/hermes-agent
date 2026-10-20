"""Sandbox-log poller (non-local backends): reading the liveness probe's result.

The probe ends in `echo $?`, so it exits 0 whenever it ran. A non-zero result (an SDK backend
error, SSH or container down, a probe that timed out) says nothing about the process; reading its
text as "exited" reported exit code -1 for a process that was still running. Such a probe is
ignored and the next poll probes again.
"""

import time
from unittest.mock import patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


def _run_poller(probe_results):
    """Drive the poller with a scripted liveness probe; returns (session, number of probes sent)."""
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_probe", command="sleep 600", task_id="t1", started_at=time.time())
    results = iter(probe_results)
    probes = []
    gone = [False]  # the wrapper writes its exit file only once the process has exited

    class FakeEnv:
        def execute(self, command, **kwargs):
            if command.startswith("kill -0"):
                probes.append(command)
                result = next(results)
                gone[0] = gone[0] or result is _GONE
                return result
            if command.startswith("cat "):
                return {"output": "3\n", "returncode": 0} if gone[0] else {"output": "", "returncode": 1}
            return {"output": "0 0\n", "returncode": 0}  # log delta: nothing new

    with patch("tools.process_registry.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
        registry._env_poller_loop(session, FakeEnv(), "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit")
    return session, len(probes)


_GONE = {"output": "1\n", "returncode": 0}  # the probe ran: the process is gone


@pytest.mark.parametrize("failed_probe", [
    {"output": "[backend error: ConnectionError: reset by peer]", "returncode": 1},
    {"output": "[Command timed out after 5s]", "returncode": 124},
    {"output": "ssh: connect to host sandbox port 22: Connection refused", "returncode": 255},
], ids=["backend_error", "probe_timeout", "ssh_down"])
def test_a_probe_that_did_not_run_is_retried(failed_probe):
    session, probes = _run_poller([failed_probe] * 8 + [_GONE])
    assert probes == 9
    assert session.exited is True and session.exit_code == 3
    assert session.completion_reason != "lost"
