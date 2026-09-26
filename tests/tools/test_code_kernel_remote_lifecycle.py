"""Remote kernel disposal owns the whole cell process group, not just its children."""
import json
import shlex
import time

import psutil
import pytest

from tools import code_kernel_remote as remote
from tools.environments.local import LocalEnvironment


def _running(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("teardown", ["timeout", "reset", "shutdown", "exit", "idle", "removed_dir"])
@pytest.mark.parametrize("ignore_term", [False, True])
def test_remote_disposal_stops_descendants_and_preserves_other_owner(tmp_path, monkeypatch, teardown, ignore_term):
    env = LocalEnvironment(cwd=str(tmp_path))
    monkeypatch.setattr(env, "get_temp_dir", lambda: str(tmp_path))
    marker = tmp_path / "tree.json"
    ignore = "signal.signal(signal.SIGTERM, signal.SIG_IGN);" if ignore_term else ""
    worker = (
        "import os,time,json,signal;from pathlib import Path;" + ignore
        + f"Path({str(marker)!r}).write_text(json.dumps([os.getppid(),os.getpid()]),encoding='utf-8');"
        "time.sleep(120)"
    )
    parent = (
        "import subprocess,sys,time,signal;" + ignore
        + f"subprocess.Popen([sys.executable,'-c',{worker!r}]);time.sleep(120)"
    )
    code = "import subprocess,sys,signal;" + ignore + f"subprocess.Popen([sys.executable,'-c',{parent!r}])"
    runner_pids = []

    def run(code, owner="target", **kwargs):
        return remote.execute_in_remote_kernel(
            code, env=env, env_type="ssh", task_env_id=owner, sandbox_tools=frozenset(),
            timeout=kwargs.pop("timeout", 10), max_tool_calls=5, reset=kwargs.pop("reset", False),
            idle_exit=1 if teardown == "idle" and owner == "target" else 1800,
        )

    try:
        assert run("import sys; assert len(sys.argv) == 1; sentinel = 41", "other")["status"] == "success"
        assert run(code)["status"] == "success"
        runner_pids = [int(k.pid) for k in remote._REMOTE_KERNELS.values()]
        assert _wait_for(marker.exists), "descendant did not start"
        pids = json.loads(marker.read_text(encoding="utf-8-sig"))
        assert all(_running(pid) for pid in pids)
        if teardown == "timeout":
            assert run("import time;time.sleep(120)", timeout=2)["status"] == "timeout"
        elif teardown == "reset":
            assert run("print('reset')", reset=True)["kernel"]["state_reset"]
        elif teardown == "shutdown":
            remote.shutdown_remote_kernels_for_owner("target")
        elif teardown == "removed_dir":
            kernel = next(k for k in remote._REMOTE_KERNELS.values() if k.owner == "target")
            env.execute("rm -rf " + shlex.quote(kernel.kernel_dir))
            remote.shutdown_remote_kernels_for_owner("target")
        elif teardown == "exit":
            run("sys.exit(0)")
        assert _wait_for(lambda: not any(_running(pid) for pid in pids)), pids
        assert run("print(sentinel + 1)", "other")["stdout"].strip() == "42"
    finally:
        remote.shutdown_all_remote_kernels()
        if marker.exists():
            pids = json.loads(marker.read_text(encoding="utf-8-sig"))
            # The failing baseline reparents these test-owned workers to init.
            env.execute("kill -KILL " + " ".join(shlex.quote(str(pid)) for pid in pids + runner_pids) + " 2>/dev/null; true")
        env.cleanup()
