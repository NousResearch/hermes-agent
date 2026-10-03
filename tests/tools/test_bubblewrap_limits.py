"""Tests for the rlimits BubblewrapEnvironment applies through the prlimit
prefix in front of bwrap: RLIMIT_AS, RLIMIT_CPU and RLIMIT_NPROC from
terminal.bubblewrap_memory_mb, _cpu_seconds and _max_procs.

On kernel 6.8.0 with bwrap 0.9.0, RLIMIT_NPROC is counted per uid
host-wide inside the bwrap user namespace too, and it counts threads. With
the limit set to 5 and 192 processes on the uid, bwrap failed with
"Creating new namespace failed: Resource temporarily unavailable", the
same as a plain fork outside bwrap; a limit of processes + 256 failed the
same way because the uid ran about 2000 threads (193 processes). A fixed default would
therefore break every spawn on a desktop with more threads than the
limit, so max_procs is applied on top of the uid's current thread count
and bounds what the sandbox may add. The default stays 256.

Unit tests never spawn bwrap. Integration tests are skipped as a module
when bwrap is missing or its runtime probe fails, so CI without bwrap
stays green.
"""

import inspect
import os
import resource
import shutil
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments import bubblewrap
from tools.environments import local as local_mod
from tools.environments.bubblewrap import (
    SANDBOX_BASE_PROCESSES,
    BubblewrapConfig,
    BubblewrapEnvironment,
    prlimit_args,
    rlimit_values,
)
from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _bwrap_probe_passed(monkeypatch):
    """Unit constructions never spawn: count the process-wide bwrap probe as passed."""
    monkeypatch.setattr(bubblewrap, "_probed_bwrap_path", shutil.which("bwrap") or "/usr/bin/bwrap")
    monkeypatch.setattr(bubblewrap, "_process_limit_scoped", True)


def _bwrap_usable() -> bool:
    if shutil.which("bwrap") is None:
        return False
    try:
        probe = subprocess.run(
            ["bwrap", "--unshare-user", "--ro-bind", "/", "/", "true"],
            capture_output=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


BWRAP_USABLE = _bwrap_usable()
needs_bwrap = pytest.mark.skipif(not BWRAP_USABLE, reason="bwrap missing or its namespace probe failed")

MB = 1024 * 1024
PRLIMIT = "/usr/bin/prlimit"


@pytest.fixture
def sandbox_root(tmp_path, monkeypatch):
    root = tmp_path / "sandboxes"
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(root))
    return root


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


def _no_session():
    return patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None)


class TestRlimitValues:
    def test_defaults_map_to_the_three_limits(self):
        limits = rlimit_values(BubblewrapConfig())
        assert limits == {
            resource.RLIMIT_AS: 256 * MB,
            resource.RLIMIT_CPU: 30,
            resource.RLIMIT_NPROC: 256 + SANDBOX_BASE_PROCESSES,
        }

    def test_max_procs_counts_only_the_wrapper_processes_on_top(self):
        # The limit is set inside the sandbox, where the kernel counts the
        # processes of that sandbox alone: bwrap's init and the shell that
        # runs the command come on top of max_procs, the host does not.
        limits = rlimit_values(BubblewrapConfig(max_procs=4))
        assert limits[resource.RLIMIT_NPROC] == 4 + SANDBOX_BASE_PROCESSES
        assert 1 <= SANDBOX_BASE_PROCESSES <= 4

    @pytest.mark.parametrize("key, res", [
        ("memory_mb", resource.RLIMIT_AS),
        ("cpu_seconds", resource.RLIMIT_CPU),
        ("max_procs", resource.RLIMIT_NPROC),
    ])
    def test_zero_leaves_that_limit_out(self, key, res):
        limits = rlimit_values(BubblewrapConfig(**{key: 0}))
        assert res not in limits
        assert len(limits) == 2

    def test_all_zero_gives_no_prefix(self):
        limits = rlimit_values(BubblewrapConfig(memory_mb=0, cpu_seconds=0, max_procs=0))
        assert limits == {}
        assert prlimit_args(limits, PRLIMIT) == []

    def test_no_process_limit_is_derived_from_a_count_of_host_threads(self):
        source = inspect.getsource(bubblewrap)
        assert not hasattr(bubblewrap, "uid_thread_count")
        assert '"task"' not in source
        assert "uid_threads" not in source


class TestPrlimitPrefix:
    """The limits ride the argv as a prlimit(1) prefix; no Python runs in the
    forked child."""

    @pytest.fixture
    def unlimited(self, monkeypatch):
        monkeypatch.setattr(bubblewrap.resource, "getrlimit",
                            lambda res: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    def test_prefix_names_each_limit_once(self, unlimited):
        argv = prlimit_args({resource.RLIMIT_AS: 5 * MB, resource.RLIMIT_CPU: 7, resource.RLIMIT_NPROC: 300}, PRLIMIT)
        assert argv == [PRLIMIT, f"--as={5 * MB}", "--cpu=7", "--nproc=300"]

    def test_prefix_clamps_to_the_inherited_hard_limit(self, monkeypatch):
        monkeypatch.setattr(bubblewrap.resource, "getrlimit", lambda res: (10, 20))
        argv = prlimit_args({resource.RLIMIT_NPROC: 300, resource.RLIMIT_CPU: 7}, PRLIMIT)
        assert argv == [PRLIMIT, "--nproc=20", "--cpu=7"]

    @pytest.mark.skipif(shutil.which("prlimit") is None, reason="needs prlimit (util-linux)")
    def test_prlimit_sets_soft_and_hard_and_stops_at_the_command(self):
        # One value per flag sets soft and hard alike, and option parsing
        # ends at the command, so the bwrap argv follows with no separator
        # and its own options are left alone.
        probe = subprocess.run(
            [shutil.which("prlimit"), "--cpu=7", "sh", "-c", "ulimit -St; ulimit -Ht"],
            capture_output=True, text=True, timeout=10,
        )
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.split() == ["7", "7"]

    def test_environment_sets_memory_and_cpu_outside_and_the_process_limit_inside(self, sandbox_root, work_dir, unlimited):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        argv = env._wrap_popen_args(["bash"])
        assert argv[:3] == [env._prlimit_path, f"--as={256 * MB}", "--cpu=30"]
        assert argv[3] == env._bwrap_path
        # After bwrap's own separator: the process limit, then the command.
        nproc = 256 + SANDBOX_BASE_PROCESSES
        assert argv[-4:] == ["--", env._prlimit_path, f"--nproc={nproc}", "bash"]
        assert not any(a.startswith("--nproc") for a in argv[:argv.index(env._bwrap_path)])

    def test_environment_prefix_skips_zeroed_keys(self, sandbox_root, work_dir, unlimited):
        config = BubblewrapConfig(memory_mb=1024, cpu_seconds=0, max_procs=0)
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=config)
        argv = env._wrap_popen_args(["bash"])
        assert argv[:2] == [env._prlimit_path, f"--as={1024 * MB}"]
        assert argv[2] == env._bwrap_path
        assert argv[-2:] == ["--", "bash"]
        assert not any(a.startswith("--nproc") for a in argv)

    def test_environment_has_no_prefix_when_every_key_is_zero(self, sandbox_root, work_dir):
        config = BubblewrapConfig(memory_mb=0, cpu_seconds=0, max_procs=0)
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=config)
        argv = env._wrap_popen_args(["bash"])
        assert argv[0] == env._bwrap_path
        assert argv[-2:] == ["--", "bash"]

    def test_nothing_runs_between_fork_and_exec(self):
        # Popen's preexec_fn is unsafe in a threaded process (the gateway is
        # one); the backend and the local base must never pass one.
        assert "preexec_fn" not in inspect.getsource(bubblewrap)
        assert "preexec_fn" not in inspect.getsource(local_mod)


FORK_SCRIPT = """
import os, sys, time
mode, n = sys.argv[1], int(sys.argv[2])
started = failed = 0
kids = []
for _ in range(n):
    try:
        pid = os.fork()
    except BlockingIOError:
        failed += 1
        continue
    if pid == 0:
        if mode == "concurrent":
            time.sleep(2)
        os._exit(0)
    started += 1
    if mode == "sequential":
        os.waitpid(pid, 0)
    else:
        kids.append(pid)
for pid in kids:
    os.waitpid(pid, 0)
print(f"started={started} failed={failed}")
"""


def _uid_threads() -> int:
    """Threads this uid runs on the host, for the test's own precondition."""
    count = 0
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                if os.stat(f"/proc/{name}").st_uid == os.getuid():
                    count += len(os.listdir(f"/proc/{name}/task"))
            except OSError:
                continue
    return count


def _wait_for(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except OSError:
            pass
        time.sleep(0.05)
    return False


@needs_bwrap
class TestLimitsIntegration:
    @pytest.fixture
    def make_env(self, sandbox_root, work_dir):
        envs = []

        def factory(**config):
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=BubblewrapConfig(**config))
            envs.append(env)
            return env

        try:
            yield factory
        finally:
            for env in envs:
                env.cleanup()

    @pytest.fixture
    def fork_script(self, work_dir):
        path = work_dir / "forks.py"
        path.write_text(FORK_SCRIPT)
        return path

    def test_memory_default_denies_400mb_and_1024mb_allows_it(self, make_env):
        alloc = "python3 -c 'bytearray(400*1024*1024)'"
        result = make_env().execute(alloc)
        assert result["returncode"] != 0
        assert "MemoryError" in result["output"]
        result = make_env(memory_mb=1024).execute(alloc)
        assert result["returncode"] == 0, result["output"]

    def test_cpu_seconds_ends_a_spinning_command_with_a_signal(self, make_env):
        env = make_env(cpu_seconds=2)
        start = time.monotonic()
        result = env.execute("yes > /dev/null", timeout=30)
        elapsed = time.monotonic() - start
        assert elapsed < 10, elapsed
        # Soft and hard are equal, so the kernel checks the hard limit first
        # and sends SIGKILL (bash reports 128 + 9) rather than SIGXCPU.
        assert result["returncode"] > 128, result

    def test_max_procs_default_lets_300_short_children_run(self, make_env, fork_script):
        result = make_env().execute(f"python3 {fork_script} sequential 300")
        assert result["returncode"] == 0, result["output"]
        assert result["output"].strip() == "started=300 failed=0"

    def test_max_procs_is_a_ceiling_for_the_sandbox_alone(self, make_env, fork_script):
        # The uid runs far more than 20 threads on the host (this test
        # runner alone does), and the command still starts: the count is
        # the sandbox's own.
        assert _uid_threads() > 20
        result = make_env(max_procs=20).execute(f"python3 {fork_script} concurrent 100")
        assert result["returncode"] == 0, result["output"]
        counts = dict(part.split("=") for part in result["output"].split())
        assert 1 <= int(counts["started"]) <= 20, counts
        assert int(counts["started"]) + int(counts["failed"]) == 100, counts

    def test_host_threads_that_exit_give_the_sandbox_no_extra_room(self, make_env, fork_script):
        # 200 threads of the same uid exist when the sandbox starts and are
        # gone before it forks. A limit taken from a host count at spawn
        # would hand that room to the command.
        holder = subprocess.Popen([
            "python3", "-c",
            "import threading, time\n"
            "for _ in range(200): threading.Thread(target=time.sleep, args=(120,), daemon=True).start()\n"
            "time.sleep(120)",
        ])
        try:
            assert _wait_for(lambda: len(os.listdir(f"/proc/{holder.pid}/task")) > 200, 10)
            env = make_env(max_procs=20)
            release = threading.Timer(0.7, holder.kill)
            release.start()
            try:
                result = env.execute(f"sleep 2; python3 {fork_script} concurrent 100")
            finally:
                release.cancel()
            assert holder.wait(timeout=10) is not None
        finally:
            holder.kill()
            holder.wait()
        assert result["returncode"] == 0, result["output"]
        counts = dict(part.split("=") for part in result["output"].split())
        assert 1 <= int(counts["started"]) <= 20, counts

    def test_process_limit_cannot_be_raised_from_inside(self, make_env):
        env = make_env(max_procs=20)
        result = env.execute("ulimit -u 5000")
        assert result["returncode"] != 0
        assert env.execute("ulimit -u")["output"].strip() == str(20 + SANDBOX_BASE_PROCESSES)
        assert env.execute("ulimit -Hu")["output"].strip() == str(20 + SANDBOX_BASE_PROCESSES)

    def test_max_procs_zero_disables_the_process_limit(self, make_env, fork_script):
        result = make_env(max_procs=0).execute(f"python3 {fork_script} concurrent 100")
        assert result["returncode"] == 0, result["output"]
        assert result["output"].strip() == "started=100 failed=0"
