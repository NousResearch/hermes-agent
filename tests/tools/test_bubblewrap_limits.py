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
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments import bubblewrap
from tools.environments import local as local_mod
from tools.environments.bubblewrap import (
    WRAPPER_PROCESS_ALLOWANCE,
    BubblewrapConfig,
    BubblewrapEnvironment,
    prlimit_args,
    rlimit_values,
    uid_thread_count,
)
from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _bwrap_probe_passed(monkeypatch):
    """Unit constructions never spawn: count the process-wide bwrap probe as passed."""
    monkeypatch.setattr(bubblewrap, "_probed_bwrap_path", shutil.which("bwrap") or "/usr/bin/bwrap")


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
# Threads the uid may gain or lose between the /proc scan and the
# forks of the concurrent test (the parallel runner spawns sandboxes under
# the same uid); the ceiling assertion tolerates this much movement.
NPROC_DRIFT = 128
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
        limits = rlimit_values(BubblewrapConfig(), uid_threads=100)
        assert limits == {
            resource.RLIMIT_AS: 256 * MB,
            resource.RLIMIT_CPU: 30,
            resource.RLIMIT_NPROC: 100 + 256 + WRAPPER_PROCESS_ALLOWANCE,
        }

    def test_max_procs_gets_the_wrapper_allowance_on_top_of_the_uid_count(self):
        # bwrap, its init, bash and its subshells fork before the command
        # runs; the allowance keeps a tight max_procs from starving them.
        limits = rlimit_values(BubblewrapConfig(max_procs=4), uid_threads=100)
        assert limits[resource.RLIMIT_NPROC] == 100 + 4 + WRAPPER_PROCESS_ALLOWANCE
        assert 4 <= WRAPPER_PROCESS_ALLOWANCE <= 64

    @pytest.mark.parametrize("key, res", [
        ("memory_mb", resource.RLIMIT_AS),
        ("cpu_seconds", resource.RLIMIT_CPU),
        ("max_procs", resource.RLIMIT_NPROC),
    ])
    def test_zero_leaves_that_limit_out(self, key, res):
        limits = rlimit_values(BubblewrapConfig(**{key: 0}), uid_threads=100)
        assert res not in limits
        assert len(limits) == 2

    def test_all_zero_gives_no_prefix(self):
        limits = rlimit_values(BubblewrapConfig(memory_mb=0, cpu_seconds=0, max_procs=0), uid_threads=100)
        assert limits == {}
        assert prlimit_args(limits, PRLIMIT) == []

    def test_uid_thread_count_covers_this_process_and_its_threads(self):
        count = uid_thread_count(os.getuid())
        own_threads = len(os.listdir("/proc/self/task"))
        assert count >= own_threads >= 1


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

    def test_environment_prefix_applies_defaults_over_the_uid_count(self, sandbox_root, work_dir, unlimited, monkeypatch):
        monkeypatch.setattr(bubblewrap, "uid_thread_count", lambda uid: 100)
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        argv = env._wrap_popen_args(["bash"])
        nproc = 100 + 256 + WRAPPER_PROCESS_ALLOWANCE
        assert argv[:4] == [env._prlimit_path, f"--as={256 * MB}", "--cpu=30", f"--nproc={nproc}"]
        assert argv[4] == env._bwrap_path
        assert argv[-1] == "bash"

    def test_environment_prefix_skips_zeroed_keys(self, sandbox_root, work_dir, unlimited, monkeypatch):
        counted = []
        monkeypatch.setattr(bubblewrap, "uid_thread_count", lambda uid: counted.append(uid) or 100)
        config = BubblewrapConfig(memory_mb=1024, cpu_seconds=0, max_procs=0)
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=config)
        argv = env._wrap_popen_args(["bash"])
        assert argv[:2] == [env._prlimit_path, f"--as={1024 * MB}"]
        assert argv[2] == env._bwrap_path
        assert counted == []  # /proc is not scanned when max_procs is 0

    def test_environment_has_no_prefix_when_every_key_is_zero(self, sandbox_root, work_dir):
        config = BubblewrapConfig(memory_mb=0, cpu_seconds=0, max_procs=0)
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=config)
        argv = env._wrap_popen_args(["bash"])
        assert argv[0] == env._bwrap_path

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
        # Sequential forks never exceed the uid thread count by more than a few,
        # so they all succeed while max_procs sits on top of that count.
        result = make_env().execute(f"python3 {fork_script} sequential 300")
        assert result["returncode"] == 0, result["output"]
        assert result["output"].strip() == "started=300 failed=0"

    def test_max_procs_bounds_what_the_sandbox_adds(self, make_env, fork_script):
        # max_procs=64 leaves room for the parallel test runner's own forks
        # under the same uid between the /proc scan and bwrap's fork; 300
        # concurrent children still overrun it by a wide margin.
        result = make_env(max_procs=64).execute(f"python3 {fork_script} concurrent 300")
        assert result["returncode"] == 0, result["output"]
        counts = dict(part.split("=") for part in result["output"].split())
        assert int(counts["failed"]) >= 1, counts
        # Host activity (the parallel test runner included) shifts the uid
        # thread count between the /proc scan and the forks, so the
        # ceiling is asserted within NPROC_DRIFT of max_procs plus the
        # wrapper allowance, not at an exact count.
        assert int(counts["started"]) < 300, counts
        assert int(counts["started"]) <= 64 + WRAPPER_PROCESS_ALLOWANCE + NPROC_DRIFT, counts

    def test_max_procs_zero_disables_the_process_limit(self, make_env, fork_script):
        result = make_env(max_procs=0).execute(f"python3 {fork_script} concurrent 100")
        assert result["returncode"] == 0, result["output"]
        assert result["output"].strip() == "started=100 failed=0"
