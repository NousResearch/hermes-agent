"""Tests for the bwrap availability probe and its degraded_mode path.

The probe is ``bwrap --unshare-user --ro-bind / / true`` with a 5 s
timeout, run once per process on the first construction; a passing
result is cached, a failing one raises EnvironmentConnectionError and is
retried on the next construction. Commands are never run unsandboxed: a
recorded Popen shows every spawn starting with the prlimit prefix and then
the bwrap path.

Unit tests never spawn bwrap. Integration tests are skipped as a module
when bwrap is missing or its runtime probe fails, so CI without bwrap
stays green.
"""

import inspect
import json
import logging
import os
import shutil
import subprocess
from unittest.mock import patch

import pytest

from tools.environments import bubblewrap
from tools.environments.base import EnvironmentConnectionError
from tools.environments.bubblewrap import (
    INSTALL_HINT,
    PROBE_ARGS,
    BubblewrapEnvironment,
    probe_bwrap,
    run_probe,
)
from tools.environments.local import LocalEnvironment


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

BWRAP = "/usr/bin/bwrap"
PRLIMIT = "/usr/bin/prlimit"


@pytest.fixture(autouse=True)
def _fresh_probe_cache(monkeypatch):
    monkeypatch.setattr(bubblewrap, "_probed_bwrap_path", None)
    monkeypatch.setattr(bubblewrap, "_process_limit_scoped", True)


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


@pytest.fixture
def no_bwrap_on_path(tmp_path, monkeypatch):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


def _no_session():
    return patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None)


def _completed(returncode, stderr="", stdout=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class TestRunProbe:
    """One probe run, with the bwrap lookup and the subprocess stubbed."""

    @pytest.fixture
    def runs(self, monkeypatch):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            outcome = fake_run.outcome
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        fake_run.outcome = _completed(0)
        monkeypatch.setattr(bubblewrap.shutil, "which", lambda name: {"bwrap": BWRAP, "prlimit": PRLIMIT}.get(name))
        monkeypatch.setattr(bubblewrap.subprocess, "run", fake_run)
        return calls, fake_run

    def test_missing_bwrap_fails_without_spawning(self, runs, monkeypatch):
        calls, _ = runs
        monkeypatch.setattr(bubblewrap.shutil, "which", lambda name: None)
        path, failure = run_probe()
        assert path is None
        assert "bwrap" in failure and "PATH" in failure
        assert calls == []

    def test_missing_prlimit_fails_without_spawning(self, runs, monkeypatch):
        calls, _ = runs
        monkeypatch.setattr(bubblewrap.shutil, "which", lambda name: BWRAP if name == "bwrap" else None)
        path, failure = run_probe()
        assert path == BWRAP
        assert "prlimit" in failure and "PATH" in failure
        assert calls == []

    def test_passing_probe_runs_the_documented_argv_with_a_5s_timeout(self, runs):
        calls, _ = runs
        assert run_probe() == (BWRAP, None)
        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv == [BWRAP, "--unshare-user", "--ro-bind", "/", "/", "true"]
        assert argv[1:] == list(PROBE_ARGS)
        assert kwargs["timeout"] == 5

    def test_nonzero_exit_reports_the_stderr(self, runs):
        _, fake_run = runs
        fake_run.outcome = _completed(1, stderr="bwrap: setting up uid map: Permission denied\n")
        path, failure = run_probe()
        assert path == BWRAP
        assert "exit 1" in failure
        assert "uid map: Permission denied" in failure

    def test_timeout_reports_the_limit(self, runs):
        _, fake_run = runs
        fake_run.outcome = subprocess.TimeoutExpired(cmd="bwrap", timeout=5)
        path, failure = run_probe()
        assert path == BWRAP
        assert "timed out" in failure and "5 s" in failure

    def test_unstartable_bwrap_reports_the_oserror(self, runs):
        _, fake_run = runs
        fake_run.outcome = PermissionError(13, "Permission denied")
        path, failure = run_probe()
        assert path == BWRAP
        assert "could not start" in failure and "Permission denied" in failure


class TestProbeCache:
    @pytest.fixture
    def probe_double(self, monkeypatch):
        calls = []

        def fake_probe():
            calls.append(1)
            return fake_probe.outcome

        fake_probe.outcome = (BWRAP, None)
        monkeypatch.setattr(bubblewrap, "run_probe", fake_probe)
        return calls, fake_probe

    def test_passing_probe_runs_once_per_process(self, probe_double):
        calls, _ = probe_double
        assert probe_bwrap() == BWRAP
        assert probe_bwrap() == BWRAP
        assert len(calls) == 1

    def test_second_construction_makes_no_probe_call(self, probe_double, sandbox_root, work_dir):
        calls, _ = probe_double
        with _no_session():
            first = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
            second = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        assert len(calls) == 1
        assert first._bwrap_path == second._bwrap_path == BWRAP
        first.cleanup()
        second.cleanup()

    def test_failing_probe_raises_and_is_retried_next_time(self, probe_double):
        calls, fake_probe = probe_double
        fake_probe.outcome = (BWRAP, "bwrap probe failed (exit 1): no permission")
        with pytest.raises(EnvironmentConnectionError) as excinfo:
            probe_bwrap()
        assert "bwrap probe failed (exit 1): no permission" in excinfo.value.reason
        assert "bubblewrap" in excinfo.value.retry_hint
        assert excinfo.value.retry_hint == INSTALL_HINT
        fake_probe.outcome = (BWRAP, None)
        assert probe_bwrap() == BWRAP
        assert len(calls) == 2


class TestConstructionFailures:
    def test_path_without_bwrap_raises_before_touching_disk(self, sandbox_root, work_dir, no_bwrap_on_path):
        with _no_session(), pytest.raises(EnvironmentConnectionError) as excinfo:
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        assert "bwrap" in excinfo.value.reason and "PATH" in excinfo.value.reason
        assert "bubblewrap" in excinfo.value.retry_hint
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())

    def test_nonzero_probe_raises_the_same(self, sandbox_root, work_dir, monkeypatch):
        monkeypatch.setattr(bubblewrap.shutil, "which", lambda name: BWRAP)
        monkeypatch.setattr(
            bubblewrap.subprocess, "run",
            lambda *a, **k: _completed(1, stderr="bwrap: No permissions to create a new namespace"),
        )
        with _no_session(), pytest.raises(EnvironmentConnectionError) as excinfo:
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        assert "No permissions to create a new namespace" in excinfo.value.reason
        assert "bubblewrap" in excinfo.value.retry_hint
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())


@pytest.fixture
def isolated_tool(tmp_path, monkeypatch):
    """terminal_tool with backend=bubblewrap, an isolated HERMES_HOME and a
    clean environment cache; the real factory constructs the environment."""
    import tools.terminal_tool as tt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "bubblewrap")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    def _clear():
        with tt._env_lock:
            tt._active_environments.clear()
            tt._last_activity.clear()

    _clear()
    yield tt
    _clear()


class TestDegradedModePath:
    """Construction sits inside the EnvironmentConnectionError handler of terminal_tool."""

    def test_warn_returns_the_degraded_result(self, isolated_tool, no_bwrap_on_path, monkeypatch):
        monkeypatch.delenv("TERMINAL_DEGRADED_MODE", raising=False)
        r = json.loads(isolated_tool.terminal_tool("echo hi", task_id="t-bwrap-degraded"))
        assert r["status"] == "degraded"
        assert r["exit_code"] == -1
        assert "bwrap" in r["reason"]
        assert "bubblewrap" in r["retry_hint"]
        assert "traceback" not in r
        with isolated_tool._env_lock:
            assert not isolated_tool._active_environments

    def test_fail_returns_the_error_result(self, isolated_tool, no_bwrap_on_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_DEGRADED_MODE", "fail")
        r = json.loads(isolated_tool.terminal_tool("echo hi", task_id="t-bwrap-fail"))
        assert r["status"] == "error"
        assert r["exit_code"] == -1
        assert "bwrap" in r["error"]
        assert "traceback" in r


class TestEverySpawnIsWrapped:
    def test_backend_module_spawns_nothing_but_its_two_probes(self):
        source = inspect.getsource(bubblewrap)
        assert "Popen(" not in source
        assert source.count("subprocess.run(") == 2
        assert "subprocess.run(" in inspect.getsource(run_probe)
        # The process limit probe runs bwrap itself: it has to, to measure
        # how the kernel counts inside a sandbox.
        assert "subprocess.run(" in inspect.getsource(bubblewrap.run_process_limit_probe)

    @needs_bwrap
    def test_every_popen_argv_is_the_probe_or_prlimit_then_bwrap(self, sandbox_root, work_dir, monkeypatch):
        recorded = []
        real_popen = subprocess.Popen

        def recording_popen(args, *a, **kw):
            recorded.append(list(args))
            return real_popen(args, *a, **kw)

        monkeypatch.setattr(subprocess, "Popen", recording_popen)
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            assert env.execute("echo hi")["output"].strip() == "hi"
            assert env.execute("cd /usr/share")["returncode"] == 0
            assert env.execute("pwd")["output"].strip() == "/usr/share"
        finally:
            env.cleanup()
        # The probe, the login bootstrap and the three commands: the probe
        # is bwrap alone, every other spawn is the prlimit limits and then
        # the sandbox prefix.
        assert len(recorded) >= 5
        assert env._bwrap_path == shutil.which("bwrap")
        assert env._prlimit_path == shutil.which("prlimit")
        assert recorded[0] == [env._bwrap_path, *PROBE_ARGS]
        for argv in recorded[1:]:
            assert argv[0] == env._prlimit_path, argv
            i = argv.index(env._bwrap_path)
            assert argv[1:i] and all(a.startswith("--") for a in argv[1:i]), argv
            assert argv[i + 1] == "--unshare-all", argv
            assert "--" in argv


class TestProcessLimitProbe:
    """RLIMIT_NPROC set inside a sandbox counts that sandbox alone only on a
    kernel that accounts it per user namespace. One probe spawn decides."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bubblewrap, "_probed_bwrap_path", shutil.which("bwrap") or "/usr/bin/bwrap")
        monkeypatch.setattr(bubblewrap, "_process_limit_scoped", None)
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))

    @staticmethod
    def _env(tmp_path, **config):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            return bubblewrap.BubblewrapEnvironment(cwd=str(work), timeout=10, config=bubblewrap.BubblewrapConfig(**config))

    def test_probe_sets_a_small_limit_inside_a_sandbox_and_forks_once(self, monkeypatch):
        seen = {}

        def fake_run(argv, **kwargs):
            seen["argv"], seen["kwargs"] = argv, kwargs
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(bubblewrap.subprocess, "run", fake_run)
        assert bubblewrap.run_process_limit_probe("/usr/bin/bwrap", "/usr/bin/prlimit") is True
        argv = seen["argv"]
        assert argv[0] == "/usr/bin/bwrap" and "--unshare-all" in argv
        inner = argv[argv.index("--") + 1:]
        assert inner[:2] == ["/usr/bin/prlimit", f"--nproc={bubblewrap.PROCESS_LIMIT_PROBE_VALUE}"]
        assert seen["kwargs"]["timeout"] == bubblewrap.PROBE_TIMEOUT_SECONDS

    @pytest.mark.parametrize("outcome", ["nonzero", "oserror", "timeout"])
    def test_probe_fails_closed(self, monkeypatch, outcome):
        def fake_run(argv, **kwargs):
            if outcome == "oserror":
                raise OSError("no bwrap")
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(argv, 5)
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="sh: Cannot fork")

        monkeypatch.setattr(bubblewrap.subprocess, "run", fake_run)
        assert bubblewrap.run_process_limit_probe("/usr/bin/bwrap", "/usr/bin/prlimit") is False

    def test_failed_probe_drops_the_process_limit_and_warns_once(self, monkeypatch, tmp_path, caplog):
        calls = []
        monkeypatch.setattr(bubblewrap, "run_process_limit_probe", lambda *a: calls.append(a) or False)
        with caplog.at_level(logging.WARNING, logger="tools.environments.bubblewrap"):
            envs = [self._env(tmp_path) for _ in range(3)]
        try:
            for env in envs:
                argv = env._wrap_popen_args(["bash"])
                assert argv[:3] == [env._prlimit_path, f"--as={256 * 1024 * 1024}", "--cpu=30"]
                assert not any(a.startswith("--nproc") for a in argv)
                assert argv[-2:] == ["--", "bash"]
        finally:
            for env in envs:
                env.cleanup()
        assert len(calls) == 1
        warnings = [r for r in caplog.records if "process limit" in r.getMessage()]
        assert len(warnings) == 1
        assert "5.14" in warnings[0].getMessage()

    def test_passed_probe_keeps_the_process_limit_and_runs_once(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(bubblewrap, "run_process_limit_probe", lambda *a: calls.append(a) or True)
        envs = [self._env(tmp_path) for _ in range(3)]
        try:
            for env in envs:
                assert any(a.startswith("--nproc=") for a in env._wrap_popen_args(["bash"]))
        finally:
            for env in envs:
                env.cleanup()
        assert len(calls) == 1

    def test_no_probe_when_max_procs_is_zero(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(bubblewrap, "run_process_limit_probe", lambda *a: calls.append(a) or True)
        env = self._env(tmp_path, max_procs=0)
        try:
            assert not any(a.startswith("--nproc") for a in env._wrap_popen_args(["bash"]))
        finally:
            env.cleanup()
        assert calls == []

    @needs_bwrap
    def test_real_probe_passes_on_this_kernel(self):
        release = tuple(int(part) for part in os.uname().release.split("-")[0].split(".")[:2])
        if release < (5, 17):
            pytest.skip("the kernel predates reliable per-namespace process accounting")
        assert bubblewrap.run_process_limit_probe(shutil.which("bwrap"), shutil.which("prlimit")) is True
        assert bubblewrap.process_limit_is_scoped(shutil.which("bwrap"), shutil.which("prlimit")) is True
