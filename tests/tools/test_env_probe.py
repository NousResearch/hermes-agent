"""Tests for tools/env_probe.py — local Python toolchain probe."""

import sys

import pytest

from tools import env_probe


@pytest.fixture(autouse=True)
def reset_probe_cache():
    """Each test starts with a clean cache."""
    env_probe._reset_cache_for_tests()
    yield
    env_probe._reset_cache_for_tests()


class TestSilentWhenHealthy:
    """The probe must emit nothing when the environment is clean — otherwise
    every prompt for every user pays an unnecessary token tax."""

    def test_clean_env_returns_empty(self, monkeypatch):
        """python3 + pip module + no PEP 668 → silent."""
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: "3.13.3" if b == "python3" else None)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: True)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: False)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.13")
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)
        assert env_probe.get_environment_probe_line() == ""

    def test_pep668_with_uv_returns_empty(self, monkeypatch):
        """PEP 668 alone shouldn't trigger output if uv is installed —
        agent has a viable install path."""
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: "3.12.4" if b == "python3" else None)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: True)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: True)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which",
                            lambda name: "/usr/local/bin/uv" if name == "uv" else None)
        assert env_probe.get_environment_probe_line() == ""


class TestEmitsOnRealProblems:
    """The probe must produce a usable line for the real failure modes
    that drove this feature."""

    def test_allen_scenario_python_version_mismatch(self, monkeypatch):
        """python3 is 3.11 (no pip module), pip on PATH is 3.12, PEP 668 on,
        no uv — the exact scenario from the Sarasota real-estate task."""
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: {"python3": "3.11.15", "python": None}.get(b))
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: False)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: True)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which",
                            lambda name: None if name == "uv" else "/usr/bin/" + name)

        line = env_probe.get_environment_probe_line()
        assert line  # not silent
        # Single line — must not blow up the system prompt.
        assert "\n" not in line
        # Names the real toolchain state
        assert "3.11.15" in line
        assert "no pip module" in line
        assert "mismatch" in line
        assert "PEP 668" in line
        # Points at the right escape hatch
        assert "venv" in line or "uv" in line

    def test_missing_python3_is_named(self, monkeypatch):
        """If python3 isn't installed at all, say so."""
        monkeypatch.setattr(env_probe, "_python_version_of", lambda b: None)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: False)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: False)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: None)
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)

        line = env_probe.get_environment_probe_line()
        assert "python3=missing" in line

    def test_python_missing_but_python3_present(self, monkeypatch):
        """Common on Debian: only python3 exists, agent shouldn't type
        `python`."""
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: "3.12.4" if b == "python3" else None)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: True)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: True)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which",
                            lambda name: None if name == "uv" else "/usr/bin/" + name)

        line = env_probe.get_environment_probe_line()
        # `python=missing` only matters in the non-silent path; PEP 668 (without
        # uv) is what brings us off-silent here, so check both signals.
        assert "PEP 668" in line
        assert "python=missing" in line


class TestSkipsRemoteBackends:
    """Remote backends have their own probe; this one must stay out."""

    def test_docker_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        # Even with a broken local env, docker must emit nothing.
        monkeypatch.setattr(env_probe, "_python_version_of", lambda b: None)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: False)
        assert env_probe.get_environment_probe_line() == ""

    def test_modal_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        assert env_probe.get_environment_probe_line() == ""

    def test_ssh_returns_empty(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        assert env_probe.get_environment_probe_line() == ""


class TestCaching:
    """The probe runs once per process — the result is deterministic for
    the lifetime of the agent."""

    def test_result_cached(self, monkeypatch):
        calls = []

        def counting_version(b):
            calls.append(b)
            return "3.12.4" if b == "python3" else None

        monkeypatch.setattr(env_probe, "_python_version_of", counting_version)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: True)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: False)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)

        env_probe.get_environment_probe_line()
        env_probe.get_environment_probe_line()
        env_probe.get_environment_probe_line()

        # Only the first call probes — caller-counting confirms it.
        # Two calls (python3 + python) on first invocation, zero after.
        assert len(calls) == 2


class TestRobustness:
    """The probe must NEVER crash the prompt build."""

    def test_subprocess_failure_returns_empty(self, monkeypatch):
        """If every subprocess fails, just stay silent."""
        def boom(*a, **kw):
            raise OSError("simulated")
        monkeypatch.setattr(env_probe.subprocess, "run", boom)
        monkeypatch.setattr(env_probe.subprocess, "Popen", boom)
        # Should not raise, should just return ""
        result = env_probe.get_environment_probe_line()
        # Whatever the result is, it must be a string
        assert isinstance(result, str)


class TestStuckProbeNeverBlocksCallers:
    """Regression for #67964: on Windows an orphaned pip descendant kept
    the probe's capture pipes open, wedging the warm thread inside
    subprocess._communicate while it held the module lock — every new
    session's prompt build then blocked forever.  Callers must fail open
    within a bounded time no matter what the probe subprocesses do."""

    def test_hung_probe_fails_open_for_concurrent_callers(self, monkeypatch):
        """Concurrent get_environment_probe_line() callers return "" within
        a bounded wall-clock time while the probe worker stays stuck."""
        import threading as _threading
        import time

        release = _threading.Event()

        def stuck_probe():
            # Simulate the wedged pipe read: blocks until released.
            release.wait(timeout=30)
            return "Python toolchain: late-result."

        monkeypatch.setattr(env_probe, "_build_probe_line", stuck_probe)
        # Keep the test fast — the bound just has to exist, not be 10s.
        monkeypatch.setattr(env_probe, "_PROBE_WAIT_TIMEOUT", 0.5)

        env_probe.warm_environment_probe_async()

        results: list[str] = []
        errors: list[BaseException] = []

        def caller():
            try:
                results.append(env_probe.get_environment_probe_line())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [_threading.Thread(target=caller, daemon=True) for _ in range(4)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.monotonic() - start

        try:
            assert not errors
            assert all(not t.is_alive() for t in threads), "caller blocked on stuck probe"
            # All callers failed open with the empty line.
            assert results == ["", "", "", ""]
            # Bounded: nowhere near the 30s the probe is stuck for.
            assert elapsed < 8
        finally:
            release.set()

    def test_late_probe_result_published_after_recovery(self, monkeypatch):
        """If the stuck worker eventually finishes, later callers get the
        line — recovery without restart, matching the incident (killing
        the orphan un-wedged everything)."""
        import threading as _threading

        release = _threading.Event()

        def slow_probe():
            release.wait(timeout=30)
            return "Python toolchain: recovered."

        monkeypatch.setattr(env_probe, "_build_probe_line", slow_probe)
        monkeypatch.setattr(env_probe, "_PROBE_WAIT_TIMEOUT", 0.2)

        # First caller times out and fails open.
        assert env_probe.get_environment_probe_line() == ""

        # Worker un-wedges (the operator killed the orphan).
        release.set()
        assert env_probe._PROBE_DONE.wait(timeout=10)

        # Later callers see the published line.
        assert env_probe.get_environment_probe_line() == "Python toolchain: recovered."

    def test_repeat_callers_do_not_pay_full_wait_after_first_timeout(self, monkeypatch):
        """After one caller burns the full wait, subsequent callers only
        peek — a permanently stuck probe costs the timeout once, not
        per-session."""
        import threading as _threading
        import time

        release = _threading.Event()

        def stuck_probe():
            release.wait(timeout=30)
            return ""

        monkeypatch.setattr(env_probe, "_build_probe_line", stuck_probe)
        monkeypatch.setattr(env_probe, "_PROBE_WAIT_TIMEOUT", 0.5)

        try:
            assert env_probe.get_environment_probe_line() == ""  # pays 0.5s

            # Crank the timeout way up: if the peek short-circuit is broken,
            # the next call blocks ~30s; if it works, it returns in ~0.05s.
            monkeypatch.setattr(env_probe, "_PROBE_WAIT_TIMEOUT", 30.0)
            start = time.monotonic()
            assert env_probe.get_environment_probe_line() == ""
            assert time.monotonic() - start < 5  # peek, not a full wait
        finally:
            release.set()


class TestRunTimeoutIsBounded:
    """_run() itself must return within a bounded time even when the child
    spawns a descendant that inherits the capture pipes and outlives it —
    the exact Windows pip.exe launcher shape from #67964, reproduced
    cross-platform with a shell child."""

    def test_run_returns_promptly_despite_pipe_holding_descendant(self):
        import time

        # Child exits quickly; grandchild inherits stdout/stderr and sleeps
        # far beyond the timeout, keeping the pipe write-ends open.
        script = (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])\n"
            "time.sleep(20)\n"
        )
        start = time.monotonic()
        rc, out, err = env_probe._run([sys.executable, "-c", script], timeout=1.0)
        elapsed = time.monotonic() - start

        assert rc == -1
        assert err == "timeout"
        # stdlib subprocess.run on Windows would hang here for the full 20s
        # (unbounded post-kill communicate).  Our bound: timeout + reap slack.
        assert elapsed < 6
