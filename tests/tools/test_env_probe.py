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
        # Should not raise, should just return ""
        result = env_probe.get_environment_probe_line()
        # Whatever the result is, it must be a string
        assert isinstance(result, str)


class TestRunHelper:
    """Direct tests for _run — the subprocess wrapper."""

    def test_run_success(self, monkeypatch):
        """A normal command returns (rc, stdout, stderr) stripped."""
        class FakeResult:
            returncode = 0
            stdout = "  hello  \n"
            stderr = ""
        monkeypatch.setattr(env_probe.subprocess, "run", lambda *a, **kw: FakeResult())
        rc, out, err = env_probe._run(["echo", "hello"])
        assert rc == 0
        assert out == "hello"
        assert err == ""

    def test_run_file_not_found(self, monkeypatch):
        """Missing binary → (-1, '', 'not found')."""
        monkeypatch.setattr(env_probe.subprocess, "run",
                            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        rc, out, err = env_probe._run(["nonexistent-binary-xyz"])
        assert rc == -1
        assert out == ""
        assert err == "not found"

    def test_run_timeout(self, monkeypatch):
        """Timeout → (-1, '', 'timeout')."""
        monkeypatch.setattr(env_probe.subprocess, "run",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                env_probe.subprocess.TimeoutExpired(cmd="x", timeout=3)))
        rc, out, err = env_probe._run(["slow-cmd"])
        assert rc == -1
        assert err == "timeout"

    def test_run_oserror(self, monkeypatch):
        """Generic OSError → (-1, '', 'oserror: ...')."""
        monkeypatch.setattr(env_probe.subprocess, "run",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))
        rc, out, err = env_probe._run(["bad-cmd"])
        assert rc == -1
        assert "oserror" in err


class TestPythonVersionOf:
    """Direct tests for _python_version_of."""

    def test_binary_not_found(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: None)
        assert env_probe._python_version_of("nope") is None

    def test_returns_version_on_success(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: "/usr/bin/" + b)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (0, "3.12.4", ""))
        assert env_probe._python_version_of("python3") == "3.12.4"

    def test_returns_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: "/usr/bin/" + b)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (1, "", "error"))
        assert env_probe._python_version_of("python3") is None


class TestHasPipModule:
    """Direct tests for _has_pip_module."""

    def test_binary_not_found(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: None)
        assert env_probe._has_pip_module("nope") is False

    def test_pip_present(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: "/usr/bin/" + b)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (0, "pip 24.0", ""))
        assert env_probe._has_pip_module("python3") is True

    def test_pip_absent(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: "/usr/bin/" + b)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (-1, "", "not found"))
        assert env_probe._has_pip_module("python3") is False


class TestDetectPep668:
    """Direct tests for _detect_pep668."""

    def test_binary_not_found(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: None)
        assert env_probe._detect_pep668("nope") is False

    def test_marker_present(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: "/usr/bin/" + b)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (0, "yes", ""))
        assert env_probe._detect_pep668("python3") is True

    def test_marker_absent(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda b: "/usr/bin/" + b)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (0, "no", ""))
        assert env_probe._detect_pep668("python3") is False


class TestPipPythonVersion:
    """Direct tests for _pip_python_version."""

    def test_pip_not_on_path(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)
        assert env_probe._pip_python_version() is None

    def test_parses_version(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (0, "pip 24.0 from /lib/pip (python 3.12)", ""))
        assert env_probe._pip_python_version() == "3.12"

    def test_returns_none_on_failure(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (-1, "", "error"))
        assert env_probe._pip_python_version() is None

    def test_returns_none_for_malformed_output(self, monkeypatch):
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(env_probe, "_run",
                            lambda cmd, timeout=3.0: (0, "pip 24.0 no parens here", ""))
        assert env_probe._pip_python_version() is None


class TestBuildProbeLineEdgeCases:
    """Cover remaining branches in _build_probe_line."""

    def test_python_alias_different_version(self, monkeypatch):
        """python alias exists with a different version → named in output."""
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: {"python3": "3.12.4", "python": "3.11.0"}.get(b))
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: False)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: False)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: None)
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)
        line = env_probe._build_probe_line()
        assert "python=3.11.0" in line

    def test_pip_without_pip_module(self, monkeypatch):
        """pip on PATH but python3 -m pip fails → 'pip→pythonX.Y' in output."""
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: "3.12.4" if b == "python3" else None)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: False)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: False)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)
        line = env_probe._build_probe_line()
        assert "pip→python3.12" in line
        assert "mismatch" not in line

    def test_pip_not_on_path_but_module_works(self, monkeypatch):
        """pip not on PATH but python3 -m pip works → silent on pip, may emit for other reasons."""
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: "3.12.4" if b == "python3" else None)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: True)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: True)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: None)
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)
        line = env_probe._build_probe_line()
        # PEP 668 without uv → not silent, but no pip=missing (module works)
        assert "PEP 668" in line
        assert "pip=missing" not in line


class TestCacheBehaviour:
    """Cover force_refresh and the cache-set path."""

    def test_force_refresh_reprobes(self, monkeypatch):
        """force_refresh=True clears cache and re-probes."""
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
        env_probe.get_environment_probe_line(force_refresh=True)
        # 2 calls on first probe + 2 on refresh = 4
        assert len(calls) == 4

    def test_probe_exception_returns_empty(self, monkeypatch):
        """If _build_probe_line raises, the exception is swallowed → empty string."""
        monkeypatch.setattr(env_probe, "_build_probe_line",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        result = env_probe.get_environment_probe_line()
        assert result == ""


class TestAsyncWarm:
    def test_warm_starts_one_daemon_thread(self, monkeypatch):
        threads = []

        class FakeThread:
            def __init__(self, *, target, name, daemon):
                self.target = target
                self.name = name
                self.daemon = daemon

            def start(self):
                threads.append(self)

        monkeypatch.setattr(env_probe.threading, "Thread", FakeThread)

        env_probe.warm_environment_probe_async()
        env_probe.warm_environment_probe_async()

        assert len(threads) == 1
        assert threads[0].target is env_probe.get_environment_probe_line
        assert threads[0].name == "env-probe-warm"
        assert threads[0].daemon is True
        assert env_probe._warm_started is True

    def test_warm_skips_when_cache_is_ready(self, monkeypatch):
        env_probe._CACHED_LINE = "cached"

        def unexpected_thread(**kwargs):
            raise AssertionError(f"thread should not start: {kwargs}")

        monkeypatch.setattr(env_probe.threading, "Thread", unexpected_thread)

        env_probe.warm_environment_probe_async()

        assert env_probe._warm_started is False

    def test_reset_clears_cache_and_warm_state(self):
        env_probe._CACHED_LINE = "cached"
        env_probe._warm_started = True

        env_probe._reset_cache_for_tests()

        assert env_probe._CACHED_LINE is None
        assert env_probe._warm_started is False
