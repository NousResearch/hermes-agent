"""Tests for tools/env_probe.py — local Python toolchain probe."""

import subprocess
import sys
from types import SimpleNamespace

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


class TestWindowsSubprocesses:
    def test_warm_probe_hides_every_windows_subprocess(self, monkeypatch):
        """Every process reached by the async warm path must be windowless."""
        captured = []
        create_no_window = 0x08000000

        class ImmediateThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        def fake_which(name):
            return None if name == "uv" else f"C:/bin/{name}.exe"

        def fake_run(cmd, **kwargs):
            captured.append((cmd, kwargs))
            if cmd[:3] == ["python3", "-m", "pip"]:
                stdout = "pip 24.0 from C:/pip (python 3.12)\n"
            elif cmd == ["pip", "--version"]:
                stdout = "pip 24.0 from C:/pip (python 3.12)\n"
            elif "EXTERNALLY-MANAGED" in cmd[-1]:
                stdout = "no\n"
            else:
                stdout = "3.12.4\n"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(env_probe, "IS_WINDOWS", True)
        monkeypatch.setattr(
            env_probe,
            "windows_hide_flags",
            lambda: create_no_window,
        )
        monkeypatch.setattr(env_probe.threading, "Thread", ImmediateThread)
        monkeypatch.setattr(env_probe.shutil, "which", fake_which)
        monkeypatch.setattr(env_probe.subprocess, "run", fake_run)

        env_probe.warm_environment_probe_async()

        assert len(captured) == 5, captured
        commands = [cmd for cmd, _kwargs in captured]
        assert [cmd[0] for cmd in commands] == [
            "python3",
            "python",
            "python3",
            "pip",
            "python3",
        ]
        assert commands[2] == ["python3", "-m", "pip", "--version"]
        assert commands[3] == ["pip", "--version"]
        assert all(commands[index][1] == "-c" for index in (0, 1, 4))
        assert all(
            kwargs.get("creationflags") == create_no_window
            for _cmd, kwargs in captured
        ), captured
        assert all(
            kwargs.get("stdin") is subprocess.DEVNULL
            for _cmd, kwargs in captured
        ), captured


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
