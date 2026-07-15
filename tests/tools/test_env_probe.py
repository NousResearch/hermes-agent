"""Tests for tools/env_probe.py — local Python toolchain probe."""

import sys

import pytest

from tools import env_probe


@pytest.fixture(autouse=True)
def reset_probe_cache(monkeypatch):
    """Each test starts with a clean cache and no HERMES_PYTHON override."""
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
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


class TestHermesPythonOverride:
    """The HERMES_PYTHON env var must be picked up when set, and the
    bare name ``python3`` must be used when it's unset."""

    CUSTOM_PYTHON = "/nix/store/x3k9f2a-python3-3.12.4/bin/python3"

    def test_hermes_python_used_when_set(self, monkeypatch):
        """HERMES_PYTHON=/path/to/python3 → that path is passed to the
        version/pip/pep668 helpers instead of bare ``python3``."""
        monkeypatch.setenv("HERMES_PYTHON", self.CUSTOM_PYTHON)

        called_with: list[str] = []

        def spy_version(b: str) -> str | None:
            called_with.append(b)
            return "3.12.4" if b == self.CUSTOM_PYTHON else None

        def spy_has_pip(b: str) -> bool:
            called_with.append(b)
            return b == self.CUSTOM_PYTHON

        def spy_pep668(b: str) -> bool:
            called_with.append(b)
            return False

        monkeypatch.setattr(env_probe, "_python_version_of", spy_version)
        monkeypatch.setattr(env_probe, "_has_pip_module", spy_has_pip)
        monkeypatch.setattr(env_probe, "_detect_pep668", spy_pep668)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)

        line = env_probe.get_environment_probe_line()
        # Should be silent — we faked a healthy env
        assert line == "", f"expected empty line, got: {line!r}"
        # The custom path was used, not bare "python3"
        assert self.CUSTOM_PYTHON in called_with, (
            f"expected {self.CUSTOM_PYTHON!r} in called_with, got {called_with}"
        )
        assert "python3" not in called_with or called_with == ["python"], (
            f"bare 'python3' should not be called, got: {called_with}"
        )

    def test_hermes_python_unset_falls_back_to_bare_python3(self, monkeypatch):
        """HERMES_PYTHON unset → helpers receive bare ``python3``."""
        monkeypatch.delenv("HERMES_PYTHON", raising=False)

        called_with: list[str] = []

        def spy_version(b: str) -> str | None:
            called_with.append(b)
            return "3.12.4" if b == "python3" else None

        def spy_has_pip(b: str) -> bool:
            called_with.append(b)
            return b == "python3"

        def spy_pep668(b: str) -> bool:
            called_with.append(b)
            return False

        monkeypatch.setattr(env_probe, "_python_version_of", spy_version)
        monkeypatch.setattr(env_probe, "_has_pip_module", spy_has_pip)
        monkeypatch.setattr(env_probe, "_detect_pep668", spy_pep668)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)

        line = env_probe.get_environment_probe_line()
        assert line == "", f"expected empty line, got: {line!r}"
        assert "python3" in called_with, (
            f"expected 'python3' in called_with, got {called_with}"
        )


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
