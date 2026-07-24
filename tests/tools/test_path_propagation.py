"""Tests for PATH propagation into subprocess environments.

Verifies that custom PATH entries set via Dockerfile ``ENV``, ``.env``
files, or ``config.yaml`` ``terminal.env`` are preserved in the agent's
subprocess execution context (issue #70905).

The key contract:
  ``_make_run_env`` must forward the *full* os.environ PATH, extended
  with ``_SANE_PATH`` fallback dirs and the Hermes install dir, never
  silently dropping custom entries.
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.local import (
    LocalEnvironment,
    _make_run_env,
    _append_missing_sane_path_entries,
    _path_env_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_make_run_env(extra_os_environ: dict | None = None,
                          terminal_env: dict | None = None) -> dict:
    """Call ``_make_run_env(env)`` with a controlled ``os.environ`` and return the
    resulting run-env dict."""
    base_environ = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
        "USER": "testuser",
    }
    if extra_os_environ:
        base_environ.update(extra_os_environ)

    with patch.dict(os.environ, base_environ, clear=True):
        return _make_run_env(terminal_env or {})


def _capture_exec_env(extra_os_environ: dict | None = None,
                      self_env: dict | None = None) -> dict:
    """Run a real LocalEnvironment.execute() call with mocked Popen
    and return the env dict passed to the subprocess."""
    captured: dict = {}
    fake_interrupt = threading.Event()
    test_environ = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
        "USER": "testuser",
    }
    if extra_os_environ:
        test_environ.update(extra_os_environ)

    env = LocalEnvironment(cwd="/tmp", timeout=10, env=self_env)
    with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
         patch("subprocess.Popen", side_effect=_make_fake_popen(captured)), \
         patch("tools.terminal_tool._interrupt_event", fake_interrupt), \
         patch.dict(os.environ, test_environ, clear=True):
        env.execute("echo hello")

    return captured.get("env", {})


def _make_fake_popen(captured: dict):
    """Return a fake Popen constructor that records the env kwarg."""
    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.stdout = MagicMock()
        proc.stdin = MagicMock()
        return proc
    return fake_popen


# ---------------------------------------------------------------------------
# Unit tests for _make_run_env
# ---------------------------------------------------------------------------

class TestMakeRunEnvPathPropagation:
    """Direct tests on ``_make_run_env`` — no subprocess needed."""

    def test_custom_path_is_preserved(self):
        """A custom PATH entry (/potato) survives the env-building pipeline."""
        result = _capture_make_run_env(
            extra_os_environ={"PATH": "/potato:/usr/bin:/bin"},
        )
        path = result.get("PATH", "")
        assert "/potato" in path, (
            f"Expected /potato in PATH, got: {path}")

    def test_custom_path_with_sane_entries(self):
        """_SANE_PATH entries are appended without removing custom dirs."""
        result = _capture_make_run_env(
            extra_os_environ={"PATH": "/custom/bin:/usr/bin"},
        )
        path = result.get("PATH", "")
        # Custom entry must still be present (hermes bin dir may be prepended
        # by _prepend_hermes_bin_dir, but the custom entry itself is kept).
        assert "/custom/bin" in path, (
            f"Expected /custom/bin in PATH, got: {path}")

    def test_path_missing_from_os_environ_falls_back_to_sane(self):
        """When os.environ has no PATH, _make_run_env still produces a path."""
        with patch.dict(os.environ, {}, clear=True):
            result = _make_run_env({})
        path = result.get("PATH", "")
        assert path, "PATH should never be empty after _make_run_env"
        assert "/usr/bin" in path or "/bin" in path

    def test_terminal_env_path_takes_precedence(self):
        """A PATH passed via terminal.env config overrides os.environ PATH."""
        result = _capture_make_run_env(
            extra_os_environ={"PATH": "/usr/bin:/bin"},
            terminal_env={"PATH": "/override/bin"},
        )
        path = result.get("PATH", "")
        # The terminal.env PATH entry must be present (hermes bin dir may
        # be prepended first by _prepend_hermes_bin_dir, but the override
        # value is in PATH).
        assert "/override/bin" in path, (
            f"Expected /override/bin in PATH from terminal.env, "
            f"got: {path}")

    def test_path_with_multiple_custom_entries(self):
        """Multiple custom PATH entries are all preserved."""
        result = _capture_make_run_env(
            extra_os_environ={
                "PATH": "/potato:/opt/custom/bin:/usr/bin:/bin",
            },
        )
        path = result.get("PATH", "")
        assert "/potato" in path, f"/potato missing from PATH: {path}"
        assert "/opt/custom/bin" in path, (
            f"/opt/custom/bin missing from PATH: {path}")

    def test_empty_path_in_env_falls_back_to_os_environ(self):
        """If run_env has an empty PATH (edge case), it falls back to
        os.environ."""
        with patch.dict(os.environ, {"PATH": "/fallback/bin"}, clear=True):
            result = _make_run_env({})
        path = result.get("PATH", "")
        assert "/fallback/bin" in path, (
            f"Expected /fallback/bin in PATH, got: {path}")


# ---------------------------------------------------------------------------
# Integration tests through LocalEnvironment.execute()
# ---------------------------------------------------------------------------

class TestPathPropagationThroughExecute:
    """PATH propagation through the full execute() path."""

    def test_custom_os_environ_path_reaches_subprocess(self):
        """Custom PATH from os.environ is visible in the subprocess env."""
        subprocess_env = _capture_exec_env(
            extra_os_environ={"PATH": "/potato:/usr/bin:/bin"},
        )
        path = subprocess_env.get("PATH", "")
        assert "/potato" in path, (
            f"Expected /potato in subprocess PATH, got: {path}")

    def test_hermes_bin_dir_is_prepended(self):
        """The Hermes install bin dir is prepended to PATH."""
        from tools.environments.local import _resolve_hermes_bin_dir
        hermes_bin = _resolve_hermes_bin_dir()
        if not hermes_bin:
            pytest.skip("Hermes bin dir not resolvable in this environment")

        subprocess_env = _capture_exec_env()
        path = subprocess_env.get("PATH", "")
        assert hermes_bin in path, (
            f"Expected hermes bin dir {hermes_bin} in PATH, got: {path}")

    def test_terminal_env_path_overrides_os_environ(self):
        """terminal.env PATH overrides os.environ PATH."""
        subprocess_env = _capture_exec_env(
            extra_os_environ={"PATH": "/usr/bin:/bin"},
            self_env={"PATH": "/config_env/bin"},
        )
        path = subprocess_env.get("PATH", "")
        # The config_env PATH entry must be present (hermes bin dir may
        # be prepended first, but the override is in PATH).
        assert "/config_env/bin" in path, (
            f"Expected /config_env/bin in subprocess PATH, got: {path}")


# ---------------------------------------------------------------------------
# Unit tests for _append_missing_sane_path_entries
# ---------------------------------------------------------------------------

class TestAppendMissingSanePathEntries:
    """Focused tests for the PATH normalisation helper."""

    def test_custom_entries_preserved(self):
        """_append_missing_sane_path_entries does not strip custom dirs."""
        result = _append_missing_sane_path_entries(
            "/potato:/usr/bin:/bin",
        )
        assert "/potato" in result

    def test_empty_entries_removed(self):
        """Empty PATH entries (from leading/trailing/double colon) are removed
        but custom entries survive."""
        result = _append_missing_sane_path_entries(
            ":/potato::/usr/bin:/bin:",
        )
        assert "/potato" in result
        # No empty entries
        for segment in result.split(":"):
            assert segment, f"Empty segment found in PATH: {result}"

    def test_duplicates_removed(self):
        """Duplicate PATH entries are collapsed, preserving first occurrence."""
        result = _append_missing_sane_path_entries(
            "/potato:/usr/bin:/potato:/bin:/potato",
        )
        parts = result.split(":")
        # /potato appears only once
        assert parts.count("/potato") == 1, (
            f"/potato appears {parts.count('/potato')} times: {result}")
