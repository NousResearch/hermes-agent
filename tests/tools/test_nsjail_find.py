"""Tests for tools.environments.nsjail.find_nsjail — nsjail CLI discovery."""

import os
from unittest.mock import patch

import pytest

from tools.environments import nsjail as nsjail_mod


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Clear the module-level cache and any inherited HERMES_NSJAIL_BINARY
    env var so tests see only what they explicitly set."""
    nsjail_mod._nsjail_executable = None
    monkeypatch.delenv("HERMES_NSJAIL_BINARY", raising=False)
    yield
    nsjail_mod._nsjail_executable = None


class TestFindNsjail:
    def test_found_via_shutil_which(self):
        with patch("tools.environments.nsjail.shutil.which", return_value="/usr/local/bin/nsjail"):
            result = nsjail_mod.find_nsjail()
        assert result == "/usr/local/bin/nsjail"

    def test_returns_none_when_not_found(self):
        with patch("tools.environments.nsjail.shutil.which", return_value=None), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_NSJAIL_BINARY", None)
            result = nsjail_mod.find_nsjail()
        assert result is None

    def test_caches_result(self):
        with patch("tools.environments.nsjail.shutil.which", return_value="/usr/bin/nsjail"):
            first = nsjail_mod.find_nsjail()
        # Second call should use cache, not call shutil.which again.
        with patch("tools.environments.nsjail.shutil.which", return_value=None):
            second = nsjail_mod.find_nsjail()
        assert first == second == "/usr/bin/nsjail"

    def test_env_var_override_takes_precedence(self, tmp_path):
        """HERMES_NSJAIL_BINARY overrides PATH discovery."""
        fake_binary = tmp_path / "nsjail"
        fake_binary.write_text("#!/bin/sh\n")
        fake_binary.chmod(0o755)

        with patch.dict(os.environ, {"HERMES_NSJAIL_BINARY": str(fake_binary)}), \
             patch("tools.environments.nsjail.shutil.which", return_value="/usr/bin/nsjail"):
            result = nsjail_mod.find_nsjail()
        assert result == str(fake_binary)

    def test_env_var_override_ignored_when_nonexistent(self):
        """A dangling HERMES_NSJAIL_BINARY must not preempt PATH lookup."""
        with patch.dict(os.environ, {"HERMES_NSJAIL_BINARY": "/does/not/exist/nsjail"}), \
             patch("tools.environments.nsjail.shutil.which", return_value="/usr/bin/nsjail"):
            result = nsjail_mod.find_nsjail()
        assert result == "/usr/bin/nsjail"

    def test_env_var_override_ignored_when_not_executable(self, tmp_path):
        """A non-executable file at HERMES_NSJAIL_BINARY must not be accepted."""
        non_exec = tmp_path / "nsjail"
        non_exec.write_text("#!/bin/sh\n")
        non_exec.chmod(0o644)

        with patch.dict(os.environ, {"HERMES_NSJAIL_BINARY": str(non_exec)}), \
             patch("tools.environments.nsjail.shutil.which", return_value="/usr/bin/nsjail"):
            result = nsjail_mod.find_nsjail()
        assert result == "/usr/bin/nsjail"
