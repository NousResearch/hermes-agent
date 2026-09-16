"""Tests for Singularity/Apptainer preflight availability check.

Verifies that a clear error is raised when neither apptainer nor
singularity is installed, instead of a cryptic FileNotFoundError.

See: https://github.com/NousResearch/hermes-agent/issues/1511
"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from tools.environments.singularity import (
    _find_singularity_executable,
    _ensure_singularity_available,
)


class TestFindSingularityExecutable:
    """_find_singularity_executable resolution tests."""

    def test_prefers_apptainer(self):
        """When both are available, apptainer should be preferred."""
        def which_both(name):
            return f"/usr/bin/{name}" if name in {"apptainer", "singularity"} else None

        with patch("shutil.which", side_effect=which_both):
            assert _find_singularity_executable() == "apptainer"


    def test_raises_when_neither_found(self):
        """Must raise RuntimeError with install instructions."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Neither.*apptainer.*nor.*singularity"):
                _find_singularity_executable()


class TestEnsureSingularityAvailable:
    """_ensure_singularity_available preflight tests."""

    def test_returns_executable_on_success(self):
        """Returns the executable name when version check passes."""
        fake_result = MagicMock(returncode=0, stderr="")

        with patch("shutil.which", side_effect=lambda n: "/usr/bin/apptainer" if n == "apptainer" else None), \
             patch("subprocess.run", return_value=fake_result):
            assert _ensure_singularity_available() == "apptainer"


    def test_raises_when_not_installed(self):
        """Raises RuntimeError when neither executable exists."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Neither.*apptainer.*nor.*singularity"):
                _ensure_singularity_available()
