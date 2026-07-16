"""Tests for issue #65854 — uninstall can delete other packages from a
shared Python folder.

The fix adds _is_hermes_project_root() which verifies a directory is
actually a Hermes Agent project before allowing shutil.rmtree.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hermes_cli.uninstall import _is_hermes_project_root


# --------------------------------------------------------------------------- #
# _is_hermes_project_root
# --------------------------------------------------------------------------- #


def test_real_hermes_project_detected():
    """The actual hermes-agent repo should be detected as a Hermes project."""
    # The repo root is the parent of the hermes_cli directory
    from hermes_cli.uninstall import get_project_root
    repo_root = get_project_root()
    assert _is_hermes_project_root(repo_root) is True


def test_shared_python_folder_not_detected(tmp_path):
    """A random folder with other packages should NOT be detected as Hermes."""
    # Create a folder that looks like a shared Python folder
    (tmp_path / "some_package").mkdir()
    (tmp_path / "some_package" / "__init__.py").write_text("")
    (tmp_path / "other_package").mkdir()
    (tmp_path / "other_package" / "__init__.py").write_text("")

    assert _is_hermes_project_root(tmp_path) is False, (
        "A shared Python folder must not be detected as a Hermes project — see issue #65854"
    )


def test_empty_folder_not_detected(tmp_path):
    """An empty folder should not be detected as a Hermes project."""
    assert _is_hermes_project_root(tmp_path) is False


def test_folder_with_copied_uninstall_but_no_markers(tmp_path):
    """A folder with just a copied uninstall.py should not be detected.

    The issue mentions 'a copied uninstall module inside an unrelated Git
    repository can also pass the current safety check.' Our fix requires
    sibling markers (hermes_cli/ + run_agent.py) or pyproject.toml.
    """
    (tmp_path / "uninstall.py").write_text("# copied from hermes")
    (tmp_path / "README.md").write_text("# unrelated project")

    assert _is_hermes_project_root(tmp_path) is False


def test_folder_with_pyproject_containing_hermes_detected(tmp_path):
    """A folder with pyproject.toml containing 'hermes-agent' is detected."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "0.1.0"\n'
    )

    assert _is_hermes_project_root(tmp_path) is True


def test_folder_with_sibling_markers_detected(tmp_path):
    """A folder with hermes_cli/__init__.py + run_agent.py is detected."""
    (tmp_path / "hermes_cli").mkdir()
    (tmp_path / "hermes_cli" / "__init__.py").write_text("")
    (tmp_path / "run_agent.py").write_text("# Hermes entry point")

    assert _is_hermes_project_root(tmp_path) is True


def test_folder_with_only_hermes_cli_not_detected(tmp_path):
    """Only hermes_cli/ without run_agent.py should not be enough."""
    (tmp_path / "hermes_cli").mkdir()
    (tmp_path / "hermes_cli" / "__init__.py").write_text("")

    assert _is_hermes_project_root(tmp_path) is False


def test_folder_with_only_run_agent_not_detected(tmp_path):
    """Only run_agent.py without hermes_cli/ should not be enough."""
    (tmp_path / "run_agent.py").write_text("# something")

    assert _is_hermes_project_root(tmp_path) is False


def test_pyproject_without_hermes_not_detected(tmp_path):
    """A pyproject.toml for a different project should not be detected."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "some-other-project"\nversion = "1.0.0"\n'
    )

    assert _is_hermes_project_root(tmp_path) is False


def test_nonexistent_path_returns_false():
    """A nonexistent path should return False, not crash."""
    assert _is_hermes_project_root(Path("/nonexistent/path/that/does/not/exist")) is False