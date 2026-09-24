"""Regression tests for local terminal initial cwd normalization."""

import os
from pathlib import Path

from tools.environments.local import (
    LocalEnvironment,
    _msys_to_windows_path,
    _resolve_local_initial_cwd,
)


def _same_dir(reported: str, expected: Path) -> bool:
    """True when the shell's ``pwd`` names *expected*.

    The terminal runs under Git Bash on Windows, where ``pwd`` reports the MSYS
    spelling (``/c/Users/...``) of the very same directory. That is a spelling
    difference, not a different cwd, so translate it with the production
    helper (a no-op off Windows) before comparing, and normcase for the
    case-insensitive filesystem.
    """
    native = _msys_to_windows_path(reported)
    return os.path.normcase(os.path.realpath(native)) == os.path.normcase(
        os.path.realpath(str(expected))
    )


def test_relative_initial_cwd_resolves_from_parent(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    assert _resolve_local_initial_cwd("hermes-agent") == str(project)


def test_local_environment_keeps_existing_relative_child_cwd(tmp_path, monkeypatch):
    project = tmp_path / "hermes-agent"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    env = LocalEnvironment(cwd="hermes-agent", timeout=5)
    try:
        result = env.execute("pwd", timeout=5)
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    reported = result["output"].strip()
    assert _same_dir(reported, project), reported
    # The bug this pins: the child cwd collapsing back to the PARENT.
    assert not _same_dir(reported, tmp_path), reported
