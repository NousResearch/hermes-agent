"""``collect_working_diff`` empties the clean/smudge filter commands a repository names.

Real temporary repositories with real ``git``: ``.gitattributes`` names a filter whose
repository-local command touches a marker file and then behaves as ``cat``. A tracked file is
made stat-dirty so git must convert it again; the marker must stay absent while the diff keeps
its normal content.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_agent.hermes_cli._subprocess_compat import FILTER_DISCOVERY_FAILED
from hermes_agent.tools import working_diff
from hermes_agent.tools.working_diff import collect_working_diff

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

_STAMP = [1_000_000_000]


def _env(home: Path) -> dict:
    return {"HOME": str(home), "PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=_env(repo.parent))


def _make_repo(tmp_path: Path, *, with_filter: bool = True, via_include: bool = False) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "README").write_text("hello\n")
    (repo / "notes.txt").write_text("alpha\n")
    if with_filter:
        (repo / ".gitattributes").write_text("README filter=probe\n*.txt filter=probe\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    marker = tmp_path / "filter-ran"
    if with_filter:
        script = tmp_path / "probe-filter.sh"
        script.write_text(f"#!/bin/sh\ntouch '{marker}'\nexec cat\n")
        script.chmod(0o755)
        if via_include:
            (repo / ".git" / "probe-filters.inc").write_text(
                f'[filter "probe"]\n\tclean = {script}\n\tsmudge = {script}\n')
            _git(repo, "config", "include.path", "probe-filters.inc")
        else:
            _git(repo, "config", "filter.probe.clean", str(script))
            _git(repo, "config", "filter.probe.smudge", str(script))
    return repo, marker


def _stat_dirty(repo: Path, marker: Path) -> None:
    """New mtimes on both tracked files so git re-reads (and re-cleans) them; clear the marker."""
    _STAMP[0] += 10
    for name in ("README", "notes.txt"):
        os.utime(repo / name, (_STAMP[0], _STAMP[0]))
    if marker.exists():
        marker.unlink()


def _modify_notes(repo: Path, *, stage: bool = False) -> None:
    (repo / "notes.txt").write_text("alpha\nbeta\n")
    if stage:
        _git(repo, "add", "notes.txt")


@pytest.mark.parametrize("mode", ["working", "all"])
def test_working_and_all_modes_run_no_repository_filter(tmp_path, mode):
    repo, marker = _make_repo(tmp_path)
    _modify_notes(repo)
    _stat_dirty(repo, marker)

    result = collect_working_diff(str(repo), mode=mode)

    assert not marker.exists()
    assert result["success"] is True
    assert "+beta" in result["diff"]
    assert "notes.txt" in result["stat"]
    assert "README" not in result["stat"]


def test_staged_mode_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path)
    _modify_notes(repo, stage=True)
    _stat_dirty(repo, marker)

    result = collect_working_diff(str(repo), mode="staged")

    assert not marker.exists()
    assert result["success"] is True
    assert "+beta" in result["diff"]


def test_paths_restricted_diff_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path)
    _modify_notes(repo)
    _stat_dirty(repo, marker)

    result = collect_working_diff(str(repo), paths=["notes.txt", "README"])

    assert not marker.exists()
    assert result["success"] is True
    assert "+beta" in result["diff"]


def test_filter_defined_through_include_path_is_neutralized(tmp_path):
    repo, marker = _make_repo(tmp_path, via_include=True)
    _modify_notes(repo)
    _stat_dirty(repo, marker)

    result = collect_working_diff(str(repo))

    assert not marker.exists()
    assert result["success"] is True
    assert "+beta" in result["diff"]


def test_repository_without_filter_returns_the_same_diff_content(tmp_path):
    repo, _marker = _make_repo(tmp_path, with_filter=False)
    _modify_notes(repo)
    (repo / "new.txt").write_text("fresh\n")

    result = collect_working_diff(str(repo))

    assert result["success"] is True
    assert "-alpha" not in result["diff"]
    assert "+beta" in result["diff"]
    assert "+fresh" in result["diff"]
    assert result["untracked"] == ["new.txt"]
    assert result["stat"].splitlines()[0].strip().startswith("notes.txt |")


def test_outside_a_git_repository_reports_not_a_git_repository(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    assert collect_working_diff(str(plain)) == {"success": False, "error": "Not a git repository."}


def test_unbuildable_repository_environment_refuses_without_spawning_git(tmp_path, monkeypatch):
    repo, marker = _make_repo(tmp_path)
    _modify_notes(repo)
    _stat_dirty(repo, marker)
    spawned: list[list[str]] = []
    real_run = subprocess.run

    def counting_run(argv, *args, **kwargs):
        spawned.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(working_diff, "noninteractive_repo_git_env", lambda cwd, base=None: None, raising=False)
    monkeypatch.setattr(working_diff.subprocess, "run", counting_run)

    result = collect_working_diff(str(repo))

    assert result["success"] is False
    assert FILTER_DISCOVERY_FAILED in result["error"]
    assert "could not be run safely" in result["error"]
    assert not [argv for argv in spawned if "diff" in argv or "rev-parse" in argv]
    assert not marker.exists()
