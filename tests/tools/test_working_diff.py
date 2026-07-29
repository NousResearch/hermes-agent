"""Tests for tools.working_diff.collect_working_diff — the git collection
layer shared by the CLI and gateway ``/diff`` command.

Runs against real temporary git repositories (no mocks) so the staged /
unstaged / untracked semantics are proven against actual git behaviour.
"""

import shutil
import subprocess

import pytest

from tools.working_diff import collect_working_diff

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required for working-diff tests"
)


def _git(repo, *args):
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
        env={"HOME": str(repo), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "PATH": __import__("os").environ["PATH"]},
    )


@pytest.fixture()
def repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q")
    (d / "tracked.py").write_text("print('hello')\n")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init")
    return d


def test_clean_repo_reports_empty(repo):
    result = collect_working_diff(str(repo))
    assert result["success"] is True
    assert result.get("empty") is True
    assert result["diff"] == ""


def test_unstaged_change_appears_in_default_mode(repo):
    (repo / "tracked.py").write_text("print('changed')\n")
    result = collect_working_diff(str(repo))
    assert result["success"] is True
    assert "-print('hello')" in result["diff"]
    assert "+print('changed')" in result["diff"]
    assert "tracked.py" in result["stat"]


def test_unknown_mode_rejected(repo):
    result = collect_working_diff(str(repo), mode="bogus")
    assert result["success"] is False
    assert "bogus" in result["error"]
