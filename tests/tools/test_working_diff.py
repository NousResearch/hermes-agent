"""Tests for tools.working_diff.collect_working_diff — the git collection
layer shared by the CLI and gateway ``/diff`` command.

Runs against real temporary git repositories (no mocks) so the staged /
unstaged / untracked semantics are proven against actual git behaviour.
"""

import shutil
import subprocess

import pytest

import tools.working_diff as working_diff
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


def test_run_decodes_git_output_as_utf8(monkeypatch, repo):
    """``_run`` must force UTF-8 decoding of git's output.

    Without ``encoding="utf-8"`` (and a lossy ``errors=``), ``subprocess.run``
    falls back to the platform locale encoding. On Windows that's typically
    cp932: UTF-8 multibyte output (e.g. a Japanese filename or diff content)
    then either raises ``UnicodeDecodeError`` or silently decodes as mojibake,
    depending on the byte sequence. The raise violates the "Never raises on
    git failure" contract in ``_run``'s docstring.
    """
    captured = {}
    real_run = subprocess.run

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(working_diff.subprocess, "run", fake_run)

    working_diff._run(["status"], str(repo))

    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"


def test_non_ascii_untracked_file_does_not_raise(repo):
    """A non-ASCII filename + content must decode cleanly, not raise.

    Regression test for the UnicodeDecodeError observed on real Windows
    machines when git output contains UTF-8 multibyte sequences (Japanese
    filename/content here) and the platform locale is not UTF-8.
    """
    (repo / "日本語ファイル.py").write_text("x = 'こんにちは'\n", encoding="utf-8")

    result = collect_working_diff(str(repo))

    assert result["success"] is True
    assert any("日本語ファイル.py" in f for f in result["untracked"])
    assert "こんにちは" in result["diff"]


def test_cp932_content_is_lossy_but_never_raises(repo):
    """Non-UTF-8 blob content degrades to replacement characters, not a crash.

    Git emits blob bytes uninterpreted, so a repo whose files are cp932-encoded
    decodes lossily under the forced UTF-8 policy. That is the same trade-off
    checkpoint_manager's ``_run_git`` already makes (utf-8 + errors="replace"
    on every git call): a readable-but-lossy diff for legacy encodings, never
    an exception. This test pins the trade-off so it stays a documented choice.
    """
    legacy = repo / "legacy.txt"
    legacy.write_bytes("before=東京\n".encode("cp932"))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "legacy")
    legacy.write_bytes("after=日本語\n".encode("cp932"))

    result = collect_working_diff(str(repo))

    assert result["success"] is True
    assert "legacy.txt" in result["diff"]
    assert "\ufffd" in result["diff"]  # lossy by design, matching _run_git
