"""Regression tests for search_files total_count under-reporting (#79530).

The rg search pipeline pipes match output through ``| head -n fetch_limit``
to bound memory, but ``total_count`` was derived from the *fetched* rows
only — so any search with more matches than the fetch ceiling reported a
total that was too small (e.g. 5 of 1300 real matches). The fix runs a
dedicated untruncated count query (``rg --count-matches`` /
``rg -l | wc -l``) when the fetch hit its ceiling and uses that as
``total_count``.
"""

import os
import subprocess

import pytest

from tools.file_operations import ShellFileOperations


def _real_shell_env(cwd):
    """A terminal env that actually runs commands (matches test_file_operations)."""

    def execute(command, **kwargs):
        # The search pipeline uses `set -o pipefail`, a bash builtin — run
        # under bash like the real terminal env (tools/environments/local.py
        # executes via bash), not /bin/sh (dash on Debian/Ubuntu rejects it).
        completed = subprocess.run(
            ["/bin/bash", "-c", command],
            text=True,
            capture_output=True,
            cwd=kwargs.get("cwd") or cwd,
        )
        return {"output": completed.stdout + completed.stderr, "returncode": completed.returncode}

    return type("Env", (), {"cwd": cwd, "execute": staticmethod(execute)})()


@pytest.fixture()
def big_dir(tmp_path):
    """A directory with more matching files than the fetch ceiling."""
    for i in range(1300):
        (tmp_path / f"file_{i}.md").write_text(f"email: pessoa{i}@teste.com\n")
    return tmp_path


def test_content_mode_total_count_is_true_total(big_dir):
    """Default content mode: total_count reflects all 1300 matches, page is 5."""
    s = ShellFileOperations(_real_shell_env(str(big_dir)))
    r = s._search_content("email", str(big_dir), None, 5, 0, "content", 0)

    assert len(r.matches) == 5
    assert r.total_count == 1300
    assert r.truncated is True


def test_files_only_mode_total_count_is_true_total(big_dir):
    s = ShellFileOperations(_real_shell_env(str(big_dir)))
    r = s._search_content("email", str(big_dir), None, 5, 0, "files_only", 0)

    assert len(r.files) == 5
    assert r.total_count == 1300
    assert r.truncated is True


def test_count_mode_total_count_is_true_total(big_dir):
    s = ShellFileOperations(_real_shell_env(str(big_dir)))
    r = s._search_content("email", str(big_dir), None, 5, 0, "count", 0)

    assert r.total_count == 1300
    assert r.truncated is True


def test_small_dir_total_count_unchanged(tmp_path):
    """Below the fetch ceiling, total_count is exact without a count query."""
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(f"needle {i}\n")

    s = ShellFileOperations(_real_shell_env(str(tmp_path)))
    r = s._search_content("needle", str(tmp_path), None, 5, 0, "content", 0)

    assert r.total_count == 3
    assert r.truncated is False


def test_limit_below_total_still_truncated(tmp_path):
    """A small dir with limit < total: truncated=True, total exact."""
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(f"needle {i}\n")

    s = ShellFileOperations(_real_shell_env(str(tmp_path)))
    r = s._search_content("needle", str(tmp_path), None, 2, 0, "content", 0)

    assert r.total_count == 3
    assert r.truncated is True
