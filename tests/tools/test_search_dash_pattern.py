"""search_files: a pattern or root that starts with "-" is data, never parsed as a flag.

rg and grep used to get the pattern as a bare positional word, so searching for a CLI
flag like ``--force`` failed with "unrecognized flag", ``-v`` silently inverted the
search, and ``--pre=<cmd>`` made rg run ``<cmd>`` on every file it searched. A root
such as ``-x`` hit the same problem in rg and in the find-based fallbacks.
"""

import os
import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "cli.py").write_text('parser.add_argument("--force")\nparser.add_argument("-v")\n')
    (root / "notes.txt").write_text("plain text\n")
    return root


def _ops(root, engine, monkeypatch):
    if shutil.which(engine) is None:
        pytest.skip(f"{engine} not installed")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))
    monkeypatch.setattr(ops, "_has_command", lambda command: command in (engine, "find"))
    return ops


@pytest.mark.parametrize("engine", ["rg", "grep"])
@pytest.mark.parametrize(("pattern", "line"), [("--force", 1), ("-v", 2)])
def test_dash_pattern_matches_literally(proj, monkeypatch, engine, pattern, line):
    result = _ops(proj, engine, monkeypatch).search(pattern, path=str(proj))

    assert result.error is None
    assert [(os.path.basename(m.path), m.line_number) for m in result.matches] == [("cli.py", line)]


def test_zero_match_probe_handles_dash_pattern(proj, monkeypatch):
    result = _ops(proj, "rg", monkeypatch).search("--FORCE", path=str(proj))

    assert result.error is None
    assert result.total_count == 0
    assert "case-insensitive" in (result.warning or "")
    assert "cli.py" in result.warning


@pytest.mark.platforms("posix")  # the preprocessor is a shell script
def test_pattern_cannot_set_an_rg_preprocessor(proj, tmp_path, monkeypatch):
    marker = tmp_path / "preprocessor-ran"
    script = tmp_path / "pre.sh"
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\ncat \"$1\"\n")
    script.chmod(0o755)

    _ops(proj, "rg", monkeypatch).search(f"--pre={script}", path=str(proj))

    assert not marker.exists()


# cwd under a dot-dir sends grep down its find-rooted fallback; file search merges
# rg's files with directories (here: the empty one) from a find run.
@pytest.mark.parametrize(("engine", "target", "pattern", "expected"), [
    ("rg", "content", "needle", ["a.py"]),
    ("grep", "content", "needle", ["a.py"]),
    ("rg", "files", "*", ["a.py", "emptydir"]),
], ids=["rg-content", "grep-find-fallback", "rg-files"])
def test_dash_root_is_searched(tmp_path, monkeypatch, engine, target, pattern, expected):
    cwd = tmp_path / ".work" / "proj"
    (cwd / "-x" / "emptydir").mkdir(parents=True)
    (cwd / "-x" / "a.py").write_text("needle = 1\n")

    result = _ops(cwd, engine, monkeypatch).search(pattern, path="-x", target=target)

    assert result.error is None
    found = [m.path for m in result.matches] if target == "content" else result.files
    assert sorted(os.path.basename(path) for path in found) == expected
    if target == "files":
        # One path form for both lanes, so the merged list can be stat'ed and sorted.
        assert all(path.startswith("./-x/") for path in found)
