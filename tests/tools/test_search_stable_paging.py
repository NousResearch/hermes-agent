"""search_files offset paging is a slice of ONE ordered list (Claude Code 2.1.327 parity).

ripgrep walks directories in parallel, so without a sort the first N lines behind
``| head -n`` differ from run to run and an ``offset`` page repeated or skipped hits
from the previous page.
"""

import os
import shutil

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="needs ripgrep")


@pytest.fixture
def wide_tree(tmp_path):
    root = tmp_path / "tree"
    for d in range(16):
        sub = root / f"d{d:02d}"
        sub.mkdir(parents=True)
        for f in range(12):
            (sub / f"f{f:02d}.txt").write_text("needle one\nneedle two\n")
    for i, entry in enumerate(sorted(root.rglob("*"))):  # newest-first = reverse path order
        os.utime(entry, (1_700_000_000 + i, 1_700_000_000 + i))  # mtime order != path order
    return root


def _pages(fetch, total, page):
    rows = []
    for offset in range(0, total, page):
        rows.extend(fetch(offset))
    return rows


@pytest.mark.parametrize("target,glob", [("content", None), ("files", "*.txt"), ("files", "*")])
def test_offset_pages_tile_one_ordered_list(wide_tree, target, glob):
    """``*`` also matches the 16 directories, which are merged into the file list."""
    ops = ShellFileOperations(LocalEnvironment(cwd=str(wide_tree)), cwd=str(wide_tree))
    total = 16 * 12 * (2 if target == "content" else 1) + (16 if glob == "*" else 0)

    def fetch(offset, limit=25):
        if target == "files":
            return ops.search(glob, str(wide_tree), target="files", limit=limit, offset=offset).files
        res = ops.search("needle", str(wide_tree), target="content", limit=limit, offset=offset)
        return [(m.path, m.line_number) for m in res.matches]

    full = fetch(0, limit=total)
    assert len(full) == total
    for _ in range(3):
        paged = _pages(fetch, total, 25)
        assert paged == full  # no repeats, no skips, same order every run
