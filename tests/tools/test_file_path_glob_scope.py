"""Path-aware globs retain roots, pagination and explicit ordering."""
import os
import shutil

import pytest

from tools.file_operations import ShellFileOperations
from tests.tools.test_file_operations import make_real_subprocess_env


@pytest.mark.parametrize("order", ["discovery", "modified"])
@pytest.mark.parametrize("multiple_roots", [False, True])
def test_path_glob_preserves_scope_and_pagination(tmp_path, order, multiple_roots):
    roots = [tmp_path / "first", tmp_path / "second"]
    wanted = []
    for i, root in enumerate(roots):
        for name in ("reference/a.txt", "reference/b.txt", "other/a.txt", ".hidden/reference/a.txt"):
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("fixture", encoding="utf-8")
            os.utime(file, (100 + len(wanted), 100 + len(wanted)))
            if name.startswith("reference/"):
                wanted.append(str(file))
    assert shutil.which("rg"), "This integration regression requires ripgrep"
    ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
    scope = [str(r) for r in roots] if multiple_roots else str(roots[0])
    expected = wanted if multiple_roots else wanted[:2]
    full = ops._search_files("**/reference/**", scope, limit=50, offset=0, order=order)
    assert full.error is None
    assert set(full.files) == set(expected)
    if order == "modified":
        assert full.files == list(reversed(expected))
        page = ops._search_files("**/reference/**", scope, limit=1, offset=1, order=order)
        assert page.error is None
        assert page.files == full.files[1:2]
    basename = ops._search_files("*.txt", scope, limit=50, offset=0, order=order)
    assert basename.error is None
    assert set(expected) < set(basename.files)
